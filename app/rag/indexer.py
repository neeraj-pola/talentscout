# app/rag/indexer.py
"""Index CommonProfiles into ChromaDB (semantic) and BM25 (keyword).

ChromaDB stores embeddings + metadata for hybrid filter-then-search queries.
BM25 lives in memory for keyword recall — profiles where the criterion is
mentioned verbatim but maybe with low semantic similarity to the query.

The indexer is rebuilt per JD (fast — 60 profiles, ~2 seconds total).
Per-JD isolation means index state never leaks across JDs.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from app.config import settings
from app.models import CommonProfile
from app.obs.events import log_event
from app.obs.llm_client import embed


def _tokenize_for_bm25(text: str) -> list[str]:
    """Cheap whitespace tokenizer for BM25. Lowercased, punctuation stripped."""
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridIndex:
    """Holds Chroma collection + BM25 + raw lookup for a single JD."""

    def __init__(self, jd_id: str):
        self.jd_id = jd_id
        self.collection_name = f"profiles_{jd_id.replace('-', '_')}"
        self.chroma_path = Path(settings.chroma_path)
        self.chroma_path.mkdir(parents=True, exist_ok=True)

        self._chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # If a collection from a prior run for the same jd_id exists, blow it away.
        try:
            self._chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.collection = self._chroma_client.create_collection(
            name=self.collection_name,
            metadata={"jd_id": jd_id},
        )

        self._profiles_by_id: dict[str, CommonProfile] = {}
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []     # id at position i in BM25 corpus

    def index(self, profiles: list[CommonProfile]) -> None:
        """Add profiles to both stores. Idempotent for a fresh index instance."""
        if not profiles:
            log_event(self.jd_id, "rag.indexer", "no_profiles_to_index")
            return

        log_event(self.jd_id, "rag.indexer", "index_start", n_profiles=len(profiles))

        # 1. Embed
        texts = [p.raw_text for p in profiles]
        embeddings = embed(texts, jd_id=self.jd_id, agent="rag.indexer")

        # 2. Chroma write — embeddings + metadata for filtering
        self.collection.add(
            ids=[str(p.id) for p in profiles],
            embeddings=embeddings,
            documents=texts,
            metadatas=[self._build_metadata(p) for p in profiles],
        )

        # 3. BM25 corpus (in-memory)
        tokenized = [_tokenize_for_bm25(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_ids = [str(p.id) for p in profiles]

        # 4. Raw lookup
        self._profiles_by_id = {str(p.id): p for p in profiles}

        log_event(self.jd_id, "rag.indexer", "index_end",
                  n_indexed=len(profiles), chroma_collection=self.collection_name)

    @staticmethod
    def _build_metadata(p: CommonProfile) -> dict:
        """Metadata for Chroma filter clauses. All values must be primitives."""
        return {
            "candidate_id": str(p.id),
            "candidate_name": p.full_name,
            "source": p.source,
            "location": p.location or "",
            "years_experience": float(p.years_experience or 0),
            # skills as a single comma-delimited string for substring filter
            "skills_csv": "," + ",".join(p.skills) + "," if p.skills else ",",
        }

    # ----- read accessors used by retriever -----
    def chroma_query(
        self,
        query_text: str,
        top_k: int,
        where: dict | None = None,
    ) -> list[tuple[str, float]]:
        """Semantic query. Returns [(candidate_id, distance), ...] sorted asc."""
        # Embed the query — single call
        q_emb = embed([query_text], jd_id=self.jd_id, agent="rag.retriever")[0]
        kwargs = {
            "query_embeddings": [q_emb],
            "n_results": top_k,
        }
        if where:
            kwargs["where"] = where

        res = self.collection.query(**kwargs)
        ids = res["ids"][0]
        dists = res["distances"][0]
        return list(zip(ids, dists))

    def bm25_query(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        """Keyword query. Returns [(candidate_id, score), ...] sorted desc."""
        if self._bm25 is None or not self._bm25_ids:
            return []
        tokens = _tokenize_for_bm25(query_text)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(self._bm25_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def get_profile(self, candidate_id: str) -> CommonProfile | None:
        return self._profiles_by_id.get(candidate_id)

    def all_profiles(self) -> list[CommonProfile]:
        return list(self._profiles_by_id.values())

    def cleanup(self) -> None:
        """Drop the Chroma collection. Call between JDs to keep disk clean."""
        try:
            self._chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass