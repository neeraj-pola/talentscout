# app/normalize/normalizer.py
"""Public normalization API. One function call per source."""
from app.models import CommonProfile
from app.normalize.linkedin import normalize_linkedin
from app.normalize.naukri import normalize_naukri
from app.normalize.ats import normalize_ats
from app.obs.events import log_event


_NORMALIZERS = {
    "linkedin": normalize_linkedin,
    "naukri": normalize_naukri,
    "ats": normalize_ats,
}


def normalize(source: str, raw: dict) -> CommonProfile:
    """Map one raw source-shaped dict to a CommonProfile."""
    fn = _NORMALIZERS.get(source)
    if fn is None:
        raise ValueError(f"Unknown source: {source!r}")
    return fn(raw)


def normalize_batch(
    raw_by_source: dict[str, list[dict]],
    jd_id: str | None = None,
) -> list[CommonProfile]:
    """Normalize the output of search_all_sources() into a flat list."""
    log_event(jd_id, "normalize", "start",
              counts={k: len(v) for k, v in raw_by_source.items()})

    out: list[CommonProfile] = []
    for source, raws in raw_by_source.items():
        for raw in raws:
            try:
                out.append(normalize(source, raw))
            except Exception as e:
                log_event(jd_id, "normalize", "skip_bad_profile",
                          source=source, error=str(e))

    log_event(jd_id, "normalize", "end", normalized_count=len(out))
    return out