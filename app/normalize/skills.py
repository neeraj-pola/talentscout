# app/normalize/skills.py
"""Skill canonicalization.

Sources emit messy skill names: "Python", "python3", "Py", "Python 3" all mean
the same thing. We canonicalize to a single lowercase form before any matching
or scoring downstream.

This is a tiny alias table for the take-home — in production it would be a
maintained taxonomy (e.g. EMSI skill graph) or learned via embedding clustering.
"""

# Map of variant -> canonical form. All keys are lowercase for case-insensitive lookup.
SKILL_ALIASES: dict[str, str] = {
    # Python
    "python": "python", "python3": "python", "py": "python", "python 3": "python",
    # AWS
    "aws": "aws", "amazon web services": "aws",
    # Azure
    "azure": "azure", "ms azure": "azure", "microsoft azure": "azure",
    # ML
    "ml": "machine learning", "machine learning": "machine learning",
    # LLMs
    "llm": "llms", "llms": "llms", "large language models": "llms", "gpt": "llms",
    # RAG
    "rag": "rag",
    "retrieval-augmented generation": "rag",
    "retrieval augmented generation": "rag",
    # LangChain
    "langchain": "langchain",
    # Docker
    "docker": "docker", "containerization": "docker",
    # Kubernetes
    "kubernetes": "kubernetes", "k8s": "kubernetes",
    # SQL family
    "sql": "sql", "postgresql": "postgresql", "mysql": "mysql",
    # Spark
    "apache spark": "spark", "spark": "spark", "pyspark": "spark",
    # Time series
    "time series analysis": "time series",
    "time-series forecasting": "time series",
    "time series": "time series",
    # NLP
    "nlp": "nlp", "natural language processing": "nlp", "spacy": "spacy",
    # Frameworks
    "pytorch": "pytorch", "torch": "pytorch",
    "tensorflow": "tensorflow",
    "react": "react", "reactjs": "react", "react.js": "react",
    "fastapi": "fastapi",
    "java": "java", "spring boot": "spring", "spring": "spring",
    "go": "go", "golang": "go",
}


def canonicalize_skill(skill: str) -> str:
    """Normalize one skill string. Returns canonical lowercase form."""
    if not skill:
        return ""
    s = skill.strip().lower()
    return SKILL_ALIASES.get(s, s)  # unknown skills pass through as lowercase


def canonicalize_skills(skills: list[str]) -> list[str]:
    """Normalize a list of skills, dedup, drop empty."""
    seen = set()
    result = []
    for s in skills:
        c = canonicalize_skill(s)
        if c and c not in seen:
            seen.add(c)
            result.append(c)
    return result


def parse_skill_string(s: str, separator: str = ",") -> list[str]:
    """Parse Naukri's comma-separated skill string into a list."""
    if not s:
        return []
    return [x.strip() for x in s.split(separator) if x.strip()]