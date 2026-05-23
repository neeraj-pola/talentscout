# data/seed.py
"""Generate synthetic profiles for LinkedIn, Naukri, and ATS sources.

Run once: `python -m data.seed`
Outputs:
  data/linkedin_profiles.json
  data/naukri_profiles.json
  data/ats_profiles.json

Design goals:
- ~40 profiles per source
- Different schemas across sources (realistic normalization challenge)
- ~30% overlap across sources (realistic dedup challenge)
- Mix of seniority, skills, locations
- Some candidates with clear must-have gaps (for screening demo)
"""
import json
import random
from pathlib import Path

from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

DATA_DIR = Path(__file__).parent

# Skill pool — realistic skill names, mix of canonical and variants
PYTHON_SKILLS = ["Python", "python3", "Py", "Python 3"]
AWS_SKILLS = ["AWS", "Amazon Web Services", "aws"]
AZURE_SKILLS = ["Azure", "MS Azure", "Microsoft Azure"]
ML_SKILLS = ["Machine Learning", "ML", "machine learning"]
LLM_SKILLS = ["LLMs", "Large Language Models", "GPT", "LLM"]
RAG_SKILLS = ["RAG", "Retrieval-Augmented Generation", "Retrieval Augmented Generation"]
LANGCHAIN_SKILLS = ["LangChain", "Langchain", "langchain"]
DOCKER_SKILLS = ["Docker", "docker", "Containerization"]
K8S_SKILLS = ["Kubernetes", "K8s", "kubernetes"]
SQL_SKILLS = ["SQL", "PostgreSQL", "MySQL"]
SPARK_SKILLS = ["Apache Spark", "Spark", "PySpark"]
TIMESERIES_SKILLS = ["Time Series Analysis", "Time-series forecasting", "Time series"]
NLP_SKILLS = ["NLP", "Natural Language Processing", "spaCy"]
PYTORCH_SKILLS = ["PyTorch", "Pytorch", "Torch"]
TF_SKILLS = ["TensorFlow", "tensorflow"]
REACT_SKILLS = ["React", "ReactJS", "React.js"]
FASTAPI_SKILLS = ["FastAPI", "fastapi"]
JAVA_SKILLS = ["Java", "Spring Boot", "Spring"]
GO_SKILLS = ["Go", "Golang"]

COMPANIES = [
    "Acme AI", "Innovatech", "DataDyne", "NeuroNet Systems", "Quantum Labs",
    "Stellar Software", "BlueOcean Tech", "Pinnacle ML", "Cortex Analytics",
    "Vertex Data", "Helix AI", "Apex Cloud", "Nimbus Analytics", "Forge AI",
    "Polaris Tech", "Synapse Labs", "Orion ML", "Lumen Data", "Catalyst AI",
    "Meridian Tech",
]
TITLES_BY_LEVEL = {
    "junior": ["Junior ML Engineer", "Associate Data Scientist", "ML Engineer I"],
    "mid":    ["ML Engineer", "Data Scientist", "AI Engineer", "Software Engineer (AI/ML)"],
    "senior": ["Senior ML Engineer", "Senior Data Scientist", "Senior AI Engineer",
               "Staff ML Engineer", "Lead Data Scientist"],
    "staff":  ["Staff ML Engineer", "Principal AI Engineer", "ML Architect"],
}
LOCATIONS = [
    "Bangalore, India", "Hyderabad, India", "Pune, India", "Delhi, India",
    "Mumbai, India", "Buffalo, NY", "New York, NY", "San Francisco, CA",
    "Seattle, WA", "Austin, TX", "Remote", "Toronto, Canada", "London, UK",
]
UNIVERSITIES = [
    "IIT Bombay", "IIT Delhi", "IIT Madras", "BITS Pilani", "NIT Trichy",
    "University at Buffalo", "Stanford University", "MIT", "Carnegie Mellon",
    "UC Berkeley", "Georgia Tech", "University of Washington",
]
DEGREES = ["B.Tech Computer Science", "M.S. Computer Science", "M.S. AI",
           "M.S. Data Science", "B.Tech IT", "M.Tech CSE"]


def pick_skills(level: str, focus: str = "ml") -> list[str]:
    base = []
    if focus == "ml":
        base += [random.choice(PYTHON_SKILLS), random.choice(ML_SKILLS)]
        if random.random() > 0.3:
            base += [random.choice(LLM_SKILLS)]
        if random.random() > 0.5:
            base += [random.choice(RAG_SKILLS), random.choice(LANGCHAIN_SKILLS)]
        if random.random() > 0.4:
            base += [random.choice(PYTORCH_SKILLS)]
        if random.random() > 0.5:
            base += [random.choice(NLP_SKILLS)]
        if random.random() > 0.3:
            base += [random.choice(AWS_SKILLS)] if random.random() > 0.5 else [random.choice(AZURE_SKILLS)]
        if random.random() > 0.5:
            base += [random.choice(SQL_SKILLS)]
        if level in ("senior", "staff") and random.random() > 0.4:
            base += [random.choice(K8S_SKILLS), random.choice(DOCKER_SKILLS)]
        if random.random() > 0.7:
            base += [random.choice(TIMESERIES_SKILLS)]
        if random.random() > 0.7:
            base += [random.choice(FASTAPI_SKILLS)]
    elif focus == "backend":
        base += [
            random.choice(PYTHON_SKILLS), random.choice(JAVA_SKILLS),
            random.choice(SQL_SKILLS), random.choice(DOCKER_SKILLS),
        ]
        if random.random() > 0.5:
            base += [random.choice(K8S_SKILLS)]
    return list(set(base))


def make_canonical_person(idx: int) -> dict:
    level = random.choices(
        ["junior", "mid", "senior", "staff"],
        weights=[0.15, 0.45, 0.30, 0.10],
    )[0]
    yoe = {"junior": (1, 3), "mid": (3, 6), "senior": (6, 12), "staff": (10, 18)}[level]
    years = round(random.uniform(*yoe), 1)
    name = fake.name()
    focus = random.choices(["ml", "backend"], weights=[0.75, 0.25])[0]
    skills = pick_skills(level, focus)
    location = random.choice(LOCATIONS)
    title = random.choice(TITLES_BY_LEVEL[level])
    n_jobs = random.randint(1, 4)

    experiences = []
    remaining_months = int(years * 12)
    for i in range(n_jobs):
        dur = max(6, remaining_months // max(1, n_jobs - i) + random.randint(-6, 6))
        dur = min(dur, remaining_months)
        if dur <= 0:
            break
        experiences.append({
            "title": title if i == 0 else random.choice(TITLES_BY_LEVEL[level if i == 0 else "mid"]),
            "company": random.choice(COMPANIES),
            "duration_months": dur,
            "description": fake.paragraph(nb_sentences=3),
        })
        remaining_months -= dur

    education = [{
        "degree": random.choice(DEGREES),
        "institution": random.choice(UNIVERSITIES),
        "graduation_year": 2026 - int(years) - random.randint(0, 2),
    }]

    return {
        "canonical_id": f"person_{idx:03d}",
        "name": name,
        "email": fake.email(),
        "phone": fake.phone_number(),
        "location": location,
        "years_experience": years,
        "level": level,
        "current_title": title,
        "skills": skills,
        "experiences": experiences,
        "education": education,
        "bio_short": fake.sentence(nb_words=12),
        "bio_long": " ".join(fake.paragraphs(nb=2)),
    }


def to_linkedin(person: dict) -> dict:
    return {
        "linkedin_id": "li_" + person["canonical_id"],
        "publicProfileUrl": f"https://linkedin.com/in/{person['name'].lower().replace(' ', '-')}",
        "firstName": person["name"].split()[0],
        "lastName": " ".join(person["name"].split()[1:]),
        "headline": person["current_title"] + " at " + (person["experiences"][0]["company"] if person["experiences"] else "Unknown"),
        "location": {"name": person["location"]},
        "summary": person["bio_long"],
        "industry": "Computer Software",
        "connections": random.randint(50, 500),
        "positions": [
            {
                "title": e["title"],
                "companyName": e["company"],
                "durationMonths": e["duration_months"],
                "description": e["description"],
            }
            for e in person["experiences"]
        ],
        "educations": [
            {
                "schoolName": ed["institution"],
                "degreeName": ed["degree"],
                "graduationYear": ed["graduation_year"],
            }
            for ed in person["education"]
        ],
        "skills": [{"name": s} for s in person["skills"]],
        "yearsOfExperience": person["years_experience"],
        "_canonical_id": person["canonical_id"],
    }


def to_naukri(person: dict) -> dict:
    return {
        "naukri_id": "nk_" + person["canonical_id"],
        "candidateName": person["name"],
        "emailId": person["email"],
        "mobile": person["phone"],
        "currentLocation": person["location"],
        "totalExp": f"{person['years_experience']} years",
        "currentDesignation": person["current_title"],
        "aboutSelf": person["bio_long"],
        "keySkills": ", ".join(person["skills"]),
        "workEx": [
            {
                "designation": e["title"],
                "organization": e["company"],
                "tenure": f"{e['duration_months']} months",
                "responsibilities": e["description"],
            }
            for e in person["experiences"]
        ],
        "education": [
            {
                "course": ed["degree"],
                "university": ed["institution"],
                "passingYear": str(ed["graduation_year"]),
            }
            for ed in person["education"]
        ],
        "_canonical_id": person["canonical_id"],
    }


def to_ats(person: dict) -> dict:
    return {
        "ats_id": "ats_" + person["canonical_id"],
        "full_name": person["name"],
        "email": person["email"],
        "phone_number": person["phone"],
        "city": person["location"],
        "tenure_years": person["years_experience"],
        "role": person["current_title"],
        "bio": person["bio_short"] + " " + person["bio_long"],
        "tags": person["skills"],
        "work_history": [
            {
                "role": e["title"],
                "employer": e["company"],
                "months": e["duration_months"],
                "summary": e["description"],
            }
            for e in person["experiences"]
        ],
        "academics": [
            {
                "qualification": ed["degree"],
                "school": ed["institution"],
                "year": ed["graduation_year"],
            }
            for ed in person["education"]
        ],
        "source_channel": random.choice(["referral", "career_site", "campus", "agency"]),
        "_canonical_id": person["canonical_id"],
    }


def main():
    people = [make_canonical_person(i) for i in range(60)]
    linkedin, naukri, ats = [], [], []
    overlap_counts = {"only_one": 0, "two_sources": 0, "all_three": 0}

    for p in people:
        n_sources = random.choices([1, 2, 3], weights=[0.45, 0.40, 0.15])[0]
        sources = random.sample(["li", "nk", "ats"], n_sources)
        if "li" in sources:
            linkedin.append(to_linkedin(p))
        if "nk" in sources:
            naukri.append(to_naukri(p))
        if "ats" in sources:
            ats.append(to_ats(p))
        overlap_counts[{1: "only_one", 2: "two_sources", 3: "all_three"}[n_sources]] += 1

    (DATA_DIR / "linkedin_profiles.json").write_text(json.dumps(linkedin, indent=2))
    (DATA_DIR / "naukri_profiles.json").write_text(json.dumps(naukri, indent=2))
    (DATA_DIR / "ats_profiles.json").write_text(json.dumps(ats, indent=2))

    print("=" * 60)
    print("SEED DATA GENERATED")
    print("=" * 60)
    print(f"Canonical people:  {len(people)}")
    print(f"  in 1 source:     {overlap_counts['only_one']}")
    print(f"  in 2 sources:    {overlap_counts['two_sources']}")
    print(f"  in 3 sources:    {overlap_counts['all_three']}")
    print()
    print(f"LinkedIn profiles: {len(linkedin)}")
    print(f"Naukri profiles:   {len(naukri)}")
    print(f"ATS profiles:      {len(ats)}")
    print()
    total_rows = len(linkedin) + len(naukri) + len(ats)
    print(f"Total rows across sources: {total_rows}")
    print(f"Expected unique people:    {len(people)}")
    print(f"Expected merges by dedup:  {total_rows - len(people)}")


if __name__ == "__main__":
    main()