# data/seed_expansion.py
"""One-shot seed-data expansion script.

Run with: python -m data.seed_expansion

What it does:
1. Backs up existing JSON files (.backup suffix)
2. Generates 120 new canonical candidates across 9 archetypes
3. Calls gpt-4o-mini once per canonical candidate to write narratives
4. Replicates cross-source candidates with source-flavored variants
5. Writes profiles back to the 3 source JSONs (existing seeds preserved)
6. Prints a validation summary

Why it exists: the original 60-candidate seed is ML-heavy and lorem-ipsum.
Real recruiter testing needs diversity (backend, frontend, devops, mobile),
realistic narratives, and cross-source duplication pressure for the dedup agent.
"""
from __future__ import annotations

import asyncio
import json
import random
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

from openai import AsyncOpenAI

from app.config import settings 


# ============================================================
# Config — deterministic with a fixed seed for reproducible runs
# ============================================================

SEED = 42
random.seed(SEED)

DATA_DIR = Path("data")
N_NEW_CANDIDATES = 120
LLM_MODEL = "gpt-4o-mini"
LLM_CONCURRENCY = 8


# ============================================================
# Reference data: skills, companies, schools, name pools
# ============================================================

ARCHETYPES = [
    ("backend_distributed",  25),
    ("frontend_specialist",  20),
    ("devops_sre",           18),
    ("data_engineering",     14),
    ("mobile",               12),
    ("ml_specialist_niche",  13),
    ("generalist_fullstack", 18),
    ("junior",               15),
    ("career_switcher",      10),
]
# Sanity check
assert sum(n for _, n in ARCHETYPES) >= N_NEW_CANDIDATES


# ----- skill pools (weighted) -----
SKILLS_BY_ARCHETYPE = {
    "backend_distributed": [
        ("Go", 0.6), ("Rust", 0.3), ("Java", 0.5), ("Python", 0.5),
        ("Kafka", 0.6), ("gRPC", 0.5), ("Kubernetes", 0.7), ("Docker", 0.8),
        ("PostgreSQL", 0.6), ("Redis", 0.5), ("AWS", 0.6), ("GCP", 0.3),
        ("microservices", 0.7), ("distributed systems", 0.8), ("observability", 0.4),
        ("Prometheus", 0.3), ("gRPC", 0.4), ("event-driven architecture", 0.4),
    ],
    "frontend_specialist": [
        ("React", 0.9), ("TypeScript", 0.85), ("JavaScript", 1.0),
        ("Next.js", 0.5), ("Vue.js", 0.3), ("CSS", 0.7), ("Tailwind", 0.6),
        ("design systems", 0.5), ("accessibility", 0.4), ("WCAG", 0.3),
        ("performance optimization", 0.4), ("webpack", 0.3), ("Vite", 0.4),
        ("Storybook", 0.4), ("React Native", 0.2), ("GraphQL", 0.4),
        ("Figma", 0.5), ("design tokens", 0.3),
    ],
    "devops_sre": [
        ("Kubernetes", 0.95), ("Terraform", 0.9), ("AWS", 0.8), ("GCP", 0.5),
        ("Azure", 0.4), ("Docker", 0.9), ("Helm", 0.6), ("Prometheus", 0.7),
        ("Grafana", 0.6), ("Datadog", 0.5), ("Jenkins", 0.5), ("GitHub Actions", 0.7),
        ("CI/CD", 0.9), ("Linux", 0.85), ("Bash", 0.7), ("Python", 0.6),
        ("on-call", 0.6), ("incident response", 0.5), ("Istio", 0.3),
        ("Ansible", 0.4), ("ArgoCD", 0.4),
    ],
    "data_engineering": [
        ("Python", 0.95), ("SQL", 1.0), ("Spark", 0.7), ("Airflow", 0.75),
        ("dbt", 0.6), ("Snowflake", 0.6), ("BigQuery", 0.5), ("Redshift", 0.4),
        ("Kafka", 0.5), ("Kinesis", 0.3), ("data modeling", 0.7),
        ("ETL", 0.85), ("data warehousing", 0.65), ("Databricks", 0.4),
        ("Delta Lake", 0.3), ("PostgreSQL", 0.5),
    ],
    "mobile": [
        ("Swift", 0.5), ("Kotlin", 0.5), ("React Native", 0.4),
        ("iOS", 0.5), ("Android", 0.5), ("SwiftUI", 0.4), ("Jetpack Compose", 0.3),
        ("Objective-C", 0.2), ("Java", 0.3), ("Flutter", 0.2),
        ("mobile architecture", 0.5), ("App Store", 0.4), ("Firebase", 0.5),
        ("CoreData", 0.3), ("RxSwift", 0.2), ("Coroutines", 0.3),
    ],
    "ml_specialist_niche": [
        ("Python", 1.0), ("PyTorch", 0.8), ("TensorFlow", 0.5),
        ("time series", 0.4), ("NLP", 0.5), ("computer vision", 0.3),
        ("transformers", 0.6), ("LLMs", 0.5), ("ranking systems", 0.3),
        ("recommender systems", 0.3), ("anomaly detection", 0.3),
        ("MLOps", 0.5), ("Prophet", 0.2), ("LSTM", 0.3), ("Hugging Face", 0.5),
        ("LangChain", 0.3), ("RAG", 0.3), ("model evaluation", 0.5),
        ("feature engineering", 0.5),
    ],
    "generalist_fullstack": [
        ("Python", 0.7), ("JavaScript", 0.85), ("TypeScript", 0.6),
        ("React", 0.75), ("Node.js", 0.65), ("PostgreSQL", 0.6),
        ("Docker", 0.6), ("AWS", 0.5), ("REST APIs", 0.85), ("GraphQL", 0.3),
        ("Django", 0.4), ("Flask", 0.4), ("Express", 0.4), ("Next.js", 0.4),
        ("Git", 0.95), ("system design", 0.4),
    ],
    "junior": [
        ("Python", 0.6), ("JavaScript", 0.7), ("HTML", 0.8), ("CSS", 0.7),
        ("React", 0.55), ("Git", 0.9), ("SQL", 0.6), ("Java", 0.4),
        ("data structures", 0.5), ("algorithms", 0.5), ("REST APIs", 0.5),
        ("debugging", 0.6),
    ],
    "career_switcher": [
        ("Python", 0.6), ("JavaScript", 0.55), ("React", 0.4), ("SQL", 0.5),
        ("Git", 0.85), ("HTML", 0.6), ("CSS", 0.55), ("data analysis", 0.4),
        ("Excel", 0.4), ("Tableau", 0.3), ("communication", 0.5),
        ("project management", 0.4),
    ],
}


# ----- companies (archetype-weighted) -----
COMPANIES_BY_ARCHETYPE = {
    "backend_distributed": [
        "Stripe", "Datadog", "Cloudflare", "Confluent", "MongoDB", "Snowflake",
        "Plaid", "Brex", "Ramp", "Notion", "Linear", "Vercel", "Supabase",
        "PlanetScale", "HashiCorp",
    ],
    "frontend_specialist": [
        "Linear", "Notion", "Vercel", "Figma", "Stripe", "Shopify", "Discord",
        "Loom", "Framer", "Airbnb", "Pinterest", "Asana", "Atlassian",
    ],
    "devops_sre": [
        "Datadog", "HashiCorp", "Snyk", "PagerDuty", "Chronosphere", "Honeycomb",
        "Grafana Labs", "Fastly", "Cloudflare", "Akamai", "GitLab", "AWS",
    ],
    "data_engineering": [
        "Snowflake", "Databricks", "dbt Labs", "Fivetran", "Airbyte",
        "Confluent", "Tecton", "Census", "Hightouch", "Mode Analytics",
    ],
    "mobile": [
        "Instagram", "WhatsApp", "Snapchat", "Cash App", "Robinhood",
        "DoorDash", "Lyft", "Uber", "Discord", "Spotify",
    ],
    "ml_specialist_niche": [
        "Anthropic", "OpenAI", "Hugging Face", "Cohere", "Mistral", "Pinecone",
        "Weights & Biases", "Scale AI", "Replicate", "Together AI",
    ],
    "generalist_fullstack": [
        "Linear", "Notion", "Airtable", "Webflow", "Retool", "Vercel",
        "Shopify", "Square", "DoorDash", "Instacart", "Wayfair",
    ],
    "junior": [
        "Big Tech Co.", "Startup Inc.", "ConsultingCo", "FinanceFirm",
        "Healthcare Systems", "EdTech Labs", "AdTech Corp",
    ],
    "career_switcher": [
        "Bootcamp Inc.", "Career Pivot Labs", "Startup Inc.", "Mid-Size SaaS",
        "Indie Studio", "Consulting Group",
    ],
}

# Previous companies (older roles tend to be smaller/less famous)
PREV_COMPANIES_INDIA = [
    "Infosys", "TCS", "Wipro", "HCL Technologies", "Tech Mahindra",
    "Cognizant", "Mindtree", "Persistent Systems", "Mphasis", "Hexaware",
    "Flipkart", "Zomato", "Swiggy", "PhonePe", "Razorpay", "Freshworks",
    "Zoho", "Postman", "InMobi", "Ola", "Paytm",
]
PREV_COMPANIES_US = [
    "Oracle", "Salesforce", "Adobe", "IBM", "Cisco", "VMware", "Dell",
    "HPE", "Intel", "AMD", "Qualcomm", "Microsoft", "Workday", "ServiceNow",
    "Atlassian", "Twilio", "Akamai", "F5", "Citrix",
]


# ----- schools -----
SCHOOLS_BY_PROFILE = {
    "us_top": [
        "MIT", "Stanford University", "UC Berkeley", "Carnegie Mellon",
        "Cornell", "Georgia Tech", "University of Washington",
        "UT Austin", "UCLA", "UCSD", "University of Michigan",
    ],
    "us_mid": [
        "Boston University", "Penn State", "Ohio State", "Texas A&M",
        "Rutgers", "Iowa State", "Arizona State", "Indiana University",
        "University at Buffalo", "Northeastern", "George Mason",
    ],
    "india_top": [
        "IIT Bombay", "IIT Delhi", "IIT Madras", "IIT Kanpur", "IIT Kharagpur",
        "IIT Roorkee", "IIIT Hyderabad", "BITS Pilani", "IISc Bangalore",
    ],
    "india_mid": [
        "NIT Trichy", "NIT Warangal", "VIT Vellore", "Manipal Institute",
        "PES University", "Anna University", "Delhi Technological University",
        "BIT Mesra", "JNTU Hyderabad",
    ],
    "bootcamp": [
        "Hack Reactor", "Lambda School", "App Academy", "Flatiron School",
        "General Assembly", "Le Wagon", "Coding Dojo",
    ],
}

DEGREES = {
    "us_top":    [("B.S. Computer Science", 0.4), ("M.S. Computer Science", 0.4), ("Ph.D. Computer Science", 0.1), ("M.S. Data Science", 0.1)],
    "us_mid":    [("B.S. Computer Science", 0.6), ("M.S. Computer Science", 0.3), ("B.S. Information Systems", 0.1)],
    "india_top": [("B.Tech Computer Science", 0.5), ("M.Tech CSE", 0.3), ("Dual Degree CSE", 0.2)],
    "india_mid": [("B.Tech CSE", 0.7), ("B.Tech IT", 0.2), ("MCA", 0.1)],
    "bootcamp":  [("Software Engineering Immersive (12 weeks)", 0.6), ("Full-Stack Web Development (16 weeks)", 0.4)],
}


# ----- locations -----
LOCATIONS_US = [
    "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
    "Boston, MA", "Chicago, IL", "Denver, CO", "Los Angeles, CA",
    "Portland, OR", "Atlanta, GA", "Remote",
]
LOCATIONS_INDIA = [
    "Bangalore, India", "Hyderabad, India", "Pune, India", "Chennai, India",
    "Mumbai, India", "Delhi NCR, India", "Gurgaon, India", "Noida, India",
    "Remote",
]


# ----- name pools (diverse) -----
NAMES_FIRST = {
    "western_male":    ["James", "Michael", "Christopher", "Daniel", "Matthew", "Andrew", "Joseph", "Ryan", "Brandon", "Tyler", "Marcus", "Jordan", "Connor", "Liam", "Noah"],
    "western_female":  ["Sarah", "Jennifer", "Jessica", "Ashley", "Amanda", "Emily", "Rachel", "Lauren", "Megan", "Olivia", "Sophia", "Madison", "Hannah", "Elizabeth", "Grace"],
    "indian_male":     ["Arjun", "Vikram", "Rohan", "Priyank", "Karthik", "Aditya", "Rahul", "Siddharth", "Anand", "Ravi", "Pradeep", "Suresh", "Vivek", "Aniket", "Manish"],
    "indian_female":   ["Priya", "Anjali", "Divya", "Pooja", "Neha", "Kavya", "Shruti", "Ritu", "Meera", "Aishwarya", "Lakshmi", "Sneha", "Radhika", "Anushka", "Tanvi"],
    "east_asian_male":   ["Wei", "Jun", "Hiroshi", "Kenji", "Daniel", "Kevin", "Andy", "Min-jun"],
    "east_asian_female": ["Mei", "Lin", "Yuki", "Sakura", "Jenny", "Grace", "Hye-jin"],
    "hispanic_male":     ["Carlos", "Diego", "Mateo", "Sebastian", "Alejandro", "Miguel", "Javier", "Luis"],
    "hispanic_female":   ["Sofia", "Isabella", "Camila", "Valentina", "Lucia", "Maria", "Ana"],
    "african_male":      ["Kwame", "Tunde", "Chidi", "Femi", "Jamal", "Amari", "DeShawn"],
    "african_female":    ["Adaeze", "Zainab", "Imani", "Aaliyah", "Maya"],
    "middle_eastern_male":   ["Omar", "Hassan", "Ali", "Tariq", "Yusuf", "Karim"],
    "middle_eastern_female": ["Layla", "Yasmin", "Noor", "Zara", "Amira"],
}

NAMES_LAST = {
    "western":       ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Wilson", "Anderson", "Taylor", "Thomas", "Moore", "Jackson", "Martin", "Lee", "Walker", "Hall", "Allen", "Hernandez", "King", "Wright", "Lopez", "Hill", "Scott", "Green", "Adams", "Baker", "Nelson", "Carter"],
    "indian":        ["Kumar", "Sharma", "Patel", "Singh", "Reddy", "Iyer", "Mehta", "Gupta", "Joshi", "Rao", "Nair", "Pillai", "Agarwal", "Chopra", "Saxena", "Verma", "Bhatt", "Khanna", "Tiwari", "Mishra"],
    "east_asian":    ["Chen", "Wang", "Li", "Zhang", "Liu", "Yang", "Kim", "Park", "Sato", "Tanaka", "Nakamura"],
    "hispanic":      ["Garcia", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Perez", "Sanchez", "Ramirez", "Torres", "Flores"],
    "african":       ["Adebayo", "Okonkwo", "Mensah", "Diallo", "Washington", "Jefferson", "Robinson"],
    "middle_eastern":["Rahman", "Hassan", "Khan", "Ahmed", "Ibrahim", "Aziz", "Malik"],
}


# ============================================================
# Data models
# ============================================================

@dataclass
class WorkExperience:
    title: str
    company: str
    duration_months: int
    description: str = ""  # Filled by LLM


@dataclass
class Education:
    degree: str
    school: str
    graduation_year: int


@dataclass
class Candidate:
    """The deterministic skeleton, BEFORE LLM augmentation."""
    canonical_id: str  # e.g. "person_087"
    name: str
    archetype: str
    yoe: float
    location: str
    skills: list[str]
    current_title: str
    current_company: str
    work_history: list[WorkExperience]
    education: list[Education]
    appears_in: list[str]  # ["linkedin", "naukri", "ats"] — subset
    name_variants: dict[str, str | None]  # per-source name spelling
    has_employment_gap: bool = False
    employment_gap_reason: str = ""
    is_career_switcher: bool = False
    prior_career: str = ""

    # LLM-filled fields (set after stage 2)
    summary: str = ""  # For LinkedIn-flavored bio
    summary_naukri: str = ""  # Rushed Naukri voice
    summary_ats: str = ""  # Terse ATS voice
    role_bullets: list[list[str]] = field(default_factory=list)  # bullets per work_history entry


# ============================================================
# Stage 1: Skeleton generation (deterministic)
# ============================================================

def _weighted_sample(items: list[tuple[str, float]], k: int) -> list[str]:
    """Sample k unique items by weight."""
    pool = items.copy()
    chosen = []
    for _ in range(min(k, len(pool))):
        weights = [w for _, w in pool]
        if not weights:
            break
        idx = random.choices(range(len(pool)), weights=weights, k=1)[0]
        chosen.append(pool[idx][0])
        pool.pop(idx)
    return chosen


def _weighted_choice(items: list[tuple[str, float]]) -> str:
    return random.choices([x for x, _ in items], weights=[w for _, w in items], k=1)[0]


def _pick_name() -> tuple[str, str]:
    """Pick a first + last name. Diverse, US/India weighted."""
    # 50/50 US-style / India-style, with smaller share of other origins
    origin_pool = [
        ("western", 0.35), ("indian", 0.35), ("east_asian", 0.10),
        ("hispanic", 0.07), ("african", 0.07), ("middle_eastern", 0.06),
    ]
    origin = _weighted_choice(origin_pool)
    gender = random.choice(["male", "female"])
    first_key = f"{origin}_{gender}"
    if first_key not in NAMES_FIRST:
        first_key = "western_male"
    first = random.choice(NAMES_FIRST[first_key])
    last = random.choice(NAMES_LAST.get(origin, NAMES_LAST["western"]))
    return first, last


def _pick_location(archetype: str) -> str:
    """US/India only, weighted by archetype."""
    # Junior + career switchers slightly more US-weighted
    us_weight = 0.6 if archetype in ("junior", "career_switcher") else 0.5
    if random.random() < us_weight:
        return random.choice(LOCATIONS_US)
    return random.choice(LOCATIONS_INDIA)


def _pick_school_profile(location: str, archetype: str) -> str:
    """Pick a school profile key based on location + archetype."""
    is_india = "India" in location
    is_career_switcher = archetype == "career_switcher"
    if is_career_switcher and random.random() < 0.5:
        return "bootcamp"
    if is_india:
        return "india_top" if random.random() < 0.4 else "india_mid"
    return "us_top" if random.random() < 0.35 else "us_mid"


def _generate_career_trajectory(archetype: str, yoe: float, current_company: str) -> tuple[str, list[WorkExperience]]:
    """Build a current title + list of past roles.

    Junior candidates have 1-2 roles. Mid (3-7 YOE) have 2-3. Senior (8+) have 3-4.
    """
    work_history = []

    if archetype == "junior":
        seniority_titles = ["Software Engineer", "Junior Software Engineer", "Associate Engineer"]
        current_title = random.choice(seniority_titles)
        n_prev = random.choice([0, 1])  # 0-1 previous roles
    elif archetype == "career_switcher":
        current_title = random.choice(["Software Engineer", "Junior Software Engineer"])
        n_prev = random.choice([0, 1])
    else:
        # Title progression based on YOE
        if yoe < 3:
            current_title = _archetype_title(archetype, level="mid")
            n_prev = 1
        elif yoe < 7:
            current_title = _archetype_title(archetype, level="senior")
            n_prev = random.choice([2, 3])
        elif yoe < 11:
            current_title = _archetype_title(archetype, level="staff")
            n_prev = random.choice([2, 3])
        else:
            current_title = _archetype_title(archetype, level="principal")
            n_prev = random.choice([3, 4])

    # Current role: 12-48 months
    current_months = random.randint(12, 48)
    work_history.append(WorkExperience(
        title=current_title,
        company=current_company,
        duration_months=current_months,
    ))

    # Previous roles
    for i in range(n_prev):
        prev_pool = (PREV_COMPANIES_INDIA + PREV_COMPANIES_US)
        prev_company = random.choice(prev_pool)
        prev_title = _archetype_title(archetype, level="mid" if i == 0 else "junior")
        prev_months = random.randint(14, 36)
        work_history.append(WorkExperience(
            title=prev_title,
            company=prev_company,
            duration_months=prev_months,
        ))

    return current_title, work_history


def _archetype_title(archetype: str, level: str) -> str:
    """Title for an archetype at a seniority level."""
    base_by_archetype = {
        "backend_distributed":   "Backend Engineer",
        "frontend_specialist":   "Frontend Engineer",
        "devops_sre":            "Site Reliability Engineer",
        "data_engineering":      "Data Engineer",
        "mobile":                "Mobile Engineer",
        "ml_specialist_niche":   "Machine Learning Engineer",
        "generalist_fullstack":  "Full-Stack Engineer",
        "junior":                "Software Engineer",
        "career_switcher":       "Software Engineer",
    }
    base = base_by_archetype.get(archetype, "Software Engineer")

    prefixes = {
        "junior":    "",
        "mid":       "",
        "senior":    "Senior ",
        "staff":     "Staff ",
        "principal": "Principal ",
    }
    return f"{prefixes.get(level, '')}{base}".strip()


def _generate_candidate(idx: int, archetype: str) -> Candidate:
    """One canonical candidate."""
    first, last = _pick_name()
    name = f"{first} {last}"

    # YOE distribution by archetype
    if archetype == "junior":
        yoe = round(random.uniform(0.5, 2.5), 1)
    elif archetype == "career_switcher":
        yoe = round(random.uniform(1.0, 3.0), 1)
    elif archetype == "ml_specialist_niche":
        yoe = round(random.uniform(4.0, 12.0), 1)
    else:
        yoe = round(random.uniform(2.5, 11.0), 1)

    location = _pick_location(archetype)

    # Skills: 8-15 from archetype pool, lowercase normalized
    skill_pool = SKILLS_BY_ARCHETYPE.get(archetype, [])
    n_skills = random.randint(8, 15)
    skills = _weighted_sample(skill_pool, n_skills)

    # Company + career trajectory
    company_pool = COMPANIES_BY_ARCHETYPE.get(archetype, ["TechCo"])
    current_company = random.choice(company_pool)
    current_title, work_history = _generate_career_trajectory(archetype, yoe, current_company)

    # Education
    school_profile_key = _pick_school_profile(location, archetype)
    school = random.choice(SCHOOLS_BY_PROFILE[school_profile_key])
    degree = _weighted_choice(DEGREES[school_profile_key])
    grad_year = 2026 - int(yoe) - random.randint(0, 1)
    education = [Education(degree=degree, school=school, graduation_year=grad_year)]

    # Career switcher: prior career
    is_career_switcher = (archetype == "career_switcher")
    prior_career = ""
    if is_career_switcher:
        prior_career = random.choice([
            "high school math teacher", "marketing analyst", "financial consultant",
            "graphic designer", "mechanical engineer", "research assistant",
            "investment banker", "physiotherapist", "accountant",
        ])

    # 8% of candidates have an employment gap
    has_gap = random.random() < 0.08
    gap_reason = ""
    if has_gap:
        gap_reason = random.choice([
            "12 months of parental leave",
            "6 months of caregiving leave",
            "10 months pursuing graduate studies",
            "8 months after a company-wide layoff",
            "9 months of career break for personal reasons",
        ])

    return Candidate(
        canonical_id=f"person_{idx:03d}",
        name=name,
        archetype=archetype,
        yoe=yoe,
        location=location,
        skills=skills,
        current_title=current_title,
        current_company=current_company,
        work_history=work_history,
        education=education,
        appears_in=[],  # Filled below
        name_variants={"linkedin": None, "naukri": None, "ats": None},
        has_employment_gap=has_gap,
        employment_gap_reason=gap_reason,
        is_career_switcher=is_career_switcher,
        prior_career=prior_career,
    )


def _assign_sources(candidates: list[Candidate]) -> None:
    """Decide which sources each candidate appears in, with realistic distribution:
      60% in 1 source
      30% in 2 sources
      10% in 3 sources
    """
    n = len(candidates)
    n_in_three = int(n * 0.10)
    n_in_two = int(n * 0.30)

    indices = list(range(n))
    random.shuffle(indices)

    in_three = set(indices[:n_in_three])
    in_two = set(indices[n_in_three:n_in_three + n_in_two])

    sources = ["linkedin", "naukri", "ats"]
    for i, c in enumerate(candidates):
        if i in in_three:
            c.appears_in = sources.copy()
        elif i in in_two:
            c.appears_in = random.sample(sources, 2)
        else:
            c.appears_in = [random.choice(sources)]

        # Default name variant: same as canonical name in all assigned sources
        for src in c.appears_in:
            c.name_variants[src] = c.name


def _apply_name_variants(candidates: list[Candidate], n_variants: int = 8) -> None:
    """Pick N candidates that appear in multiple sources and give them slight
    name variations across sources. Stresses fuzzy matching in dedup.
    """
    multi_source = [c for c in candidates if len(c.appears_in) >= 2]
    chosen = random.sample(multi_source, min(n_variants, len(multi_source)))

    for c in chosen:
        first, *rest = c.name.split(" ")
        last = rest[-1] if rest else ""
        middle = rest[0] if len(rest) > 1 else None

        # Variant strategies — pick one
        strategy = random.choice(["middle_initial", "abbreviated_first", "swapped"])

        if strategy == "middle_initial":
            mi = random.choice(["A.", "J.", "M.", "K.", "L."])
            variants_pool = [
                c.name,                    # full
                f"{first} {mi} {last}",    # with middle initial
                f"{first[0]}. {last}",     # abbreviated
            ]
        elif strategy == "abbreviated_first":
            variants_pool = [
                c.name,
                f"{first[0]}. {last}",
                f"{first} {last}",
            ]
        else:  # swapped (e.g. Indian names sometimes "Last, First")
            variants_pool = [
                c.name,
                f"{last} {first}",
                c.name,
            ]

        # Assign variants to sources
        for i, src in enumerate(c.appears_in):
            c.name_variants[src] = variants_pool[i % len(variants_pool)]


# ============================================================
# Stage 2: LLM-augmented narratives
# ============================================================

NARRATIVE_PROMPT = """You are writing realistic professional profile content for a fictional software engineer candidate. The candidate is for use in test data — they are not a real person.

CANDIDATE SKELETON:
- Name: {name}
- Current title: {current_title} at {current_company}
- Years of experience: {yoe}
- Location: {location}
- Top skills: {skills}
- Career history: {career_summary}
- Education: {education_str}
{extras}

Generate THREE versions of this person's professional bio + bullet points, each in a different voice:

1. LINKEDIN VOICE (polished, ~80 words): A confident, professional summary that highlights impact. Mention specific technologies and achievements.

2. NAUKRI VOICE (rushed, ~60 words): A more abbreviated bio. Less polished, sometimes awkward phrasing. Focus on years of experience and skills.

3. ATS VOICE (terse, ~50 words): Like resume objective text. Skills-forward, minimal personality. No "I" pronouns.

Then generate 3 bullet points describing the work at their CURRENT role (concise, specific, technology-mentioning).

Return ONLY a JSON object with this structure:
{{
  "linkedin_summary": "...",
  "naukri_summary": "...",
  "ats_summary": "...",
  "current_role_bullets": ["bullet 1", "bullet 2", "bullet 3"]
}}"""


async def _augment_one(client: AsyncOpenAI, c: Candidate, sem: asyncio.Semaphore) -> None:
    """Generate narratives for one candidate."""
    async with sem:
        # Build extras
        extras_parts = []
        if c.has_employment_gap:
            extras_parts.append(f"- Has a {c.employment_gap_reason} in work history — mention this naturally")
        if c.is_career_switcher:
            extras_parts.append(f"- Previously worked as a {c.prior_career} before transitioning to engineering — mention this")
        extras = "\n".join(extras_parts) if extras_parts else ""

        career_summary = "; ".join(
            f"{w.title} at {w.company} ({w.duration_months}mo)" for w in c.work_history
        )
        education_str = ", ".join(
            f"{e.degree} from {e.school}" for e in c.education
        )

        prompt = NARRATIVE_PROMPT.format(
            name=c.name,
            current_title=c.current_title,
            current_company=c.current_company,
            yoe=c.yoe,
            location=c.location,
            skills=", ".join(c.skills[:10]),
            career_summary=career_summary,
            education_str=education_str,
            extras=extras,
        )

        try:
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.8,  # Variety matters here
            )
            data = json.loads(response.choices[0].message.content)
            c.summary = data.get("linkedin_summary", "")
            c.summary_naukri = data.get("naukri_summary", "")
            c.summary_ats = data.get("ats_summary", "")
            current_bullets = data.get("current_role_bullets", [])
            # Pad role_bullets to match work_history length
            c.role_bullets = [current_bullets]
            for _ in range(len(c.work_history) - 1):
                # Older roles get one generic bullet (no LLM call to save cost)
                c.role_bullets.append([])
        except Exception as e:
            print(f"  ⚠ Failed for {c.name}: {e}. Using fallback narrative.")
            c.summary = f"Experienced {c.current_title} with {c.yoe} years of experience."
            c.summary_naukri = f"{c.current_title} at {c.current_company}. {c.yoe}+ years exp."
            c.summary_ats = f"{c.current_title}. Skills: {', '.join(c.skills[:5])}."
            c.role_bullets = [[f"Worked on {c.skills[0]} systems."]] + [[] for _ in range(len(c.work_history) - 1)]


async def _augment_all(candidates: list[Candidate]) -> None:
    """Fan out narrative generation across all candidates."""
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    sem = asyncio.Semaphore(LLM_CONCURRENCY)
    tasks = [_augment_one(client, c, sem) for c in candidates]
    completed = 0
    for fut in asyncio.as_completed(tasks):
        await fut
        completed += 1
        if completed % 10 == 0:
            print(f"  ... {completed}/{len(candidates)} narratives generated")


# ============================================================
# Stage 3: Source-specific serialization
# ============================================================

def _to_linkedin(c: Candidate) -> dict:
    """Render the candidate as a LinkedIn-shaped profile."""
    name = c.name_variants.get("linkedin") or c.name
    parts = name.split(" ", 1)
    first, last = parts[0], parts[1] if len(parts) > 1 else ""

    return {
        "linkedin_id": f"li_{c.canonical_id}",
        "publicProfileUrl": f"https://linkedin.com/in/{first.lower()}-{last.lower().replace(' ', '-')}",
        "firstName": first,
        "lastName": last,
        "headline": f"{c.current_title} at {c.current_company}",
        "location": {"name": c.location},
        "summary": c.summary,
        "industry": "Computer Software",
        "connections": random.randint(150, 800),
        "positions": [
            {
                "title": w.title,
                "companyName": w.company,
                "durationMonths": w.duration_months,
                "description": " ".join(c.role_bullets[i]) if i < len(c.role_bullets) else "",
            }
            for i, w in enumerate(c.work_history)
        ],
        "educations": [
            {
                "schoolName": e.school,
                "degreeName": e.degree,
                "graduationYear": e.graduation_year,
            }
            for e in c.education
        ],
        "skills": [{"name": s} for s in c.skills],
        "yearsOfExperience": c.yoe,
        "_canonical_id": c.canonical_id,
    }


def _to_naukri(c: Candidate) -> dict:
    """Render as a Naukri-shaped profile (different schema)."""
    name = c.name_variants.get("naukri") or c.name

    return {
        "naukri_id": f"nk_{c.canonical_id}",
        "candidateName": name,
        "emailId": f"{name.lower().replace(' ', '.').replace('.', '')}@example.com",
        "mobile": f"+91-{random.randint(7000000000, 9999999999)}",
        "currentLocation": c.location,
        "totalExp": f"{c.yoe} years",
        "currentDesignation": c.current_title,
        "aboutSelf": c.summary_naukri,
        "keySkills": ", ".join(c.skills),  # Comma-separated string, NOT list
        "workEx": [
            {
                "designation": w.title,
                "organization": w.company,
                "tenure": f"{w.duration_months} months",
                "responsibilities": " ".join(c.role_bullets[i]) if i < len(c.role_bullets) else "",
            }
            for i, w in enumerate(c.work_history)
        ],
        "education": [
            {
                "course": e.degree,
                "university": e.school,
                "passingYear": str(e.graduation_year),
            }
            for e in c.education
        ],
        "_canonical_id": c.canonical_id,
    }


def _to_ats(c: Candidate) -> dict:
    """Render as an ATS-shaped profile (free-form blob style)."""
    name = c.name_variants.get("ats") or c.name

    bio = c.summary_ats
    if c.has_employment_gap:
        bio += f" Note: {c.employment_gap_reason} in work history."
    if c.is_career_switcher:
        bio += f" Career transition from {c.prior_career}."

    return {
        "ats_id": f"ats_{c.canonical_id}",
        "full_name": name,
        "email": f"{name.lower().replace(' ', '.').replace('.', '')}@example.com",
        "phone_number": f"+1-{random.randint(2000000000, 9999999999)}",
        "city": c.location,
        "tenure_years": c.yoe,
        "role": c.current_title,
        "bio": bio,
        "tags": c.skills,
        "work_history": [
            {
                "role": w.title,
                "employer": w.company,
                "months": w.duration_months,
                "summary": " ".join(c.role_bullets[i]) if i < len(c.role_bullets) else "",
            }
            for i, w in enumerate(c.work_history)
        ],
        "academics": [
            {
                "qualification": e.degree,
                "school": e.school,
                "year": e.graduation_year,
            }
            for e in c.education
        ],
        "_canonical_id": c.canonical_id,
    }


# ============================================================
# Main orchestration
# ============================================================

def backup_files() -> None:
    print("→ Backing up existing JSON files...")
    for src in ["linkedin", "naukri", "ats"]:
        f = DATA_DIR / f"{src}_profiles.json"
        if f.exists():
            backup = DATA_DIR / f"{src}_profiles.json.backup"
            shutil.copy(f, backup)
            print(f"   ✓ {f.name} → {backup.name}")


def get_next_index() -> int:
    """Look at existing files and find the highest person_NNN index."""
    max_idx = 0
    for src in ["linkedin", "naukri", "ats"]:
        f = DATA_DIR / f"{src}_profiles.json"
        if f.exists():
            data = json.loads(f.read_text())
            for p in data:
                cid = p.get("_canonical_id", "")
                if cid.startswith("person_"):
                    try:
                        idx = int(cid.split("_")[1])
                        max_idx = max(max_idx, idx)
                    except (ValueError, IndexError):
                        pass
    return max_idx + 1


def generate_skeletons(start_idx: int) -> list[Candidate]:
    """Stage 1: deterministic skeletons.

    Builds a flat list of archetypes (one entry per candidate slot),
    shuffles it, then truncates to N_NEW_CANDIDATES. Guarantees that
    even when the archetype counts sum higher than the target, every
    archetype gets at least its proportional share.
    """
    print(f"→ Generating {N_NEW_CANDIDATES} candidate skeletons...")

    # Build a flat archetype list scaled to N_NEW_CANDIDATES
    total = sum(n for _, n in ARCHETYPES)
    scale = N_NEW_CANDIDATES / total
    archetype_slots: list[str] = []
    for archetype, count in ARCHETYPES:
        scaled = max(1, round(count * scale))
        archetype_slots.extend([archetype] * scaled)

    # Top up or trim to exact target
    while len(archetype_slots) < N_NEW_CANDIDATES:
        archetype_slots.append(random.choice([a for a, _ in ARCHETYPES]))
    archetype_slots = archetype_slots[:N_NEW_CANDIDATES]
    random.shuffle(archetype_slots)

    candidates = []
    for i, archetype in enumerate(archetype_slots):
        candidates.append(_generate_candidate(start_idx + i, archetype))

    _assign_sources(candidates)
    _apply_name_variants(candidates, n_variants=8)
    return candidates

def write_to_sources(candidates: list[Candidate]) -> dict[str, int]:
    """Append candidates to each source JSON. Returns per-source counts written."""
    print("→ Writing candidates to source JSONs...")
    counts = {"linkedin": 0, "naukri": 0, "ats": 0}

    for src in ["linkedin", "naukri", "ats"]:
        f = DATA_DIR / f"{src}_profiles.json"
        existing = json.loads(f.read_text()) if f.exists() else []

        new_profiles = []
        for c in candidates:
            if src not in c.appears_in:
                continue
            if src == "linkedin":
                new_profiles.append(_to_linkedin(c))
            elif src == "naukri":
                new_profiles.append(_to_naukri(c))
            elif src == "ats":
                new_profiles.append(_to_ats(c))

        counts[src] = len(new_profiles)
        combined = existing + new_profiles
        f.write_text(json.dumps(combined, indent=2, default=str))
        print(f"   ✓ {f.name}: {len(existing)} existing + {len(new_profiles)} new = {len(combined)} total")

    return counts


def print_summary(candidates: list[Candidate], source_counts: dict[str, int]) -> None:
    print()
    print("=" * 60)
    print("EXPANSION SUMMARY")
    print("=" * 60)
    n = len(candidates)
    print(f"\nCanonical candidates generated: {n}")
    in_one = sum(1 for c in candidates if len(c.appears_in) == 1)
    in_two = sum(1 for c in candidates if len(c.appears_in) == 2)
    in_three = sum(1 for c in candidates if len(c.appears_in) == 3)
    print(f"  Appearing in 1 source:  {in_one}  ({in_one/n*100:.0f}%)")
    print(f"  Appearing in 2 sources: {in_two}  ({in_two/n*100:.0f}%)")
    print(f"  Appearing in 3 sources: {in_three}  ({in_three/n*100:.0f}%)")

    print(f"\nNew profiles written per source:")
    for src, count in source_counts.items():
        print(f"  {src:10s}: {count}")

    print(f"\nArchetype distribution:")
    by_archetype: dict[str, int] = {}
    for c in candidates:
        by_archetype[c.archetype] = by_archetype.get(c.archetype, 0) + 1
    for arch, count in sorted(by_archetype.items(), key=lambda x: -x[1]):
        print(f"  {arch:25s}: {count}")

    n_gaps = sum(1 for c in candidates if c.has_employment_gap)
    n_switchers = sum(1 for c in candidates if c.is_career_switcher)
    n_variants = sum(
        1 for c in candidates
        if len(c.appears_in) >= 2
        and len({v for v in c.name_variants.values() if v}) > 1
    )
    expected_merges = sum(len(c.appears_in) - 1 for c in candidates if len(c.appears_in) >= 2)

    print(f"\nRealism counters:")
    print(f"  Name variants (multi-source candidates): {n_variants}")
    print(f"  Employment gaps:                          {n_gaps}")
    print(f"  Career switchers:                         {n_switchers}")
    print(f"  Expected dedup merges (ground truth):     {expected_merges}")

    print()
    print("✓ Done. Restart the mock API server to pick up new data.")
    print("  ./scripts/run_mock_server.sh")


async def main_async() -> None:
    backup_files()
    start_idx = get_next_index()
    print(f"   Next person index will start at: person_{start_idx:03d}")
    candidates = generate_skeletons(start_idx)

    print(f"→ LLM-augmenting {len(candidates)} candidate narratives (gpt-4o-mini)...")
    await _augment_all(candidates)

    source_counts = write_to_sources(candidates)
    print_summary(candidates, source_counts)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()