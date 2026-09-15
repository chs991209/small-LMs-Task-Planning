"""Domain configuration and client factories (Factory pattern).

Central place for per-domain paths and for building the OpenAI / Firestore clients,
so no other module hardcodes credentials or paths.
"""
import glob
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent  # repo root (planner/ -> ..)


@dataclass(frozen=True)
class DomainConfig:
    """Per-domain settings for the task planner."""
    name: str                 # "tabletop" | "kitchen"
    place: str                # dataset 'place' label
    template_path: Path       # action-steps prompt template
    dataset_path: Path        # COST dataset json
    fixed_objects: bool       # True -> template has (objects, command); False -> (command,)

    def read_template(self) -> str:
        return self.template_path.read_text()


_TPL = ROOT / "code/prompt_templates"
DOMAINS = {
    "tabletop": DomainConfig(
        name="tabletop",
        place="Block",
        template_path=_TPL / "fixed_objects (tabletop)/Action_steps_generating_prompt.txt",
        dataset_path=ROOT / "datasets/dataset_tabletop.json",
        fixed_objects=True,
    ),
    "kitchen": DomainConfig(
        name="kitchen",
        place="Kitchen",
        template_path=_TPL / "flexible_objects/Action_steps_generating_prompt.txt",
        dataset_path=ROOT / "datasets/dataset_kitchen.json",
        fixed_objects=False,
    ),
}


def get_domain(name: str) -> DomainConfig:
    if name not in DOMAINS:
        raise SystemExit(f"Unknown domain {name!r}; choose from {list(DOMAINS)}")
    return DOMAINS[name]


def openai_client():
    """Factory: an authenticated OpenAI client (key from env/.env)."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY (e.g. in .env) before running.")
    from openai import OpenAI
    return OpenAI()


def firestore_client():
    """Factory: an initialized Firestore client from a service-account key."""
    load_dotenv()
    import firebase_admin
    from firebase_admin import credentials, firestore

    if not firebase_admin._apps:
        cred = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not (cred and Path(cred).exists()):
            hits = glob.glob(str(ROOT / ".secrets" / "*firebase-adminsdk*.json"))
            if not hits:
                raise SystemExit("No Firebase service-account key (set GOOGLE_APPLICATION_CREDENTIALS or add one under .secrets/).")
            cred = hits[0]
        firebase_admin.initialize_app(credentials.Certificate(cred))
    return firestore.client()
