"""Settings for the decision model and agent, read from the environment / .env."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    decision_backend: str = os.getenv("DECISION_BACKEND", "decisions")  # "decisions" | "embedding"
    decision_model: str = os.getenv("DECISION_MODEL", "gpt-6-luna")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    agent_model: str = os.getenv("AGENT_MODEL", "gpt-6-luna")
    agent_reasoning_effort: str | None = os.getenv("AGENT_REASONING_EFFORT") or None  # e.g. "low"
    shortlist_fraction: float = float(os.getenv("SHORTLIST_FRACTION", "0.05"))
    max_steps: int = int(os.getenv("AGENT_MAX_STEPS", "30"))
    tools_path: Path = ROOT / "tools.json"
    cache_dir: Path = ROOT / ".cache"
