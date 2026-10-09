"""Agent settings, read from the environment / .env."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    agent_model: str = os.getenv("AGENT_MODEL", "gpt-6-luna")
    agent_reasoning_effort: str | None = os.getenv("AGENT_REASONING_EFFORT") or None  # e.g. "low"
    max_steps: int = int(os.getenv("AGENT_MAX_STEPS", "30"))
