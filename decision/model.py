"""Decision model: shortlist the top fraction (default 5%) of robot tools for the next step of a task.

The decision model sees the whole catalog; only its shortlist is handed to the controlling LLM.

DecisionModel (default) uses the OpenAI Decisions API: the shared input is the current situation
(task, robot/object state, recent actions) plus every tool's description, and there is one yes/no
predicate per tool ("should the robot use this tool for its next step?"). Tools are ranked by the
returned probabilities.

EmbeddingDecisionModel is the cheap fallback: rank tools by embedding similarity to the situation.
"""

import json
import math
from dataclasses import dataclass, field

from decision.config import Settings
from decision.embeddings import ToolIndex


def situation_text(task: str, state: dict | None, history: list[str]) -> str:
    """Natural-language summary of where we are."""
    lines = [f"Task: {task}"]
    if history:
        lines.append("Actions so far:\n" + "\n".join(history[-8:]))
    if state:
        palm = state["palm"]
        lines.append(f"Palm at {palm['position']} facing {palm['facing']}, fingers pointing {palm['fingers_pointing']}.")
        curl = state.get("finger_curl", {})
        if curl:
            lines.append("Finger curl (0 straight .. 1 fist): " + ", ".join(f"{f} {c}" for f, c in curl.items()))
        for name, obj in state.get("objects", {}).items():
            status = [f"held={obj['held']}", f"lifted {obj['lifted_cm']} cm", f"in target zone={obj['in_target_zone']}"]
            if obj["touching_hand_parts"]:
                status.append("touching " + ", ".join(obj["touching_hand_parts"]))
            lines.append(f"{name.replace('_', ' ')} at {obj['position']}: " + ", ".join(status))
    return "\n".join(lines)


@dataclass
class Shortlist:
    tools: list[str]                 # the k selected tool names, best first
    scores: list[float]              # probability (Decisions API) or cosine similarity (embeddings)
    query: str                       # the situation text the decision was based on
    usage: dict = field(default_factory=dict)


def _k(settings: Settings, n_tools: int) -> int:
    return max(1, math.ceil(settings.shortlist_fraction * n_tools))


class DecisionModel:
    """Shortlist via the OpenAI Decisions API: one predicate per tool, ranked by probability."""

    QUESTION = "Is `{name}` a tool the robot should use for its next step toward the task?"

    def __init__(self, client, tools: list[dict], settings: Settings | None = None):
        self.client = client
        self.settings = settings or Settings()
        self.tools = {t["name"]: t for t in tools}
        self.k = _k(self.settings, len(tools))
        self.catalog = "\n".join(f"- {t['name']}: {t['description']}" for t in tools)
        self.questions = [{"type": "predicate", "name": t["name"], "instructions": self.QUESTION.format(name=t["name"])}
                          for t in tools]

    @classmethod
    def from_catalog_file(cls, client, settings: Settings | None = None):
        settings = settings or Settings()
        tools = json.loads(settings.tools_path.read_text())
        backend = EmbeddingDecisionModel if settings.decision_backend == "embedding" else cls
        return backend(client, tools, settings)

    def shortlist(self, task: str, state: dict | None = None, history: list[str] | None = None) -> Shortlist:
        situation = situation_text(task, state, history or [])
        decision = self.client.decisions.create(
            model=self.settings.decision_model,
            input=f"{situation}\n\nAvailable tools:\n{self.catalog}",
            questions=self.questions,
        )
        scores = {a.name: a.probability for a in decision.answers if a.type == "predicate"}
        order = list(self.tools)  # ties (common at p=1.0) keep catalog order
        ranked = sorted(order, key=lambda n: (-scores.get(n, 0.0), order.index(n)))[: self.k]
        usage = {"input_tokens": decision.usage.input_tokens, "output_tokens": decision.usage.output_tokens}
        return Shortlist(ranked, [scores.get(n, 0.0) for n in ranked], situation, usage)


class EmbeddingDecisionModel:
    """Fallback: shortlist the tools most similar (embeddings) to the situation text."""

    def __init__(self, client, tools: list[dict], settings: Settings | None = None):
        self.settings = settings or Settings()
        self.tools = {t["name"]: t for t in tools}
        self.k = _k(self.settings, len(tools))
        self.index = ToolIndex(client, tools, self.settings.embedding_model, self.settings.cache_dir)

    def shortlist(self, task: str, state: dict | None = None, history: list[str] | None = None) -> Shortlist:
        query = situation_text(task, state, history or [])
        top = self.index.search(query, self.k)
        return Shortlist([n for n, _ in top], [s for _, s in top], query)
