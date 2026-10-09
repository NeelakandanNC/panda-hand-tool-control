"""Stand-ins for the OpenAI client used by the decision model and agent tests."""

import hashlib
import json
import re
from types import SimpleNamespace

import numpy as np


def _bow(text: str) -> np.ndarray:
    v = np.full(512, 1e-6)
    for word in re.findall(r"[a-z]+", text.lower()):
        v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 512] += 1
    return v / np.linalg.norm(v)


class FakeOpenAI:
    """Bag-of-words embeddings, a word-overlap `decisions.create`, and a scripted `responses.create` (the "LLM").

    `policy(offered_tool_names, prompt) -> (name, arguments)` decides each LLM turn.
    """

    def __init__(self, policy=None):
        self.embedded_texts: list[str] = []
        self.requests: list[dict] = []
        self.policy = policy
        self.decision_requests: list[dict] = []
        self.embeddings = SimpleNamespace(create=self._embed)
        self.decisions = SimpleNamespace(create=self._decide)
        self.responses = SimpleNamespace(create=self._create)

    def _embed(self, model, input):
        self.embedded_texts += input
        return SimpleNamespace(data=[SimpleNamespace(embedding=_bow(t).tolist()) for t in input])

    def _decide(self, model, input, questions):
        """Predicate probability = word overlap between the situation (before the catalog) and the tool name."""
        self.decision_requests.append({"model": model, "input": input, "questions": questions})
        situation = _bow(input.split("Available tools:")[0])
        answers = [SimpleNamespace(type="predicate", name=q["name"],
                                   probability=float(situation @ _bow(q["name"].replace("_", " "))))
                   for q in questions]
        return SimpleNamespace(answers=answers, usage=SimpleNamespace(input_tokens=200 * len(questions),
                                                                      output_tokens=0))

    def _create(self, **request):
        self.requests.append(request)
        offered = [t["name"] for t in request["tools"] if t["name"] != "finish"]
        choice = self.policy(offered, request["input"])
        if choice is None:
            return SimpleNamespace(output=[SimpleNamespace(type="message")], output_text="thinking...", usage=None)
        name, args = choice
        call = SimpleNamespace(type="function_call", name=name, arguments=json.dumps(args), call_id="c1")
        return SimpleNamespace(output=[call], output_text="", usage=SimpleNamespace(input_tokens=100 * len(request["tools"]), output_tokens=20))
