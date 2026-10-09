import json
from dataclasses import replace

import pytest

from decision.config import Settings
from decision.model import DecisionModel, EmbeddingDecisionModel, situation_text
from robot_sim.server import Simulation
from tests.fakes import FakeOpenAI

TOOLS = json.loads(Settings().tools_path.read_text())


@pytest.fixture
def settings(tmp_path):
    return replace(Settings(), cache_dir=tmp_path)


@pytest.fixture(scope="module")
def home_state():
    return Simulation().controller.get_state()


def test_decisions_api_sees_every_tool_and_returns_five_percent(settings, home_state):
    fake = FakeOpenAI()
    model = DecisionModel(fake, TOOLS, settings)
    result = model.shortlist("curl the index finger", home_state, ["1. move_palm_up({}) -> ok"])
    request = fake.decision_requests[0]
    assert len(request["questions"]) == 140 and all(q["type"] == "predicate" for q in request["questions"])
    assert all(t["description"] in request["input"] for t in TOOLS)        # whole catalog goes in
    assert "move_palm_up" in request["input"].split("Available tools:")[0]  # with the history as context
    assert model.k == 7 and len(result.tools) == 7 and len(set(result.tools)) == 7
    assert result.scores == sorted(result.scores, reverse=True)
    assert "curl_index" in result.tools
    assert result.usage["input_tokens"] > 0


def test_backend_is_configurable(settings):
    fake = FakeOpenAI()
    assert isinstance(DecisionModel.from_catalog_file(fake, settings), DecisionModel)
    embed = DecisionModel.from_catalog_file(fake, replace(settings, decision_backend="embedding"))
    assert isinstance(embed, EmbeddingDecisionModel)


def test_embedding_fallback_shortlists_and_caches(settings, home_state):
    first = FakeOpenAI()
    model = EmbeddingDecisionModel(first, TOOLS, settings)
    assert "curl_index" in model.shortlist("curl the index finger", home_state).tools
    second = FakeOpenAI()
    EmbeddingDecisionModel(second, TOOLS, settings).shortlist("wave")
    assert len(second.embedded_texts) == 1  # tool embeddings came from the cache


def test_situation_text_mentions_held_objects(home_state):
    state = json.loads(json.dumps(home_state))
    state["objects"]["red_cube"].update(held=True, lifted_cm=12.0, touching_hand_parts=["thumb", "index"])
    text = situation_text("put the cube in the zone", state, ["1. hand_pose_close_hand({}) -> ok"])
    assert "red cube" in text and "held=True" in text and "hand_pose_close_hand" in text
