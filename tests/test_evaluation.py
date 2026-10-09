from agent.config import Settings
from evaluation.run import report, run_one
from evaluation.tasks import TASKS_BY_ID
from tests.fakes import FakeOpenAI


def fist_policy(offered, prompt):
    if " -> " in prompt:
        return "finish", {"success": True, "summary": "fist made"}
    return "hand_pose_close_hand", {}


def test_run_one_checks_the_simulator_and_reports():
    settings = Settings()
    good = run_one(TASKS_BY_ID["fist"], 0, FakeOpenAI(fist_policy), settings, max_steps=4)
    assert "error" not in good, good.get("traceback")
    assert good["success"] and good["steps"] == 1 and good["tokens_in"] > 0

    # The LLM claims success without doing anything: the simulator check must fail it.
    liar = run_one(TASKS_BY_ID["fist"], 1, FakeOpenAI(lambda o, p: ("finish", {"success": True, "summary": "done"})),
                   settings, max_steps=4)
    assert liar["agent_claimed_success"] and not liar["success"]

    text = report([good, liar], settings)
    assert "| fist | gesture | 1/2 |" in text and "claimed success but the check failed: 1" in text
