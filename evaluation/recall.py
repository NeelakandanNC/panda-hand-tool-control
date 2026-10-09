"""Offline check of the decision model alone: at each stage of each task (states recorded by running
a known-good plan), is a tool that can do the next step inside the shortlist?

    uv run python -m evaluation.recall
"""

import json

from openai import OpenAI

from agent.executors import InProcessExecutor
from decision.config import Settings
from decision.model import DecisionModel
from evaluation.tasks import TASKS
from robot_sim.server import Simulation
from tests.test_tasks import ORACLE

# Tools that are equally good substitutes for a step of the known-good plan.
EQUIVALENT = {
    "hand_pose_close_hand": {"hand_pose_close_hand", "hand_pose_rock", "hand_pose_grasp_hard", "hand_pose_grasp_sphere",
                             "hand_pose_grasp_soft"},
    "hand_pose_open_hand": {"hand_pose_open_hand", "hand_pose_paper"},
    "move_palm_up": {"move_palm_up"},
    "curl_index": {"curl_index", "set_index_curl", "index_knuckle_toward_palm", "index_upper_joints_toward_palm",
                   "index_knuckle_set_angle"},
}


def collect_stages():
    """[(task, state, history, acceptable tool names)] for every step of every task's known-good plan."""
    executor = InProcessExecutor(Simulation())
    stages = []
    for task in TASKS:
        executor.call("reset_scene", {})
        history = []
        for name, args in ORACLE[task.id]:
            stages.append((task, executor.state(), list(history), EQUIVALENT.get(name, {name})))
            executor.call(name, args)
            history.append(name)
    return stages


def recall_report(decider, stages) -> tuple[int, list[str]]:
    hits, misses = 0, []
    for task, state, history, ok in stages:
        picked = decider.shortlist(task.prompt, state, history).tools
        if ok & set(picked):
            hits += 1
        else:
            misses.append(f"{task.id} step {len(history) + 1}: needed {sorted(ok)[0]}, got {picked}")
    return hits, misses


def main() -> None:
    settings = Settings()
    llm = OpenAI()
    decider = DecisionModel(llm, json.loads(settings.tools_path.read_text()), settings)
    stages = collect_stages()
    hits, misses = recall_report(decider, stages)
    print(f"needed tool in shortlist at {hits}/{len(stages)} stages ({hits / len(stages):.0%})")
    for m in misses:
        print("  miss:", m)


if __name__ == "__main__":
    main()
