"""Every benchmark task must be solvable with catalog tools: run a known-good plan, check passes."""

import pytest

from agent.executors import InProcessExecutor
from evaluation.tasks import TASKS, TASKS_BY_ID
from robot_sim.server import Simulation

PLACE = [("move_palm_above_target_zone", {}), ("lower_palm_to_target_zone", {}),
         ("hand_pose_open_hand", {}), ("move_palm_up", {"distance_cm": 10})]
ORACLE = {
    "lift_cube": [("move_palm_above_red_cube", {}), ("lower_palm_onto_red_cube", {}),
                  ("hand_pose_close_hand", {}), ("move_palm_up", {"distance_cm": 15})],
    "cube_to_zone": [("move_palm_above_red_cube", {}), ("lower_palm_onto_red_cube", {}),
                     ("hand_pose_close_hand", {}), *PLACE],
    "ball_to_zone": [("move_palm_above_blue_ball", {}), ("lower_palm_onto_blue_ball", {}),
                     ("hand_pose_close_hand", {}), *PLACE],
    "cylinder_to_zone": [("move_palm_above_green_cylinder", {}), ("lower_palm_onto_green_cylinder",
                         {"height_offset_cm": 2}), ("hand_pose_close_hand", {}), *PLACE],
    "peace_sign": [("hand_pose_scissors", {})],
    "point": [("hand_pose_point", {})],
    "fist": [("hand_pose_close_hand", {})],
    "bend_index_only": [("curl_index", {})],
    "hover_ball": [("move_palm_above_blue_ball", {})],
    "palm_left": [("turn_palm_to_face_left", {})],
}


@pytest.fixture(scope="module")
def executor():
    return InProcessExecutor(Simulation())


def test_every_task_has_an_oracle():
    assert set(ORACLE) == set(TASKS_BY_ID)


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_task_is_solvable(executor, task):
    executor.call("reset_scene", {})
    assert not task.check(executor.state()), "task must not already be satisfied at the start"
    for name, args in ORACLE[task.id]:
        _, is_error = executor.call(name, args)
        assert not is_error, name
    assert task.check(executor.state())
