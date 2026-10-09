import threading

import pytest

from robot_sim.client import SimClient, SimError
from robot_sim.scene import TABLE_HEIGHT, TARGET_ZONE
from robot_sim.server import Simulation, start_tcp_server


@pytest.fixture
def sim():
    return Simulation()


def test_home_pose_is_palm_down_fingers_forward(sim):
    palm = sim.controller.get_state()["palm"]
    assert palm["facing"] == "down"
    assert palm["fingers_pointing"] == "forward"


def test_move_palm_reaches_target(sim):
    result = sim.run_command("move_palm", position=[0.3, -0.2, TABLE_HEIGHT + 0.15], palm="down")
    assert result["moved"]
    reached = result["state"]["palm"]["position"]
    assert all(abs(a - b) < 0.01 for a, b in zip(reached, [0.3, -0.2, TABLE_HEIGHT + 0.15]))


def test_unreachable_target_is_refused_without_moving(sim):
    before = sim.controller.get_state()["palm"]["position"]
    result = sim.run_command("move_palm", position=[1.5, 0, 0.5])
    assert result["moved"] is False
    assert sim.controller.get_state()["palm"]["position"] == before


def test_move_actuators_clips_to_limits(sim):
    result = sim.run_command("move_actuators", targets={"rh_A_FFJ3": 5.0}, duration=0.5)
    assert "rh_A_FFJ3" in result["clipped_to_limits"]


def test_bad_arguments_raise(sim):
    with pytest.raises(ValueError):
        sim.run_command("hand_pose", name="no_such_pose")
    with pytest.raises(ValueError):
        sim.run_command("move_actuators", targets={"no_such_actuator": 0.1})


def test_pick_and_place_cube_into_target_zone(sim):
    cube = sim.controller.get_state()["objects"]["red_cube"]["position"]
    zone = TARGET_ZONE["pos"]

    sim.run_command("move_palm", position=[cube[0], cube[1], cube[2] + 0.15], palm="down")
    sim.run_command("move_palm", position=cube)
    sim.run_command("hand_pose", name="close_hand")
    lifted = sim.run_command("move_palm", position=[cube[0], cube[1], cube[2] + 0.15])["state"]["objects"]["red_cube"]
    assert lifted["held"] and lifted["lifted_cm"] > 10

    sim.run_command("move_palm", position=[zone[0], zone[1], cube[2] + 0.15])
    sim.run_command("move_palm", position=[zone[0], zone[1], cube[2] + 0.01])
    sim.run_command("hand_pose", name="open_hand")
    final = sim.run_command("move_palm", position=[zone[0], zone[1], cube[2] + 0.15])["state"]["objects"]["red_cube"]
    assert final["in_target_zone"] and not final["held"]


def test_tcp_round_trip():
    sim = Simulation()
    server = start_tcp_server(sim, port=0)
    port = server.server_address[1]
    threading.Thread(target=sim.run_headless, daemon=True).start()
    try:
        with SimClient(port=port) as client:
            assert len(client.call("describe")["actuators"]) == 27
            moved = client.call("hand_pose", name="point", duration=0.5)
            assert moved["state"]["finger_curl"]["middle"] > 0.5  # pointing curls the other fingers
            with pytest.raises(SimError):
                client.call("not_a_command")
    finally:
        server.shutdown()
