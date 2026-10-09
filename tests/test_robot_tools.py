import sys
import threading

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from robot_sim.client import SimError
from robot_sim.server import Simulation, start_tcp_server
from robot_tools.catalog import build_catalog
from robot_tools.mcp_server import run_tool


class InProcessClient:
    """SimClient stand-in that drives a Simulation in this thread (no TCP)."""

    def __init__(self, sim: Simulation):
        self.sim = sim

    def call(self, cmd, **args):
        try:
            return self.sim.run_command(cmd, **args)
        except ValueError as exc:
            raise SimError(str(exc)) from exc


@pytest.fixture(scope="module")
def sim():
    return Simulation()


@pytest.fixture(scope="module")
def catalog(sim):
    return {t.name: t for t in build_catalog(sim.run_command("describe"))}


def _run(sim, catalog, name, **args):
    result = run_tool(catalog[name], InProcessClient(sim), args)
    assert not result.is_error, result.content[0].text
    return sim.controller.get_state()


def test_every_actuator_has_its_own_tools(sim, catalog):
    for act in sim.run_command("describe")["actuators"]:
        tools = [t for t in catalog.values() if t.actuators == [act["name"]]]
        assert len(tools) == 3, act["name"]  # set angle, nudge +, nudge -


def test_every_tool_runs(sim, catalog):
    sample = {"degrees": 10, "fraction": 0.5, "x": 0.25, "y": 0.0, "z": 0.65}
    for spec in catalog.values():
        sim.run_command("reset")
        args = {k: sample[k] for k in spec.input_schema["required"]}
        if spec.name.endswith("_set_angle"):
            prop = spec.input_schema["properties"]["degrees"]
            args["degrees"] = (prop["minimum"] + prop["maximum"]) / 2
        result = run_tool(spec, InProcessClient(sim), args)
        assert not result.is_error, (spec.name, result.content[0].text)


def test_tool_names_match_what_happens(sim, catalog):
    sim.run_command("reset")
    before = sim.controller.get_state()
    after = _run(sim, catalog, "index_knuckle_toward_palm", degrees=40)
    assert after["finger_curl"]["index"] > before["finger_curl"]["index"] + 0.1
    assert after["finger_curl"]["middle"] == pytest.approx(before["finger_curl"]["middle"], abs=0.02)

    sim.run_command("reset")
    after = _run(sim, catalog, "arm_base_move_left", degrees=10)
    assert after["palm"]["position"][1] > before["palm"]["position"][1] + 0.03

    sim.run_command("reset")
    after = _run(sim, catalog, "move_palm_up", distance_cm=10)
    assert after["palm"]["position"][2] == pytest.approx(before["palm"]["position"][2] + 0.10, abs=0.01)

    sim.run_command("reset")
    after = _run(sim, catalog, "arm_flange_twist_turn_right", degrees=30)
    fingers = sim.data.site_xmat[sim.controller.palm_site].reshape(3, 3)[:, 2]
    assert fingers[1] < -0.3  # fingers now point toward the robot's right (-y)


def test_object_tools_pick_up_cube(sim, catalog):
    sim.run_command("reset")
    _run(sim, catalog, "move_palm_above_red_cube")
    _run(sim, catalog, "lower_palm_onto_red_cube")
    _run(sim, catalog, "hand_pose_close_hand")
    state = _run(sim, catalog, "move_palm_up", distance_cm=15)
    assert state["objects"]["red_cube"]["held"]


def test_unreachable_move_is_reported_as_error(sim, catalog):
    sim.run_command("reset")
    result = run_tool(catalog["move_palm_to_position"], InProcessClient(sim), {"x": 2.0, "y": 0, "z": 0.5})
    assert result.is_error


def test_mcp_stdio_round_trip():
    sim = Simulation()
    tcp = start_tcp_server(sim, port=0)
    threading.Thread(target=sim.run_headless, daemon=True).start()
    params = StdioServerParameters(command=sys.executable,
                                   args=["-m", "robot_tools.mcp_server", "--sim-port", str(tcp.server_address[1])])

    async def scenario():
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            result = await session.call_tool("move_palm_above_blue_ball", {"height_cm": 12})
            bad = await session.call_tool("hand_pose_point", {"no_such_arg": 1})
            return tools, result, bad

    try:
        tools, result, bad = anyio.run(scenario)
    finally:
        tcp.shutdown()
    assert len(tools) == len(build_catalog(sim.controller.describe()))
    assert not result.is_error and '"moved": true' in result.content[0].text
    assert bad.is_error
