from agent.executors import InProcessExecutor
from agent.loop import Agent
from robot_sim.server import Simulation
from tests.fakes import FakeOpenAI

PLAN = [
    ("move_palm_above_red_cube", {}),
    ("lower_palm_onto_red_cube", {}),
    ("hand_pose_close_hand", {}),
    ("move_palm_above_target_zone", {}),
    ("lower_palm_to_target_zone", {}),
    ("hand_pose_open_hand", {}),
    ("move_palm_up", {"distance_cm": 10}),
]


def scripted_policy(plan):
    def policy(offered, prompt):
        done = prompt.count(" -> ")  # number of history lines
        if done < len(plan):
            return plan[done]
        return "finish", {"success": True, "summary": "cube placed"}
    return policy


def test_agent_completes_pick_and_place_with_mcp_tools():
    sim = Simulation()
    llm = FakeOpenAI(scripted_policy(PLAN))
    result = Agent(llm, InProcessExecutor(sim)).run("put the red cube in the yellow zone")
    assert result.success and len(result.steps) == len(PLAN) + 1
    assert sim.controller.get_state()["objects"]["red_cube"]["in_target_zone"]
    for request in llm.requests:  # the LLM gets every robot tool + finish
        assert len(request["tools"]) == 141 and request["tools"][-1]["name"] == "finish"
        assert request["tool_choice"] == "required"


def test_unknown_tool_is_reported_and_run_continues():
    llm = FakeOpenAI(lambda offered, prompt: ("teleport_cube", {}) if " -> " not in prompt
                     else ("finish", {"success": False, "summary": "gave up"}))
    result = Agent(llm, InProcessExecutor(Simulation())).run("anything")
    assert result.steps[0].is_error and "unknown tool" in result.steps[0].outcome
    assert not result.success and result.steps[1].tool == "finish"


def test_missing_tool_call_and_step_limit():
    result = Agent(FakeOpenAI(lambda offered, prompt: None), InProcessExecutor(Simulation())).run("x", max_steps=3)
    assert not result.success and len(result.steps) == 3
    assert all(s.tool is None and s.is_error for s in result.steps)


def test_agent_over_real_mcp_server():
    from agent.executors import MCPExecutor
    from agent.run import start_headless_sim

    port, stop_sim = start_headless_sim()
    executor = MCPExecutor(port)
    try:
        assert len(executor.tool_schemas()) == 140
        result = Agent(FakeOpenAI(scripted_policy(PLAN)), executor).run("put the cube in the zone")
        assert result.success
        assert executor.state()["objects"]["red_cube"]["in_target_zone"]
    finally:
        executor.close()
        stop_sim()
