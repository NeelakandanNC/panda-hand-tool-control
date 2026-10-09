"""Run the robot agent on a task.

    scripts/mjpython.sh -m robot_sim.server          # terminal 1: simulation window (to watch)
    uv run python -m agent.run "pick up the red cube and put it in the yellow zone"

    uv run python -m agent.run "make a peace sign" --headless   # no window: starts its own sim

Each step the LLM is given the robot's MCP tools and the current state, and calls one tool.
A JSON log of every step is written to runs/.
"""

import argparse
import json
import threading
import time
from typing import Callable

from openai import OpenAI

from agent.executors import MCPExecutor
from agent.config import ROOT, Settings
from agent.loop import Agent, StepRecord
from robot_sim.server import DEFAULT_PORT


def start_headless_sim() -> tuple[int, Callable[[], None]]:
    """Start a private headless simulation with its own TCP port; returns (port, stop)."""
    from robot_sim.server import Simulation, start_tcp_server

    sim = Simulation()
    tcp = start_tcp_server(sim, port=0)
    stop_event = threading.Event()
    threading.Thread(target=sim.run_headless, args=(stop_event,), daemon=True).start()

    def stop() -> None:
        stop_event.set()
        tcp.shutdown()
        tcp.server_close()

    return tcp.server_address[1], stop


def print_step(record: StepRecord) -> None:
    mark = "✗" if record.is_error else "✓"
    args = json.dumps(record.arguments) if record.arguments else ""
    print(f"[{record.step:2}] {mark} {record.tool}{'(' + args + ')' if args else ''}")
    print(f"      -> {record.outcome}   (llm {record.llm_s:.1f}s, robot {record.tool_s:.1f}s, "
          f"{record.tokens_in:,} tokens in)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task")
    parser.add_argument("--headless", action="store_true", help="start a private headless simulation")
    parser.add_argument("--sim-port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--no-reset", action="store_true", help="start from the current scene instead of resetting")
    args = parser.parse_args()

    settings = Settings()
    llm = OpenAI()
    port, stop_sim = start_headless_sim() if args.headless else (args.sim_port, lambda: None)
    executor = MCPExecutor(port)
    try:
        if not args.no_reset:
            executor.call("reset_scene", {})
        agent = Agent(llm, executor, settings)
        print(f"task: {args.task}\nmodel: {settings.agent_model}; {len(agent.schemas)} MCP tools\n")
        result = agent.run(args.task, args.max_steps, on_step=print_step)
    finally:
        executor.close()
        stop_sim()

    print(f"\n{'SUCCESS' if result.success else 'NOT DONE'}: {result.summary}  "
          f"({len(result.steps)} steps, {result.seconds:.0f}s)")
    out = ROOT / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result.to_json(), indent=1))
    print(f"log: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
