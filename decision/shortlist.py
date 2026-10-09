"""Try the decision model from the command line.

    uv run python -m decision.shortlist "pick up the red cube"

Uses the live simulation's state if robot_sim.server is running, otherwise the home state.
"""

import argparse

from openai import OpenAI

from decision.model import DecisionModel


def current_state() -> dict:
    try:
        from robot_sim.client import SimClient

        with SimClient(timeout=2) as sim:
            return sim.call("get_state")
    except OSError:
        from robot_sim.server import Simulation

        return Simulation().controller.get_state()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task")
    args = parser.parse_args()

    model = DecisionModel.from_catalog_file(OpenAI())
    result = model.shortlist(args.task, current_state())
    print(f"top {model.k} of {len(model.tools)} tools for: {args.task!r}")
    for i, (name, score) in enumerate(zip(result.tools, result.scores), 1):
        print(f"  {i}. {name}  ({score:.3f})")


if __name__ == "__main__":
    main()
