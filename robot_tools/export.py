"""Write the tool catalog to tools.json (name, description, schema, family, actuators).

The decision model indexes this file. It is generated offline from the model itself, so no
simulation process needs to be running:

    uv run python -m robot_tools.export            # -> tools.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import mujoco

from robot_sim.controller import Controller
from robot_sim.scene import build_model
from robot_tools.catalog import build_catalog


def describe_offline() -> dict:
    model = build_model()
    return Controller(model, mujoco.MjData(model)).describe()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("tools.json"))
    args = parser.parse_args()
    catalog = build_catalog(describe_offline())
    args.out.write_text(json.dumps([t.to_json() for t in catalog], indent=1))
    families = Counter(t.family for t in catalog)
    print(f"wrote {len(catalog)} tools to {args.out}: {dict(families)}")


if __name__ == "__main__":
    main()
