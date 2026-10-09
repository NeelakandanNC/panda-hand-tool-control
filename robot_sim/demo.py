"""Drive the running simulation through a pick-and-place over TCP (watch it in the viewer).

    scripts/mjpython.sh -m robot_sim.server     # terminal 1: the sim window
    uv run python -m robot_sim.demo             # terminal 2: this script
"""

from robot_sim.client import SimClient
from robot_sim.scene import TARGET_ZONE


def main() -> None:
    with SimClient() as sim:
        sim.call("reset")
        cube = sim.call("get_state")["objects"]["red_cube"]["position"]
        zone = TARGET_ZONE["pos"]
        above = cube[2] + 0.15
        steps = [
            ("approach the cube", "move_palm", {"position": [cube[0], cube[1], above], "palm": "down"}),
            ("lower onto it", "move_palm", {"position": cube}),
            ("close the hand", "hand_pose", {"name": "close_hand"}),
            ("lift", "move_palm", {"position": [cube[0], cube[1], above]}),
            ("carry to the target zone", "move_palm", {"position": [zone[0], zone[1], above]}),
            ("lower", "move_palm", {"position": [zone[0], zone[1], cube[2] + 0.01]}),
            ("release", "hand_pose", {"name": "open_hand"}),
            ("retreat", "move_palm", {"position": [zone[0], zone[1], above]}),
            ("wave goodbye", "hand_pose", {"name": "paper"}),
        ]
        for label, cmd, args in steps:
            result = sim.call(cmd, **args)
            cube_state = result["state"]["objects"]["red_cube"]
            print(f"{label:26} held={cube_state['held']!s:5} in_zone={cube_state['in_target_zone']!s:5} "
                  f"lifted={cube_state['lifted_cm']:5.1f} cm")


if __name__ == "__main__":
    main()
