"""Benchmark tasks. Success is judged from the simulator's final state, never from the LLM's own claim.

Checks read the compact state: palm {position, facing, fingers_pointing}, finger_curl
(0 straight .. 1 fully curled), objects {position, lifted_cm, held, in_target_zone, ...}.
"""

from dataclasses import dataclass
from typing import Callable

STRAIGHT, CURLED = 0.3, 0.6


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    check: Callable[[dict], bool]
    kind: str  # "manipulation" | "gesture" | "positioning"


def _curl(state, *fingers):
    return [state["finger_curl"][f] for f in fingers]


def _above(state, obj, tol=0.03):
    palm, pos = state["palm"]["position"], state["objects"][obj]["position"]
    return abs(palm[0] - pos[0]) < tol and abs(palm[1] - pos[1]) < tol and palm[2] > pos[2] + 0.04


TASKS = [
    Task("lift_cube", "Pick up the red cube and hold it at least 10 cm above the table.",
         lambda s: s["objects"]["red_cube"]["held"] and s["objects"]["red_cube"]["lifted_cm"] >= 8, "manipulation"),
    Task("cube_to_zone", "Put the red cube in the yellow target zone.",
         lambda s: s["objects"]["red_cube"]["in_target_zone"] and not s["objects"]["red_cube"]["held"], "manipulation"),
    Task("ball_to_zone", "Place the blue ball in the yellow target zone.",
         lambda s: s["objects"]["blue_ball"]["in_target_zone"] and not s["objects"]["blue_ball"]["held"], "manipulation"),
    Task("cylinder_to_zone", "Move the green cylinder into the yellow target zone.",
         lambda s: s["objects"]["green_cylinder"]["in_target_zone"], "manipulation"),
    Task("peace_sign", "Make a peace sign with the hand: index and middle fingers straight, ring and little "
         "fingers curled.",
         lambda s: max(_curl(s, "index", "middle")) < STRAIGHT and min(_curl(s, "ring", "little")) > CURLED, "gesture"),
    Task("point", "Point with the index finger: index straight, the other three fingers curled.",
         lambda s: _curl(s, "index")[0] < STRAIGHT and min(_curl(s, "middle", "ring", "little")) > CURLED, "gesture"),
    Task("fist", "Make a fist.",
         lambda s: min(_curl(s, "index", "middle", "ring", "little")) > CURLED, "gesture"),
    Task("bend_index_only", "Bend only the index finger; keep the middle, ring and little fingers straight.",
         # "bend" = clearly not straight (a half bend counts), unlike "curled" which means a fist-like curl
         lambda s: _curl(s, "index")[0] > STRAIGHT and max(_curl(s, "middle", "ring", "little")) < STRAIGHT, "gesture"),
    Task("hover_ball", "Move the hand so it hovers directly above the blue ball, palm facing down.",
         lambda s: _above(s, "blue_ball") and s["palm"]["facing"] == "down", "positioning"),
    Task("palm_left", "Rotate the hand so the palm faces to the robot's left.",
         lambda s: s["palm"]["facing"] == "left", "positioning"),
]
TASKS_BY_ID = {t.id: t for t in TASKS}
