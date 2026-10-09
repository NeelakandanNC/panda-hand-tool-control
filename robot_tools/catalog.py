"""Generate the robot's tool catalog from the simulation's `describe` output.

Nothing here is a hand-picked list of tools: every tool is derived from something the model
reports — an actuator and its measured motion directions, a finger group, a hand-pose keyframe,
a scene object, a world direction, a camera. Families:

  actuator   3 tools per actuator: set its angle, nudge it +, nudge it - (directions are measured)
  finger     per finger: curl, straighten, set curl fraction (between the open/close keyframes)
  spread     all knuckle-spread actuators together: spread fingers apart / bring them together
  pose       one tool per hand-pose keyframe shipped with the hand model
  palm       Cartesian moves of the palm: to a point, along each world direction, face each direction
  object     per scene object: move palm above it / onto it; plus the target zone
  sense      state, per-object state, camera snapshot per camera, wait, reset, go home
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

FINGER_ORDER = ["thumb", "index", "middle", "ring", "little"]

# Extra words for each measured direction, so tool descriptions match how people phrase tasks.
DIRECTION_WORDS = {
    "toward_palm": "bend / flex / curl it in toward the palm",
    "toward_back_of_hand": "straighten / extend / open it back toward the back of the hand",
    "toward_thumb_side": "swing it sideways toward the thumb side of the hand",
    "toward_little_finger_side": "swing it sideways toward the little-finger side of the hand",
    "toward_fingertips": "move it toward the fingertips",
    "toward_wrist": "move it back toward the wrist",
    "left": "move the hand to the robot's left", "right": "move the hand to the robot's right",
    "up": "raise the hand", "down": "lower the hand",
    "forward": "move the hand forward, away from the robot base", "back": "move the hand back toward the robot base",
    "turn_left": "turn the hand counter-clockwise (seen from above)",
    "turn_right": "turn the hand clockwise (seen from above)",
    "tilt_fingers_up": "tilt the fingers up", "tilt_fingers_down": "tilt the fingers down",
    "roll_thumb_side_up": "roll the hand so the thumb side goes up",
    "roll_thumb_side_down": "roll the hand so the thumb side goes down",
}
WORLD_MOVES = {"up", "down", "left", "right", "forward", "back"}


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    family: str
    actuators: list[str] = field(default_factory=list)  # actuators this tool can move
    run: Callable[[Any, dict], dict] | None = field(default=None, repr=False)  # (sim client, args) -> result

    def to_json(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema,
                "family": self.family, "actuators": self.actuators}


def _schema(props: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props or {}, "required": required or [], "additionalProperties": False}


def _number(desc: str, default=None, minimum=None, maximum=None) -> dict:
    out = {"type": "number", "description": desc}
    if default is not None:
        out["default"] = default
    if minimum is not None:
        out["minimum"] = minimum
    if maximum is not None:
        out["maximum"] = maximum
    return out


DURATION = _number("Seconds the motion takes.", default=1.0, minimum=0.2, maximum=10)


def compact_state(state: dict) -> dict:
    """The parts of the robot state worth showing an LLM after every action."""
    return {"palm": state["palm"], "finger_curl": state["finger_curl"], "objects": state["objects"]}


def _finish(result: dict) -> dict:
    """Shrink a motion reply: drop the bulky full state, keep a compact one."""
    out = {k: v for k, v in result.items() if k != "state"}
    if "state" in result:
        out["state"] = compact_state(result["state"])
    return out


# ---------------------------------------------------------------------------------------------
def _actuator_tools(act: dict) -> list[ToolSpec]:
    name, stem, lo, hi = act["name"], act["stem"], *act["range"]
    coupled = act["drives"].endswith("J0")  # tendon driving two joints: value is their summed angle
    unit_note = " (sum of the two joint angles)" if coupled else ""
    lo_deg, hi_deg = round(math.degrees(lo), 1), round(math.degrees(hi), 1)
    what = act["description"]
    tools = [ToolSpec(
        name=f"{stem}_set_angle",
        description=(f"Set the {what} to an absolute angle in degrees{unit_note}. "
                     f"Range {lo_deg}° to {hi_deg}°; at home it is {round(math.degrees(act['home']), 1)}°."),
        input_schema=_schema({"degrees": _number("Target angle in degrees.", minimum=lo_deg, maximum=hi_deg),
                              "duration": DURATION}, ["degrees"]),
        family="actuator", actuators=[name],
        run=lambda sim, a, n=name: _finish(sim.call("move_actuators", targets={n: math.radians(a["degrees"])},
                                                    duration=a.get("duration", 1.0))),
    )]
    for sign in ("positive", "negative"):
        label = act["motion"][sign]
        verb = f"move_{label}" if label in WORLD_MOVES else label
        tools.append(ToolSpec(
            name=f"{stem}_{verb}",
            description=(f"Using only the {what}: {DIRECTION_WORDS[label]}. "
                         f"Moves it by the given number of degrees from where it is now (stops at its limits)."),
            input_schema=_schema({"degrees": _number("How far to move, in degrees.", default=15, minimum=0, maximum=180),
                                  "duration": DURATION}),
            family="actuator", actuators=[name],
            run=lambda sim, a, n=name, s=(1 if sign == "positive" else -1): _finish(sim.call(
                "move_actuators", targets={n: s * math.radians(a.get("degrees", 15))}, relative=True,
                duration=a.get("duration", 1.0))),
        ))
    return tools


def _finger_tools(desc: dict) -> list[ToolSpec]:
    open_pose, closed_pose = desc["hand_poses"]["open_hand"], desc["hand_poses"]["close_hand"]
    by_finger = defaultdict(list)
    for act in desc["actuators"]:
        if act["group"] in FINGER_ORDER and open_pose.get(act["name"]) != closed_pose.get(act["name"]):
            by_finger[act["group"]].append(act["name"])

    def curl_targets(acts, fraction):
        return {a: open_pose[a] + fraction * (closed_pose[a] - open_pose[a]) for a in acts}

    tools = []
    for finger in FINGER_ORDER:
        acts = by_finger[finger]
        tools += [
            ToolSpec(f"curl_{finger}", f"Curl the whole {finger}{'' if finger == 'thumb' else ' finger'} closed "
                     f"(all its joints, as in a fist). Other fingers are not moved.",
                     _schema({"duration": DURATION}), "finger", acts,
                     lambda sim, a, acts=acts: _finish(sim.call("move_actuators", targets=curl_targets(acts, 1.0),
                                                                duration=a.get("duration", 1.0)))),
            ToolSpec(f"straighten_{finger}", f"Straighten / open the whole {finger}"
                     f"{'' if finger == 'thumb' else ' finger'} fully. Other fingers are not moved.",
                     _schema({"duration": DURATION}), "finger", acts,
                     lambda sim, a, acts=acts: _finish(sim.call("move_actuators", targets=curl_targets(acts, 0.0),
                                                                duration=a.get("duration", 1.0)))),
            ToolSpec(f"set_{finger}_curl", f"Set how curled the {finger}{'' if finger == 'thumb' else ' finger'} is: "
                     "0 = fully straight, 1 = fully curled, 0.5 = half way.",
                     _schema({"fraction": _number("Curl amount 0..1.", minimum=0, maximum=1), "duration": DURATION},
                             ["fraction"]), "finger", acts,
                     lambda sim, a, acts=acts: _finish(sim.call(
                         "move_actuators", targets=curl_targets(acts, a["fraction"]), duration=a.get("duration", 1.0)))),
        ]
    return tools


def _spread_tools(desc: dict) -> list[ToolSpec]:
    """All knuckle side-swing actuators together. 'Apart' = away from the middle finger, from measured motion."""
    side_of = {"index": "toward_thumb_side", "ring": "toward_little_finger_side", "little": "toward_little_finger_side"}
    apart = {}
    for act in desc["actuators"]:
        if act["stem"].endswith("_spread") and act["group"] in side_of:
            lo, hi = act["range"]
            apart[act["name"]] = hi if act["motion"]["positive"] == side_of[act["group"]] else lo
    together = {name: 0.0 for name in apart}
    return [
        ToolSpec("spread_fingers_apart", "Spread the four fingers apart sideways (fan the hand out), "
                 "using each finger's knuckle side-swing joint.", _schema({"duration": DURATION}), "spread",
                 list(apart), lambda sim, a: _finish(sim.call("move_actuators", targets=apart,
                                                              duration=a.get("duration", 1.0)))),
        ToolSpec("bring_fingers_together", "Bring the four fingers back together side by side (no gaps).",
                 _schema({"duration": DURATION}), "spread", list(together),
                 lambda sim, a: _finish(sim.call("move_actuators", targets=together, duration=a.get("duration", 1.0)))),
    ]


POSE_HINTS = {
    "close_hand": "a full fist: all fingers and thumb curled. Use it to grab, grasp and hold an object once the "
                  "open palm is resting on it, before lifting it; this is how the hand picks things up",
    "open_hand": "every finger and the thumb straight. Use it to let go of / release / drop a held object once it "
                 "has been lowered onto the table or target zone, or to open the hand before grasping",
    "rock": "a fist (rock-paper-scissors 'rock')", "paper": "a flat open hand (rock-paper-scissors 'paper')",
    "scissors": "index and middle extended, others curled ('scissors' / peace sign)",
    "point": "index finger extended, the rest curled (pointing)",
    "two_finger_pinch": "thumb and index fingertips pressed together (precision pinch)",
    "three_finger_pinch": "thumb, index and middle fingertips together (tripod pinch)",
    "parallel_extension": "fingers straight but bent at the knuckles, thumb opposite (flat clamp)",
    "grasp_sphere": "fingers and thumb wrapped as around a ball; can grab a ball the palm is resting on",
    "pre_grasp": "hand shaped ready to grasp (partially open)",
    "grasp_soft": "a gentle wrap grasp to hold an object the palm is on",
    "grasp_hard": "a firm wrap grasp to hold an object the palm is on",
}


def _pose_tools(desc: dict) -> list[ToolSpec]:
    tools = []
    for pose, targets in desc["hand_poses"].items():
        hint = POSE_HINTS.get(pose, pose.replace("_", " "))
        tools.append(ToolSpec(
            f"hand_pose_{pose}", f"Shape the hand into the '{pose.replace('_', ' ')}' pose: {hint}. "
            "Optionally apply it to only some fingers.",
            _schema({"fingers": {"type": "array", "items": {"enum": FINGER_ORDER},
                                 "description": "Only shape these fingers (default: all)."},
                     "duration": DURATION}),
            "pose", list(targets),
            lambda sim, a, pose=pose: _finish(sim.call("hand_pose", name=pose, duration=a.get("duration", 1.0),
                                                       fingers=a.get("fingers")))))
    return tools


def _palm_position(sim) -> list[float]:
    return sim.call("get_state")["palm"]["position"]


MOVE_USES = {
    "up": " Use it to lift a grasped object off the table, or to move the hand away after releasing an object.",
    "down": " Use it to lower the hand, or a held object, toward the table.",
}


def _palm_tools(desc: dict) -> list[ToolSpec]:
    directions = desc["directions"]
    facing_enum = {"enum": list(directions), "description": "Direction the palm faces (default: keep current)."}
    tools = [ToolSpec(
        "move_palm_to_position", "Move the palm centre to an exact [x, y, z] point in metres (world frame; "
        f"table top is at z={desc['table_height_m']}, +x is away from the robot, +y is the robot's left). "
        "Optionally choose which way the palm faces.",
        _schema({"x": _number("metres"), "y": _number("metres"), "z": _number("metres"), "palm_facing": facing_enum,
                 "path": {"enum": ["linear", "joint"], "description": "Straight line (default) or joint-space."},
                 "duration": _number("Seconds (default: based on distance).", minimum=0.2, maximum=10)},
                ["x", "y", "z"]), "palm",
        run=lambda sim, a: _finish(sim.call("move_palm", position=[a["x"], a["y"], a["z"]], palm=a.get("palm_facing"),
                                            path=a.get("path", "linear"), duration=a.get("duration"))))]
    for d, vec in directions.items():
        tools.append(ToolSpec(
            f"move_palm_{d}", f"Move the whole hand {d} in a straight line by a distance in centimetres, "
            f"keeping its orientation ({DIRECTION_WORDS[d]}).{MOVE_USES.get(d, '')}",
            _schema({"distance_cm": _number("How far to move.", default=10, minimum=0, maximum=60), "duration": DURATION}),
            "palm",
            run=lambda sim, a, vec=vec: _finish(sim.call(
                "move_palm", position=[p + v * a.get("distance_cm", 10) / 100 for p, v in zip(_palm_position(sim), vec)]))))
    for d in directions:
        tools.append(ToolSpec(
            f"turn_palm_to_face_{d}", f"Rotate the hand in place so the palm faces {d} "
            f"({'toward the table' if d == 'down' else 'up toward the sky' if d == 'up' else f'to the {d}'}).",
            _schema({"duration": DURATION}), "palm",
            run=lambda sim, a, d=d: _finish(sim.call("move_palm", position=_palm_position(sim), palm=d,
                                                     duration=a.get("duration")))))
    return tools


def _object_tools(desc: dict) -> list[ToolSpec]:
    tools = []
    for obj, info in desc["objects"].items():
        label = obj.replace("_", " ")

        def above(sim, a, obj=obj):
            x, y, z = sim.call("get_state")["objects"][obj]["position"]
            return _finish(sim.call("move_palm", palm="down", position=[x, y, z + a.get("height_cm", 15) / 100]))

        def onto(sim, a, obj=obj):
            x, y, z = sim.call("get_state")["objects"][obj]["position"]
            return _finish(sim.call("move_palm", position=[x, y, z + a.get("height_offset_cm", 0) / 100]))

        tools += [
            ToolSpec(f"move_palm_above_{obj}", f"Move the hand, palm facing down, to hover above the {label} "
                     f"({info['shape']}). Use this to approach the {label} before grasping it.",
                     _schema({"height_cm": _number("Height of the palm centre above the object's centre.", default=15,
                                                   minimum=3, maximum=40)}), "object", run=above),
            ToolSpec(f"lower_palm_onto_{obj}", f"Lower the hand straight down so the palm centre is at the {label} "
                     f"(grasping position), keeping the current orientation. Close the hand afterwards to grab it.",
                     _schema({"height_offset_cm": _number("Stop this far above the object's centre.", default=0,
                                                          minimum=-2, maximum=10)}), "object", run=onto),
        ]
    zone = desc["target_zone"]
    rest_z = desc["table_height_m"]
    tools += [
        ToolSpec("move_palm_above_target_zone", "Carry the hand (and anything it holds) to hover above the yellow "
                 "target zone, palm facing down.",
                 _schema({"height_cm": _number("Palm height above the table.", default=17, minimum=5, maximum=40)}),
                 "object", run=lambda sim, a: _finish(sim.call(
                     "move_palm", palm="down", position=[*zone["pos"], rest_z + a.get("height_cm", 17) / 100]))),
        ToolSpec("lower_palm_to_target_zone", "Lower the hand straight down over the target zone to place a held "
                 "object on the table there. Open the hand afterwards to release it.",
                 _schema({"height_cm": _number("Palm height above the table when stopping.", default=3.5,
                                               minimum=1, maximum=20)}),
                 "object", run=lambda sim, a: _finish(sim.call(
                     "move_palm", position=[*zone["pos"], rest_z + a.get("height_cm", 3.5) / 100]))),
    ]
    return tools


def _sense_tools(desc: dict) -> list[ToolSpec]:
    home = {act["name"]: act["home"] for act in desc["actuators"]}
    tools = [
        ToolSpec("get_robot_state", "Read the full robot state: palm position and orientation, fingertip positions, "
                 "finger curl, wrist and arm joint angles, and every object's position, contacts and status.",
                 _schema(), "sense", run=lambda sim, a: sim.call("get_state")),
        ToolSpec("wait", "Do nothing for a moment and let the scene settle.",
                 _schema({"seconds": _number("How long to wait.", default=0.5, minimum=0, maximum=10)}), "sense",
                 run=lambda sim, a: _finish(sim.call("wait", seconds=a.get("seconds", 0.5)))),
        ToolSpec("move_to_home_pose", "Move the arm and hand back to the ready pose (palm down above the table, "
                 "hand open) without touching the objects.", _schema({"duration": _number("Seconds.", default=2.0)}),
                 "sense", list(home),
                 run=lambda sim, a: _finish(sim.call("move_actuators", targets=home, duration=a.get("duration", 2.0)))),
        ToolSpec("reset_scene", "Reset everything: robot to the home pose and all objects back to their start "
                 "positions.", _schema(), "sense", run=lambda sim, a: sim.call("reset")),
    ]
    for obj in desc["objects"]:
        tools.append(ToolSpec(
            f"get_{obj}_state", f"Read the {obj.replace('_', ' ')}'s state: position, how high it is lifted, "
            "which hand parts touch it, whether it is held and whether it is in the target zone.",
            _schema(), "sense", run=lambda sim, a, obj=obj: sim.call("get_state")["objects"][obj]))
    for cam in desc["cameras"]:
        tools.append(ToolSpec(
            f"camera_snapshot_{cam}", f"Take a picture of the scene from the '{cam}' camera (returns a PNG image).",
            _schema(), "sense", run=lambda sim, a, cam=cam: sim.call("snapshot", camera=cam)))
    return tools


def build_catalog(desc: dict) -> list[ToolSpec]:
    tools = []
    for act in desc["actuators"]:
        tools += _actuator_tools(act)
    tools += _finger_tools(desc) + _spread_tools(desc) + _pose_tools(desc)
    tools += _palm_tools(desc) + _object_tools(desc) + _sense_tools(desc)
    names = [t.name for t in tools]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"duplicate tool names: {sorted(duplicates)}")
    return tools
