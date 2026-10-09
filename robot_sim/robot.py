"""Static description of the robot: actuator groups, body -> hand-part mapping, and the
named hand poses shipped with the Shadow Hand model (Menagerie keyframes)."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from robot_sim.scene import HAND_XML, MENAGERIE

KEYFRAMES_XML = MENAGERIE / "shadow_hand" / "keyframes.xml"

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
ARM_ACTUATORS = [f"actuator{i}" for i in range(1, 8)]
WRIST_JOINTS = ["rh_WRJ2", "rh_WRJ1"]
WRIST_ACTUATORS = ["rh_A_WRJ2", "rh_A_WRJ1"]
PALM_SITE = "grasp_site"  # in front of the palm, where a grasped object sits

FINGERS = ["thumb", "index", "middle", "ring", "little"]
_PART_PREFIX = {"rh_th": "thumb", "rh_ff": "index", "rh_mf": "middle", "rh_rf": "ring", "rh_lf": "little"}
FINGERTIP_SITES = {finger: f"tip_{finger}" for finger in FINGERS}

# World directions, from the point of view of the robot (its base faces +x).
DIRECTIONS = {
    "forward": [1, 0, 0], "back": [-1, 0, 0],
    "left": [0, 1, 0], "right": [0, -1, 0],
    "up": [0, 0, 1], "down": [0, 0, -1],
}
# Default finger direction for each palm-facing direction (fingers must be perpendicular to the palm normal).
DEFAULT_FINGERS_FOR_PALM = {
    "down": "forward", "up": "forward",
    "forward": "down", "back": "down", "left": "down", "right": "down",
}


def hand_part(body_name: str) -> str | None:
    """Map a body name to a hand part: thumb/index/middle/ring/little/palm/wrist, or 'arm'."""
    if body_name in ("rh_palm",):
        return "palm"
    if body_name == "rh_wrist":
        return "wrist"
    for prefix, part in _PART_PREFIX.items():
        if body_name.startswith(prefix):
            return part
    if body_name.startswith("link") or body_name == "attachment":
        return "arm"
    return None


def actuator_group(name: str) -> str:
    if name in ARM_ACTUATORS:
        return "arm"
    if "WR" in name:
        return "wrist"
    return _PART_PREFIX["rh_" + name.removeprefix("rh_A_")[:2].lower()]


def load_hand_poses(model: mujoco.MjModel) -> dict[str, dict[str, float]]:
    """Named finger poses from the Shadow Hand keyframes, as {pose: {actuator: ctrl}}.

    Keyframes are joint angles for the hand-only model; we convert them to actuator
    targets via the actuator transmission lengths (coupled J1+J2 tendons included).
    Wrist actuators are left out so a pose only shapes the fingers.
    """
    hand_only = mujoco.MjModel.from_xml_path(str(HAND_XML))
    joint_names = [hand_only.joint(i).name for i in range(hand_only.njnt)]
    finger_acts = [i for i in range(model.nu)
                   if model.actuator(i).name.startswith("rh_A_") and "WR" not in model.actuator(i).name]
    scratch = mujoco.MjData(model)
    poses = {}
    for key in ET.parse(KEYFRAMES_XML).iter("key"):
        qpos = np.array(key.get("qpos").split(), dtype=float)
        scratch.qpos[:] = model.qpos0
        for name, value in zip(joint_names, qpos):
            scratch.qpos[model.joint(name).qposadr[0]] = value
        mujoco.mj_fwdPosition(model, scratch)
        lo, hi = model.actuator_ctrlrange[finger_acts].T
        ctrl = np.clip(scratch.actuator_length[finger_acts], lo, hi)
        poses[key.get("name").replace(" ", "_")] = {
            model.actuator(i).name: round(float(c), 4) for i, c in zip(finger_acts, ctrl)
        }
    return poses


# What each actuator physically does: (short name stem used for tool names, description).
ACTUATOR_INFO = {
    "actuator1": ("arm_base", "arm joint 1: base rotation (turn the whole arm)"),
    "actuator2": ("arm_shoulder", "arm joint 2: shoulder pitch (lean the arm forward/back)"),
    "actuator3": ("arm_upper_arm_twist", "arm joint 3: upper-arm rotation"),
    "actuator4": ("arm_elbow", "arm joint 4: elbow bend"),
    "actuator5": ("arm_forearm_twist", "arm joint 5: forearm rotation"),
    "actuator6": ("arm_wrist_pitch", "arm joint 6: arm-side wrist pitch"),
    "actuator7": ("arm_flange_twist", "arm joint 7: flange rotation (spin the hand about the arm's last axis)"),
    "rh_A_WRJ2": ("wrist_side", "wrist side-to-side deviation (radial/ulnar)"),
    "rh_A_WRJ1": ("wrist_bend", "wrist flexion/extension (bend the hand toward the palm side or the back)"),
    "rh_A_THJ5": ("thumb_base_rotation", "thumb base rotation (swing the thumb around its base)"),
    "rh_A_THJ4": ("thumb_base_lift", "thumb base abduction (lift the thumb away from the side of the palm)"),
    "rh_A_THJ3": ("thumb_hub_twist", "thumb hub joint (small twist)"),
    "rh_A_THJ2": ("thumb_middle_joint", "thumb middle joint bend"),
    "rh_A_THJ1": ("thumb_tip_joint", "thumb tip joint bend"),
    "rh_A_FFJ4": ("index_spread", "index finger side-to-side swing at the knuckle (spread)"),
    "rh_A_FFJ3": ("index_knuckle", "index finger knuckle (base joint) bend"),
    "rh_A_FFJ0": ("index_upper_joints", "index finger middle + tip joints bending together (coupled)"),
    "rh_A_MFJ4": ("middle_spread", "middle finger side-to-side swing at the knuckle (spread)"),
    "rh_A_MFJ3": ("middle_knuckle", "middle finger knuckle (base joint) bend"),
    "rh_A_MFJ0": ("middle_upper_joints", "middle finger middle + tip joints bending together (coupled)"),
    "rh_A_RFJ4": ("ring_spread", "ring finger side-to-side swing at the knuckle (spread)"),
    "rh_A_RFJ3": ("ring_knuckle", "ring finger knuckle (base joint) bend"),
    "rh_A_RFJ0": ("ring_upper_joints", "ring finger middle + tip joints bending together (coupled)"),
    "rh_A_LFJ5": ("little_palm_cup", "little finger metacarpal (cup the palm on the little-finger side)"),
    "rh_A_LFJ4": ("little_spread", "little finger side-to-side swing at the knuckle (spread)"),
    "rh_A_LFJ3": ("little_knuckle", "little finger knuckle (base joint) bend"),
    "rh_A_LFJ0": ("little_upper_joints", "little finger middle + tip joints bending together (coupled)"),
}

# Joints whose angle counts as "curl" for each finger (0 = straight, 1 = fully bent).
CURL_JOINTS = {
    "thumb": ["rh_THJ4", "rh_THJ2", "rh_THJ1"],
    "index": ["rh_FFJ3", "rh_FFJ2", "rh_FFJ1"],
    "middle": ["rh_MFJ3", "rh_MFJ2", "rh_MFJ1"],
    "ring": ["rh_RFJ3", "rh_RFJ2", "rh_RFJ1"],
    "little": ["rh_LFJ3", "rh_LFJ2", "rh_LFJ1"],
}
