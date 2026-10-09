"""Measure what each actuator physically does, so tools can be named from the model, not guesses.

For every actuator we nudge its joint(s) in the + direction from the home pose (pure kinematics,
no physics) and look at what moves:
  * finger/thumb actuators: that finger's tip, in the palm frame,
  * wrist actuators: the middle fingertip, relative to the arm's flange,
  * arm actuators: the palm centre in the world (or, if it barely translates, its rotation).
The dominant motion is turned into a direction label, e.g. "toward_palm" or "left".
"""

import mujoco
import numpy as np

from robot_sim.robot import PALM_SITE, actuator_group
from robot_sim.scene import reset_home

# Palm frame: +x toward the thumb side, -y out of the palm (palm normal), +z along the fingers.
_PALM_AXES = {"toward_thumb_side": [1, 0, 0], "toward_little_finger_side": [-1, 0, 0],
              "toward_palm": [0, -1, 0], "toward_back_of_hand": [0, 1, 0],
              "toward_fingertips": [0, 0, 1], "toward_wrist": [0, 0, -1]}
_WORLD_AXES = {"forward": [1, 0, 0], "back": [-1, 0, 0], "left": [0, 1, 0], "right": [0, -1, 0],
               "up": [0, 0, 1], "down": [0, 0, -1]}
_ROTATIONS = {"turn_left": [0, 0, 1], "turn_right": [0, 0, -1], "tilt_fingers_down": [0, 1, 0],
              "tilt_fingers_up": [0, -1, 0], "roll_thumb_side_up": [-1, 0, 0], "roll_thumb_side_down": [1, 0, 0]}
OPPOSITE = {}
for table in (_PALM_AXES, _WORLD_AXES, _ROTATIONS):
    for a, va in table.items():
        for b, vb in table.items():
            if np.allclose(va, -np.asarray(vb)):
                OPPOSITE[a] = b

NUDGE = 0.1  # rad


def _label(vec: np.ndarray, axes: dict) -> str:
    return max(axes, key=lambda k: float(np.dot(vec, axes[k])))


def _actuator_joints(model, i):
    """Joint ids driven by actuator i (one joint, or the joints of a coupled fixed tendon)."""
    if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
        return [model.actuator_trnid[i, 0]]
    t = model.actuator_trnid[i, 0]
    return [model.wrap_objid[model.tendon_adr[t] + k] for k in range(model.tendon_num[t])]


def calibrate(model: mujoco.MjModel) -> dict[str, dict[str, str]]:
    """{actuator: {"positive": label, "negative": label}} measured at the home pose."""
    home = mujoco.MjData(model)
    reset_home(model, home)
    palm_site = model.site(PALM_SITE).id
    flange = model.body("attachment").id
    result = {}
    for i in range(model.nu):
        name = model.actuator(i).name
        group = actuator_group(name)
        joints = _actuator_joints(model, i)
        d = mujoco.MjData(model)
        d.qpos[:] = home.qpos
        # Start where a + nudge is possible: mid-range-ish for the hand, home for the arm.
        for j in joints:
            lo, hi = model.jnt_range[j]
            start = home.qpos[model.jnt_qposadr[j]] if group == "arm" else 0.3 * lo + 0.7 * max(lo, 0.0)
            d.qpos[model.jnt_qposadr[j]] = np.clip(start, lo + NUDGE, hi - NUDGE)

        def measure():
            mujoco.mj_kinematics(model, d)
            if group == "arm":
                return d.site_xpos[palm_site].copy(), d.site_xmat[palm_site].reshape(3, 3).copy()
            tip = "middle" if group == "wrist" else group
            ref = d.xmat[flange].reshape(3, 3) if group == "wrist" else d.site_xmat[palm_site].reshape(3, 3)
            origin = d.xpos[flange] if group == "wrist" else d.site_xpos[palm_site]
            return ref.T @ (d.site(f"tip_{tip}").xpos - origin), None

        p0, r0 = measure()
        for j in joints:
            d.qpos[model.jnt_qposadr[j]] += NUDGE
        p1, r1 = measure()

        if group == "arm":
            dp = p1 - p0
            if np.linalg.norm(dp) > 0.02 * NUDGE / 0.1:  # translates more than 2 cm per 0.1 rad
                label = _label(dp, _WORLD_AXES)
            else:
                rel = r1 @ r0.T
                axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
                label = _label(axis, _ROTATIONS)
        elif group == "wrist":
            # Express in the palm axes the hand has when the wrist is straight (bracket frame).
            straight = mujoco.MjData(model)
            straight.qpos[:] = home.qpos
            mujoco.mj_kinematics(model, straight)
            to_palm = straight.site_xmat[palm_site].reshape(3, 3).T @ straight.xmat[flange].reshape(3, 3)
            label = _label(to_palm @ (p1 - p0), _PALM_AXES)
        else:
            label = _label(p1 - p0, _PALM_AXES)
        result[name] = {"positive": label, "negative": OPPOSITE[label]}
    return result


if __name__ == "__main__":
    from robot_sim.scene import build_model

    for act, motion in calibrate(build_model()).items():
        print(f"{act:10} {actuator_group(act):6}  + {motion['positive']:26}  - {motion['negative']}")
