"""Robot controller: turns commands into smooth actuator trajectories and reports readable state.

The controller never steps time on its own: the owner (Simulation) calls `step()` in its
physics loop and polls `busy()` to know when the current motion has finished and settled.
Command methods (`move_actuators`, `move_palm`, `hand_pose`, ...) plan a motion and return
an info dict immediately.
"""

import base64
import io

import mujoco
import numpy as np

from robot_sim.calibration import calibrate
from robot_sim.ik import IK_JOINTS, ArmIK, palm_frame
from robot_sim.robot import (ACTUATOR_INFO, ARM_ACTUATORS, CURL_JOINTS, DEFAULT_FINGERS_FOR_PALM, DIRECTIONS,
                             FINGERTIP_SITES, PALM_SITE, WRIST_ACTUATORS, actuator_group, hand_part,
                             load_hand_poses)
from robot_sim.scene import OBJECTS, TABLE_HEIGHT, TARGET_ZONE, reset_home

SETTLE_TIMEOUT = 1.0     # s to wait after a trajectory for the robot to come to rest
SETTLE_VELOCITY = 0.05   # rad/s or m/s
PALM_SPEED = 0.25        # m/s nominal Cartesian speed used for default durations


def _min_jerk(s: float) -> float:
    return s * s * s * (10 - 15 * s + 6 * s * s)


class Trajectory:
    """Min-jerk time scaling along a piecewise-linear path of actuator targets."""

    def __init__(self, waypoints: np.ndarray, start_time: float, duration: float):
        self.waypoints = np.asarray(waypoints)
        self.start, self.end = start_time, start_time + max(duration, 1e-3)

    def sample(self, t: float) -> np.ndarray:
        s = _min_jerk(np.clip((t - self.start) / (self.end - self.start), 0.0, 1.0))
        u = s * (len(self.waypoints) - 1)
        i = min(int(u), len(self.waypoints) - 2)
        f = u - i
        return (1 - f) * self.waypoints[i] + f * self.waypoints[i + 1]


def _direction(value, what: str) -> np.ndarray:
    if isinstance(value, str):
        if value not in DIRECTIONS:
            raise ValueError(f"unknown {what} direction {value!r}; use one of {list(DIRECTIONS)} or a 3-vector")
        return np.array(DIRECTIONS[value], dtype=float)
    vec = np.asarray(value, dtype=float)
    if vec.shape != (3,) or np.linalg.norm(vec) < 1e-6:
        raise ValueError(f"{what} must be a direction name or a non-zero 3-vector")
    return vec / np.linalg.norm(vec)


def _nearest_direction(vec: np.ndarray) -> str:
    name, cos = max(((n, float(np.dot(vec, d))) for n, d in DIRECTIONS.items()), key=lambda x: x[1])
    return name if cos > 0.9 else f"~{name}"


def _slerp_rot(r0: np.ndarray, r1: np.ndarray, s: float) -> np.ndarray:
    q0, q1 = np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q0, r0.flatten())
    mujoco.mju_mat2Quat(q1, r1.flatten())
    if q0 @ q1 < 0:
        q1 = -q1
    theta = np.arccos(np.clip(q0 @ q1, -1, 1))
    q = q0 if theta < 1e-6 else (np.sin((1 - s) * theta) * q0 + np.sin(s * theta) * q1) / np.sin(theta)
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, q / np.linalg.norm(q))
    return out.reshape(3, 3)


def _r(x, nd=3):
    return [round(float(v), nd) for v in x]


class Controller:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model, self.data = model, data
        self.act_names = [model.actuator(i).name for i in range(model.nu)]
        self.act_id = {n: i for i, n in enumerate(self.act_names)}
        self.ik = ArmIK(model)
        self.ik_acts = [self.act_id[n] for n in ARM_ACTUATORS + WRIST_ACTUATORS]  # same order as IK_JOINTS
        self.hand_poses = load_hand_poses(model)
        self.motion = calibrate(model)  # what + / - does for each actuator, measured at home
        self.palm_site = model.site(PALM_SITE).id
        self.robot_dofs = np.array([model.jnt_dofadr[j] for j in range(model.njnt)
                                    if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE])
        self.table_geom = model.geom("table_top").id
        self.trajectory: Trajectory | None = None
        self._renderer: mujoco.Renderer | None = None
        self.reset()

    # ------------------------------------------------------------------ stepping
    def step(self) -> None:
        if self.trajectory is not None:
            self.data.ctrl[:] = self.trajectory.sample(self.data.time)
        mujoco.mj_step(self.model, self.data)

    def busy(self) -> bool:
        traj = self.trajectory
        if traj is None:
            return False
        t = self.data.time
        if t < traj.end:
            return True
        settled = np.abs(self.data.qvel[self.robot_dofs]).max() < SETTLE_VELOCITY
        if settled or t > traj.end + SETTLE_TIMEOUT:
            self.data.ctrl[:] = traj.waypoints[-1]
            self.trajectory = None
            return False
        return True

    def _start(self, waypoints: list[np.ndarray], duration: float) -> None:
        self.trajectory = Trajectory(np.vstack([self.data.ctrl.copy(), *waypoints]), self.data.time, duration)

    # ------------------------------------------------------------------ commands
    def reset(self) -> dict:
        reset_home(self.model, self.data)
        self.trajectory = None
        self.rest_z = {name: float(self.data.body(name).xpos[2]) for name in OBJECTS}
        return {"reset": True}

    def wait(self, seconds: float = 0.5) -> dict:
        self.trajectory = Trajectory(np.vstack([self.data.ctrl.copy()] * 2), self.data.time, float(seconds))
        return {"waited": seconds}

    def move_actuators(self, targets: dict[str, float], duration: float = 1.0, relative: bool = False) -> dict:
        """Drive any subset of actuators to target values (radians, or tendon length for *J0)."""
        if not targets:
            raise ValueError("targets must name at least one actuator")
        goal = self.data.ctrl.copy()
        clipped = {}
        for name, value in targets.items():
            if name not in self.act_id:
                raise ValueError(f"unknown actuator {name!r}")
            i = self.act_id[name]
            want = goal[i] + value if relative else float(value)
            lo, hi = self.model.actuator_ctrlrange[i]
            goal[i] = np.clip(want, lo, hi)
            if goal[i] != want:
                clipped[name] = {"requested": round(want, 4), "limit": round(float(goal[i]), 4)}
        self._start([goal], duration)
        return {"moved": list(targets), "clipped_to_limits": clipped}

    def hand_pose(self, name: str, duration: float = 1.0, fingers: list[str] | None = None) -> dict:
        """Shape the hand into a named pose (optionally only for some fingers)."""
        if name not in self.hand_poses:
            raise ValueError(f"unknown hand pose {name!r}; available: {list(self.hand_poses)}")
        pose = self.hand_poses[name]
        if fingers:
            pose = {a: v for a, v in pose.items() if actuator_group(a) in fingers}
        return {"pose": name, **self.move_actuators(pose, duration)}

    def move_palm(self, position, palm=None, fingers=None, duration: float | None = None,
                  path: str = "linear") -> dict:
        """Move the palm centre to `position` [x, y, z] (world, metres).

        palm: direction the palm faces ('down', 'forward', ... or a 3-vector); None keeps the current orientation.
        fingers: direction the fingers point; defaults sensibly for the palm direction.
        path: 'linear' (straight line in space) or 'joint' (joint-space interpolation).
        """
        target = np.asarray(position, dtype=float)
        if target.shape != (3,):
            raise ValueError("position must be [x, y, z]")
        start_pos = self.data.site_xpos[self.palm_site].copy()
        start_rot = self.data.site_xmat[self.palm_site].reshape(3, 3).copy()
        if palm is None:
            target_rot = start_rot
        else:
            if fingers is None:
                fingers = DEFAULT_FINGERS_FOR_PALM[palm] if isinstance(palm, str) else "down"
            target_rot = palm_frame(_direction(palm, "palm"), _direction(fingers, "fingers"))

        if path not in ("linear", "joint"):
            raise ValueError("path must be 'linear' or 'joint'")
        distance = float(np.linalg.norm(target - start_pos))
        turn = float(np.arccos(np.clip((np.trace(start_rot.T @ target_rot) - 1) / 2, -1, 1)))
        if duration is None:
            duration = max(1.0, distance / PALM_SPEED, turn / 1.5)

        final = self.ik.solve(self.data.qpos, target, target_rot)
        if not final.success:
            return {"moved": False, "reason": "target not reachable with that orientation",
                    "position_error_m": round(final.pos_error, 3), "orientation_error_rad": round(final.rot_error, 3)}

        arm_path = self._linear_path(start_pos, start_rot, target, target_rot) if path == "linear" else None
        used = "linear"
        if arm_path is None:  # straight line infeasible (or not asked for): interpolate in joint space
            arm_path, used = [final.q], "joint"
        waypoints = []
        for q in arm_path:
            goal = self.data.ctrl.copy()
            goal[self.ik_acts] = q
            waypoints.append(goal)
        self._start(waypoints, duration)
        return {"moved": True, "target": _r(target), "path": used, "duration_s": round(duration, 2)}

    def _linear_path(self, p0, r0, p1, r1, step: float = 0.01) -> list[np.ndarray] | None:
        """IK along a straight line (with slerped orientation); None if any waypoint is infeasible."""
        n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / step)) + 1)
        seed = self.data.qpos.copy()
        path = []
        for k in range(1, n):
            s = k / (n - 1)
            sol = self.ik.solve(seed, p0 + s * (p1 - p0), _slerp_rot(r0, r1, s), restarts=0)
            if not sol.success:
                return None
            seed[self.ik.qadr] = sol.q
            path.append(sol.q)
        return path

    # ------------------------------------------------------------------ perception
    def _contacts(self) -> tuple[dict[str, set], set]:
        touching = {name: set() for name in OBJECTS}
        on_table = set()
        m = self.model
        for c in self.data.contact[: self.data.ncon]:
            for g_obj, g_other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                obj = m.body(m.geom_bodyid[g_obj]).name
                if obj not in touching:
                    continue
                if g_other == self.table_geom:
                    on_table.add(obj)
                part = hand_part(m.body(m.geom_bodyid[g_other]).name)
                if part:
                    touching[obj].add(part)
        return touching, on_table

    def get_state(self) -> dict:
        m, d = self.model, self.data
        rot = d.site_xmat[self.palm_site].reshape(3, 3)
        palm_normal, finger_dir = -rot[:, 1], rot[:, 2]
        touching, on_table = self._contacts()

        objects = {}
        for name in OBJECTS:
            pos = d.body(name).xpos
            parts = sorted(touching[name])
            lifted = float(pos[2] - self.rest_z[name])
            held = "thumb" in parts and len(parts) >= 2 and name not in on_table
            zone_dist = np.linalg.norm(pos[:2] - np.array(TARGET_ZONE["pos"]))
            objects[name] = {
                "position": _r(pos),
                "lifted_cm": round(100 * lifted, 1),
                "touching_hand_parts": parts,
                "held": held,
                "in_target_zone": bool(zone_dist < TARGET_ZONE["radius"] and name in on_table),
            }

        curl = {}
        for finger, joints in CURL_JOINTS.items():
            fr = [(d.joint(j).qpos[0] - m.joint(j).range[0]) / (m.joint(j).range[1] - m.joint(j).range[0])
                  for j in joints]
            curl[finger] = round(float(np.clip(np.mean(fr), 0, 1)), 2)

        return {
            "time_s": round(float(d.time), 2),
            "palm": {
                "position": _r(d.site_xpos[self.palm_site]),
                "facing": _nearest_direction(palm_normal),
                "fingers_pointing": _nearest_direction(finger_dir),
            },
            "fingertips": {f: _r(d.site(s).xpos) for f, s in FINGERTIP_SITES.items()},
            "finger_curl": curl,
            "wrist_deg": {j: round(float(np.degrees(d.joint(j).qpos[0])), 1) for j in IK_JOINTS[-2:]},
            "arm_joints_deg": {j: round(float(np.degrees(d.joint(j).qpos[0])), 1) for j in IK_JOINTS[:7]},
            "objects": objects,
        }

    def describe(self) -> dict:
        """Static description of the robot and scene: everything a planner needs to know about the action space."""
        m = self.model
        actuators = []
        for i, name in enumerate(self.act_names):
            target = (m.joint(m.actuator_trnid[i, 0]).name if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
                      else m.tendon(m.actuator_trnid[i, 0]).name)
            stem, description = ACTUATOR_INFO[name]
            actuators.append({
                "name": name, "stem": stem, "group": actuator_group(name), "drives": target,
                "description": description, "range": _r(m.actuator_ctrlrange[i], 4),
                "motion": self.motion[name], "home": round(float(m.key("home").ctrl[i]), 4),
            })
        return {
            "actuators": actuators,
            "hand_poses": self.hand_poses,
            "directions": DIRECTIONS,
            "objects": {name: {"shape": spec[0], "size_m": spec[1]} for name, spec in OBJECTS.items()},
            "table_height_m": TABLE_HEIGHT,
            "target_zone": TARGET_ZONE,
            "cameras": [m.camera(i).name for i in range(m.ncam)],
        }

    def snapshot(self, camera: str = "overview", width: int = 640, height: int = 480) -> dict:
        """Render a camera image (PNG, base64). Must be called from the physics/render thread."""
        from PIL import Image

        if self._renderer is None or (self._renderer.width, self._renderer.height) != (width, height):
            self._renderer = mujoco.Renderer(self.model, height, width)
        self._renderer.update_scene(self.data, camera=camera)
        buf = io.BytesIO()
        Image.fromarray(self._renderer.render()).save(buf, format="PNG")
        return {"camera": camera, "png_base64": base64.b64encode(buf.getvalue()).decode()}
