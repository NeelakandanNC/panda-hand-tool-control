"""Damped-least-squares inverse kinematics for the palm site over the 7 arm joints + 2 wrist joints."""

from dataclasses import dataclass

import mujoco
import numpy as np

from robot_sim.robot import ARM_JOINTS, PALM_SITE, WRIST_JOINTS
from robot_sim.scene import HOME_ARM


@dataclass
class IKResult:
    q: np.ndarray          # joint angles for IK_JOINTS (9,)
    pos_error: float       # metres
    rot_error: float       # radians (0 when orientation unconstrained)
    success: bool


def rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Axis-angle vector rotating `current` onto `target` (world frame); well-defined up to 180 deg."""
    q_cur, q_tgt, q_err = np.zeros(4), np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q_cur, current.flatten())
    mujoco.mju_mat2Quat(q_tgt, target.flatten())
    mujoco.mju_negQuat(q_cur, q_cur)
    mujoco.mju_mulQuat(q_err, q_tgt, q_cur)
    err = np.zeros(3)
    mujoco.mju_quat2Vel(err, q_err, 1.0)
    return err


def palm_frame(normal, fingers) -> np.ndarray:
    """Rotation matrix of the palm site for a given palm normal and finger direction (world frame).

    In the Shadow Hand palm frame the palm faces -y and the fingers extend along +z.
    """
    y = -np.asarray(normal, dtype=float)
    z = np.asarray(fingers, dtype=float)
    y /= np.linalg.norm(y)
    z -= y * (z @ y)  # make fingers perpendicular to the normal
    if np.linalg.norm(z) < 1e-6:
        raise ValueError("finger direction must not be parallel to the palm normal")
    z /= np.linalg.norm(z)
    return np.column_stack([np.cross(y, z), y, z])


IK_JOINTS = ARM_JOINTS + WRIST_JOINTS
IK_REST = np.array(HOME_ARM + [0.0, 0.0])


class ArmIK:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)  # scratch copy; never touches the live simulation
        self.site = model.site(PALM_SITE).id
        jids = [model.joint(j).id for j in IK_JOINTS]
        self.qadr = model.jnt_qposadr[jids]
        self.dadr = model.jnt_dofadr[jids]
        self.lo, self.hi = model.jnt_range[jids].T
        self.q_rest = IK_REST

    def solve(self, qpos_seed: np.ndarray, target_pos, target_rot: np.ndarray | None = None,
              restarts: int = 6) -> IKResult:
        """Solve from the seed; on failure retry from home and random postures, keep the best."""
        best = self._solve(qpos_seed, None, target_pos, target_rot)
        rng = np.random.default_rng(0)
        for i in range(restarts):
            if best.success:
                break
            arm_q = self.q_rest if i == 0 else rng.uniform(self.lo, self.hi)
            result = self._solve(qpos_seed, arm_q, target_pos, target_rot)
            if result.pos_error + 0.1 * result.rot_error < best.pos_error + 0.1 * best.rot_error:
                best = result
        return best

    def _step(self, J: np.ndarray, e: np.ndarray, q: np.ndarray, damping: float = 1e-2) -> np.ndarray:
        """Damped least-squares step plus a null-space pull toward the rest posture."""
        J_pinv = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(len(e)))
        # The null-space projector must use the exact pseudo-inverse: the damped one leaks the
        # pull into the task and leaves a steady-state error.
        null = np.eye(len(q)) - np.linalg.pinv(J, rcond=1e-4) @ J
        return J_pinv @ e + null @ (0.05 * (self.q_rest - q))

    def _solve(self, qpos_seed, arm_q, target_pos, target_rot, iters: int = 400,
               pos_tol: float = 2e-3, rot_tol: float = 2e-2) -> IKResult:
        m, d = self.model, self.data
        d.qpos[:] = qpos_seed
        if arm_q is not None:
            d.qpos[self.qadr] = arm_q
        target_pos = np.asarray(target_pos, dtype=float)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        rot_weight = 0.4

        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err_p = target_pos - d.site_xpos[self.site]
            err_r = np.zeros(3)
            if target_rot is not None:
                err_r = rotation_error(d.site_xmat[self.site].reshape(3, 3), target_rot)
            if np.linalg.norm(err_p) < pos_tol and np.linalg.norm(err_r) < rot_tol:
                break

            mujoco.mj_jacSite(m, d, jacp, jacr, self.site)
            if target_rot is None:
                J, e = jacp[:, self.dadr], err_p
            else:
                J = np.vstack([jacp[:, self.dadr], rot_weight * jacr[:, self.dadr]])
                e = np.concatenate([err_p, rot_weight * err_r])
            q = d.qpos[self.qadr]
            dq = self._step(J, e, q)
            # Active set: a joint pinned at a limit can't help. Drop it and let the others take over,
            # otherwise the solver keeps assigning motion to it and stalls short of the target.
            blocked = ((q <= self.lo + 1e-6) & (dq < 0)) | ((q >= self.hi - 1e-6) & (dq > 0))
            if blocked.any():
                J = J.copy()
                J[:, blocked] = 0.0
                dq = self._step(J, e, q)
                dq[blocked] = 0.0
            dq *= min(1.0, 0.2 / (np.abs(dq).max() + 1e-12))
            d.qpos[self.qadr] = np.clip(q + dq, self.lo, self.hi)

        mujoco.mj_kinematics(m, d)
        pos_error = float(np.linalg.norm(target_pos - d.site_xpos[self.site]))
        rot_error = 0.0
        if target_rot is not None:
            rot_error = float(np.linalg.norm(rotation_error(d.site_xmat[self.site].reshape(3, 3), target_rot)))
        success = pos_error < 0.01 and rot_error < 0.1
        return IKResult(d.qpos[self.qadr].copy(), pos_error, rot_error, success)
