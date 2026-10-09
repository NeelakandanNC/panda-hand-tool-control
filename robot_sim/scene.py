"""Builds the MuJoCo scene: a Franka Emika Panda arm bolted to a table with a Shadow Dexterous Hand
mounted on its flange (in place of the two-finger gripper), plus objects to manipulate."""

from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MENAGERIE = ROOT / "assets" / "menagerie"
ARM_XML = MENAGERIE / "franka_emika_panda" / "panda_nohand.xml"
HAND_XML = MENAGERIE / "shadow_hand" / "right_hand.xml"

TABLE_HEIGHT = 0.40
TABLE_CENTER = [0.15, 0.0]
TABLE_HALF = [0.55, 0.55]
ROBOT_BASE = [-0.30, 0.0, TABLE_HEIGHT]  # arm stands near the back edge of the table

# Home ("ready") pose: palm down, fingers forward, wrist straight, palm 30 cm above the table at
# (0.25, 0, 0.70). Solved by IK from Menagerie's home with the flange turned 180 deg. Hand open.
HOME_ARM = [0.0, -0.2598, 0.0, -2.5694, 0.0, 2.3096, 2.3562]
# finger: (distal body, tip distance along the distal segment in metres)
FINGERTIP_SITES = {"thumb": ("rh_thdistal", 0.033), "index": ("rh_ffdistal", 0.029),
                   "middle": ("rh_mfdistal", 0.029), "ring": ("rh_rfdistal", 0.029), "little": ("rh_lfdistal", 0.029)}
HAND_MOUNT_OFFSET = 0.02  # short adapter between the Panda flange and the Shadow wrist
# Angled bracket: tilts the hand so the palm faces along the flange axis. With the Panda in its
# natural "flange pointing down" posture the palm then faces the table (like a wrist bent back).
HAND_MOUNT_PITCH_DEG = -90.0

# name: (shape, size, xy, rgba, mass) -- sized for a human-scale hand.
OBJECTS = {
    "red_cube":       ("box",      [0.025, 0.025, 0.025], [0.25,  0.00], [0.85, 0.20, 0.20, 1], 0.06),
    "blue_ball":      ("sphere",   [0.035],               [0.30,  0.25], [0.20, 0.40, 0.90, 1], 0.06),
    "green_cylinder": ("cylinder", [0.025, 0.05],         [0.30, -0.25], [0.20, 0.75, 0.30, 1], 0.07),
}
TARGET_ZONE = {"pos": [0.40, 0.15], "radius": 0.07}


def _look_at(eye, target) -> list[float]:
    """Camera xyaxes for a camera at `eye` looking at `target` with world-z up."""
    fwd = np.subtract(target, eye) / np.linalg.norm(np.subtract(target, eye))
    x = np.cross(fwd, [0, 0, 1])
    x /= np.linalg.norm(x)
    return [*x, *np.cross(x, fwd)]


def _add_world(spec: mujoco.MjSpec) -> None:
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3], width=300, height=300)
    spec.add_material(name="floor", textures=["", "grid"], texrepeat=[6, 6], texuniform=True, reflectance=0.1)
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.3, 0.5, 0.7], rgb2=[0, 0, 0], width=512, height=3072)
    spec.visual.global_.azimuth = 150
    spec.visual.global_.elevation = -25
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.stat.extent = 1.2
    spec.stat.center = [0.1, 0.0, TABLE_HEIGHT + 0.25]

    world = spec.worldbody
    world.add_light(pos=[0, 0, 2.0], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="floor")
    world.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[*TABLE_CENTER, TABLE_HEIGHT / 2],
                   size=[*TABLE_HALF, TABLE_HEIGHT / 2], rgba=[0.55, 0.40, 0.28, 1], friction=[1.0, 0.01, 0.001])
    world.add_geom(name="target_zone", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   pos=[*TARGET_ZONE["pos"], TABLE_HEIGHT + 0.0005], size=[TARGET_ZONE["radius"], 0.0005, 0],
                   rgba=[1.0, 0.85, 0.1, 0.6], contype=0, conaffinity=0)

    shapes = {"box": mujoco.mjtGeom.mjGEOM_BOX, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
              "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER}
    for name, (shape, size, xy, rgba, mass) in OBJECTS.items():
        half_height = size[0] if shape == "sphere" else size[-1]
        body = world.add_body(name=name, pos=[*xy, TABLE_HEIGHT + half_height + 0.001])
        body.add_freejoint(name=f"{name}_joint")
        body.add_geom(name=f"{name}_geom", type=shapes[shape], size=size + [0] * (3 - len(size)), rgba=rgba,
                      mass=mass, condim=4, priority=1, friction=[1.5, 0.02, 0.002])

    world.add_camera(name="overview", pos=[1.55, -1.25, 1.45], xyaxes=_look_at([1.55, -1.25, 1.45], [0.1, 0.0, 0.6]))


def build_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(ARM_XML))
    spec.modelname = "panda_shadow_hand_tabletop"
    # Shadow Hand contact settings (finger contacts need the elliptic cone).
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10

    spec.body("link0").pos = ROBOT_BASE
    _attach_hand(spec)
    # Real arm controllers compensate gravity; do the same so joint targets are tracked accurately.
    for body in spec.bodies:
        if body.name.startswith(("link", "rh_")):
            body.gravcomp = 1.0

    _add_world(spec)

    # Menagerie's "home" key only covers the arm; rebuild it for the full scene
    # (arm + hand + free-floating objects) once qpos0 is known.
    for key in list(spec.keys):
        spec.delete(key)
    model = spec.compile()
    qpos = model.qpos0.copy()
    qpos[:7] = HOME_ARM
    ctrl = np.zeros(model.nu)
    ctrl[:7] = HOME_ARM
    spec.add_key(name="home", qpos=qpos.tolist(), ctrl=ctrl.tolist())
    return spec


def _attach_hand(arm: mujoco.MjSpec) -> None:
    """Mount the Shadow Hand (wrist and everything distal; no forearm) on the Panda flange."""
    hand = mujoco.MjSpec.from_file(str(HAND_XML))
    for exclude in list(hand.excludes):  # refers to the forearm we leave behind
        if "rh_forearm" in (exclude.bodyname1, exclude.bodyname2):
            hand.delete(exclude)
    # Fingertip sites ~2 mm inside the end of each distal segment (the joint origin is not the tip).
    for finger, (body, length) in FINGERTIP_SITES.items():
        hand.body(body).add_site(name=f"tip_{finger}", pos=[0, 0, length], size=[0.004, 0, 0], group=4)
    wrist = hand.body("rh_wrist")
    wrist.pos = [0, 0, 0]
    wrist.quat = [1, 0, 0, 0]

    flange = arm.body("attachment")
    flange.add_geom(name="hand_adapter", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, HAND_MOUNT_OFFSET / 2],
                    size=[0.032, HAND_MOUNT_OFFSET / 2, 0], rgba=[0.2, 0.2, 0.2, 1], contype=0, conaffinity=0)
    pitch = np.radians(HAND_MOUNT_PITCH_DEG)
    flange.add_frame(pos=[0, 0, HAND_MOUNT_OFFSET],
                     quat=[np.cos(pitch / 2), np.sin(pitch / 2), 0, 0]).attach_body(wrist, "", "")


def build_model() -> mujoco.MjModel:
    return build_spec().compile()


def reset_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)


if __name__ == "__main__":
    m = build_model()
    print(f"nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody}")
    print("actuators:", [m.actuator(i).name for i in range(m.nu)])
