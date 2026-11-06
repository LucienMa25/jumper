"""The dunk world: one ball to grasp and one hoop to score in.

Two entities, authored the way `five_foot/objects.py` authors its row — an
`MjSpec` per prop, **in radians** (the trap that file documents: MjSpec
defaults to degrees and a radian-authored euler silently rotates nothing),
collision on the prop channel, `PROP_*` contact parameters imported from
five_foot rather than re-declared, so the two tasks' props cannot drift into
different physics.

Layout (x ahead of the robot's spawn, y lateral):

    robot @ 0        ball @ BALL_X=0.55        hoop @ HOOP_X=1.45, y=HOOP_Y=0.35

The ball sits on a diagonal, not straight ahead, so the path to the hoop
requires walking *and* strafing — the five-foot skill set — instead of a
single straight sprint one policy could fake with one gait phase.

The hoop is five_foot's dropbin scaled to the ball: a 20 mm rim (the claw's
measured lift tops out near 30 mm; a taller rim is a wall the task cannot
ask it to climb) with inner half-width `HOOP_INNER` = ball diameter + a
10 mm margin. Its rim is painted Sixers blue; the ball is leather orange.
"""

from __future__ import annotations

import mujoco

from mjlab.entity import EntityCfg

from ..five_foot.objects import (
    PROP_BIT,
    PROP_CONDIM,
    PROP_FRICTION,
    PROP_PRIORITY,
    PROP_SOLREF,
)

#: Ball: a junior-sized leather ball the claw can actually close on. The
#: official game ball (0.624 kg) is out of the claw's measured grip budget at
#: stage 1; mass is set by density over the sphere volume below.
BALL_R = 0.075
#: v3: was (0.55, -0.20) — d=0.585 m put the high-value region too far for
# the entropy budget to bump into it by iteration ~900 (v2 measurement).
# 0.43 m now: still a walk, reachable by a exploring policy within its first
# steps of interest.
BALL_X, BALL_Y = 0.30, -0.12
BALL_MASS = 0.16
#: leather orange, the same hue as the arena-map ball so the finished video
#: reads as one object across simulator and replay.
BALL_RGBA = (0.82, 0.33, 0.10, 1.0)

HOOP_X, HOOP_Y = 1.45, 0.35
HOOP_INNER = round(2.0 * BALL_R + 0.010, 3)   # 0.160 — one ball + margins
HOOP_WALL_T, HOOP_RIM_H = 0.006, 0.020
HOOP_RGBA = (0.0, 0.302, 0.596, 1.0)          # Sixers blue #004D98

#: The hoop stands **above** the box: post up to a low rim, backboard behind
#: it, and the scoring tray directly under the rim — "carry to the hoop and
#: drop it in" reads as a basketball play, and the robot only has to reach
#: the box at the floor. The rim height is display, not a grasp target.
RIM_Z = 0.60
RIM_R_MAJOR, RIM_R_TUBE = 0.12, 0.012
POST_RGBA = (0.25, 0.27, 0.32, 1.0)
BOARD_RGBA = (0.85, 0.92, 0.98, 0.6)


def _prop_spec():
    spec = mujoco.MjSpec()
    spec.compiler.degree = False  # radians, like every asset in this repo
    return spec


def _prop_geom(body, **kw) -> None:
    kw.setdefault("condim", PROP_CONDIM)
    kw.setdefault("friction", list(PROP_FRICTION))
    kw.setdefault("priority", PROP_PRIORITY)
    kw.setdefault("solref", list(PROP_SOLREF))
    kw.setdefault("group", 2)
    body.add_geom(contype=1 | PROP_BIT, conaffinity=1 | PROP_BIT, **kw)


def _ball_spec():
    spec = _prop_spec()
    body = spec.worldbody.add_body(name="dunk_ball", pos=[0.0, 0.0, 0.0])
    body.add_freejoint()
    _prop_geom(body, name="dunk_ball", type=mujoco.mjtGeom.mjGEOM_SPHERE,
               size=[BALL_R, 0.0, 0.0], rgba=list(BALL_RGBA), mass=BALL_MASS)
    return spec


def _hoop_spec():
    """The hoop unit: a scoring box on the floor **under a low rim**, with the
    post and backboard that say "basketball".

    The box is the task's target and the only colliding part of the assembly
    (five_foot's dropbin scaled to the ball: 20 mm rim — the claw's measured
    lift tops out near 30 mm). The post, rim and backboard above it are
    **visual only** (`contype=0`): the robot walks under the rim to reach the
    box, and nothing overhead can bump the trunk or block a release.

    Static — no freejoint — so it stays where it is put and a missed drop
    bounces off the rim of its own box rather than sliding the target away.
    """
    spec = _prop_spec()
    body = spec.worldbody.add_body(name="dunk_hoop", pos=[0.0, 0.0, 0.0])
    inner, t, h = HOOP_INNER, HOOP_WALL_T, HOOP_RIM_H
    _prop_geom(body, name="hoop_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
               size=[inner + t, inner + t, t / 2], pos=[0.0, 0.0, t / 2],
               rgba=list(HOOP_RGBA), mass=0.4)
    for nm, sx, sy, px, py in (
        ("hoop_wall_xp", t / 2, inner + t, inner + t / 2, 0.0),
        ("hoop_wall_xm", t / 2, inner + t, -inner - t / 2, 0.0),
        ("hoop_wall_yp", inner + t, t / 2, 0.0, inner + t / 2),
        ("hoop_wall_ym", inner + t, t / 2, 0.0, -inner - t / 2),
    ):
        _prop_geom(body, name=nm, type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[sx, sy, h / 2], pos=[px, py, t + h / 2],
                   rgba=list(HOOP_RGBA), mass=0.1)

    def _visual(**kw):
        kw.setdefault("contype", 0)
        kw.setdefault("conaffinity", 0)
        kw.setdefault("group", 2)
        body.add_geom(**kw)

    # The stand rises **behind the box** (on the +x side): the robot and the
    # ball approach from -x, so nothing of the hoop silhouette ever stands
    # between the claw and its target. Visual only, but a post the robot
    # walks "through" still reads wrong in a video.
    _visual(name="hoop_post", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, RIM_Z / 2 + 0.1],
            pos=[inner + 0.14, 0.0, (RIM_Z + 0.2) / 2], rgba=list(POST_RGBA))
    # arm from post to over the box centre
    _visual(name="hoop_arm", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[(inner + 0.14) / 2, 0.02, 0.02],
            pos=[(inner + 0.14) / 2 - 0.02, 0.0, RIM_Z], rgba=list(POST_RGBA))
    # backboard behind the rim (its bottom just above the rim plane)
    _visual(name="hoop_board", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.012, 0.20, 0.15],
            pos=[inner + 0.14, 0.0, RIM_Z + 0.20], rgba=list(BOARD_RGBA))
    # the rim, directly over the box: this MuJoCo binding has no torus geom,
    # so ring it the way the arena map rings its rims — short capsules laid
    # end to end in a horizontal circle at RIM_Z (visual only)
    import math as _math
    nseg = 12
    half_len = _math.pi * RIM_R_MAJOR / nseg          # chord half-length
    for i in range(nseg):
        a = 2.0 * _math.pi * i / nseg
        # capsule axis (default +z) rotated onto the ring tangent
        # (-sin a, cos a, 0): axis-angle of z -> tangent
        q = _quat_z_to((-_math.sin(a), _math.cos(a), 0.0))
        _visual(name=f"hoop_rim_{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=[RIM_R_TUBE, half_len, 0.0],
                pos=[RIM_R_MAJOR * _math.cos(a), RIM_R_MAJOR * _math.sin(a), RIM_Z],
                quat=q, rgba=[0.898, 0.278, 0.098, 1.0])  # orange, arena-matched
    return spec


def _quat_z_to(vec):
    """Quaternion (w, x, y, z) rotating +z onto `vec` (MJCF/Spec order)."""
    import numpy as np
    v = np.asarray(vec, dtype=float)
    v = v / np.linalg.norm(v)
    axis = np.cross([0.0, 0.0, 1.0], v)
    s = float(np.linalg.norm(axis))
    c = float(np.dot([0.0, 0.0, 1.0], v))
    if s < 1e-9:
        return [1.0, 0.0, 0.0, 0.0] if c > 0 else [0.0, 1.0, 0.0, 0.0]
    ang = float(np.arctan2(s, c))
    ax = axis / s
    return [float(np.cos(ang / 2)), *(float(x) for x in (ax * np.sin(ang / 2)))]


def dunk_entities() -> dict[str, EntityCfg]:
    """The two props, at their scene positions (relative to each env origin,
    the way five_foot's row places its props)."""
    return {
        "dunk_ball": EntityCfg(
            spec_fn=_ball_spec,
            init_state=EntityCfg.InitialStateCfg(
                pos=(BALL_X, BALL_Y, BALL_R + 0.002))),   # a hair above rest:
        #   settles, never starts buried (objects.py: burial ejects the prop)
        "dunk_hoop": EntityCfg(
            spec_fn=_hoop_spec,
            init_state=EntityCfg.InitialStateCfg(pos=(HOOP_X, HOOP_Y, 0.0))),
    }
