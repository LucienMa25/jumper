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
BALL_X, BALL_Y = 0.55, -0.20
BALL_MASS = 0.16
#: leather orange, the same hue as the arena-map ball so the finished video
#: reads as one object across simulator and replay.
BALL_RGBA = (0.82, 0.33, 0.10, 1.0)

HOOP_X, HOOP_Y = 1.45, 0.35
HOOP_INNER = round(2.0 * BALL_R + 0.010, 3)   # 0.160 — one ball + margins
HOOP_WALL_T, HOOP_RIM_H = 0.006, 0.020
HOOP_RGBA = (0.0, 0.302, 0.596, 1.0)          # Sixers blue #004D98


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
    """Static pedestal tray: floor + four blue walls, open top.

    No freejoint — it never moves; a missed release bounces off the rim
    instead of sliding the target away from the task's own scoring geometry.
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
    # A tall post behind the hoop, visual only: the "backboard" cue in video.
    body.add_geom(name="hoop_post", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[0.006, 0.18, 0.28], pos=[-inner - 0.02, 0.0, 0.30],
                  rgba=[1.0, 1.0, 1.0, 1.0], contype=0, conaffinity=0, group=2)
    return spec


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
