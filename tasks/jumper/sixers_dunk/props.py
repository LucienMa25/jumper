"""The dunk world: one ball to grasp and one hoop to score in.

Two entities, authored the way `five_foot/objects.py` authors its row — an
`MjSpec` per prop, **in radians** (the trap that file documents: MjSpec
defaults to degrees and a radian-authored euler silently rotates nothing),
collision on the prop channel, `PROP_*` contact parameters imported from
five_foot rather than re-declared, so the two tasks' props cannot drift into
different physics.

Layout (x ahead of the robot's spawn, y lateral):

    robot @ 0        ball @ BALL_X=0.35        box @ HOOP_X=1.20, y=HOOP_Y=0.30

v10 (operator review): the ball now spawns **dead ahead** — the mechanic this
claw can actually perform is pushing the ball forward from behind, and a
diagonal only taught flanking. The hoop unit is the scoring box alone: the
basketball stand was removed once measurement showed the claw never gets the
ball off the floor, at any height (see `_hoop_spec`). The court and its NBA
hoops come from the arena scene (`--scene`), so the video still reads as
basketball while the physics asks only for what the machine can do.

The box walls are `HOOP_RIM_H` = 35 mm on the three far sides and an 8 mm
sill on the approach side — higher than five_foot's dropbin rim, but the
entrance stays shallow because a ball rolled along the floor cannot be shoved
up a step near its own radius. Far walls are Sixers blue; the ball is leather
orange.
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

#: Ball: **scaled to the claw's measured mouth, not to a real basketball.**
#: `five_foot/claw.py` measures the jaw aperture across the pinch at 73.3 mm
#: at `GRIPPER_OPEN` (-0.65), 82.6 mm at the widest usable -0.75, and authors
#: its largest grippable prop — a 58 mm can — 15 mm inside that. v1-v7 ran a
#: 150 mm ball: the pinch was **physically impossible**, which is why
#: `ball_lifted` sat at chance level no matter what the policy or the action
#: space tried. 56 mm diameter sits exactly on the can's documented budget,
#: and at the robot's ~40 cm scale it still reads as a toy basketball on
#: video. **v9: 64 mm.** v8 proved a 56 mm ball rolls out from under the palm
#: before the jaws reach it — the smallest pushable prop the official table
#: lists is 66 mm across, so the ball is set just inside that: still a hair
#: under the pinch envelope (73 mm at `GRIPPER_OPEN`) for any future grip
#: attempt, and big enough that the palm face meets it, not the floor.
BALL_R = 0.032
#: v3: was (0.55, -0.20) — d=0.585 m put the high-value region too far for
# the entropy budget to bump into it by iteration ~900 (v2 measurement).
# 0.43 m now: still a walk, reachable by a exploring policy within its first
# steps of interest.
#: v6: the needle goes into the haystack. v5 (4000 iters x 1024 envs, 41M
# steps, std held at 0.79) never discovered goal-directed walking from 0.43 m
# -- a random walk carries no bearing prior, so no reward gradient was ever
# *touched* to learn from. The ball now spawns 0.19 m ahead: inside the first
# flail's reach. Combined with the sharpened cone below, guarding the ball
# becomes the new (intended) comfort zone, and progress/lift unlock outward.
#: v9: dead ahead at 0.20 m. v8's y=-0.05 only taught the policy to flank the
# ball; a bulldozer drives straight, and so does the claw-palm here.
#: v10 (operator review, 2026-11-08): 0.20 m sat **inside** the approach
# cone's fat end — spawn already paid ~5.7/8 and the replay showed a robot
# that parks next to its own ball. 0.35 m cuts the spawn value to ~2.9/8:
# the rest of that cone has to be walked for.
BALL_X, BALL_Y = 0.35, -0.02
#: 30 g — a superball. Held across two jaws at mu_s 0.35 the pinch needs only
#: N >= mg / 2mu ~= 0.42 N, and the actuator tops out at `STIFFNESS *
#: SQUEEZE_LEAD` = 0.6 N·m at the fingertip — the grip has room to spare.
#: (The 160 g of v1-v7 was sized for the old 150 mm ball.)
BALL_MASS = 0.03
#: leather orange, the same hue family as the arena's court markings read
#: against; the arena map's own game ball was removed at v10 so this is the
#: only ball in the video.
BALL_RGBA = (0.82, 0.33, 0.10, 1.0)

HOOP_X, HOOP_Y = 1.20, 0.30
#: Three balls across, not one: the release is a blind drop off an open claw
#: from a few centimetres up, and a one-ball box would bounce most near-misses
#: back out — the score must measure "got it to the hoop", not pinball luck.
#: (v1-v7: this derived 0.160 for the old 150 mm ball and never got used.)
HOOP_INNER = round(3.5 * BALL_R + 0.010, 3)   # 0.118 — ~2 ball-widths across
#: v10 (operator review): the post/arm/backboard/rim stand is **gone** — ten
#: runs proved the claw cannot get the ball off the floor at all, so nothing
#: overhead was ever part of the task. The box takes the whole role, and the
#: walls rise from 20 to 35 mm as asked ("做高一点"). The near wall stays a
#: 8 mm sill: a ball pushed along the floor cannot climb a wall taller than a
#: fraction of its radius (32 mm), so a full-height near wall would make the
#: box a monument, not a target. Enter over the sill, ringed by high walls.
HOOP_WALL_T, HOOP_RIM_H = 0.006, 0.035
HOOP_SILL_H = 0.008   # the approach-side wall, kept climbable
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
    """The scoring box, standing alone on the floor.

    **v10: the basketball stand is gone.** Ten runs measured that this claw
    never gets the ball off the floor at all (the palm hangs 40 mm up, the
    ball's centre is 32 mm), so a rim at any height — even a lowered one —
    describes a throw the machine cannot make. The operator's call (2026-11-08):
    keep only the box and make it taller. The arena map still carries the NBA
    hoops and the court, so the *video* reads as basketball; the *task* is the
    honest one: push the ball into the box.

    The walls rise to `HOOP_RIM_H` (35 mm) on the three far sides, and the
    **near wall — the side the robot comes from — is an 8 mm sill**: a ball
    rolled along the floor cannot be shoved up a step anywhere near its own
    radius, so the entrance has to stay shallow or the box is a monument.

    Static — no freejoint — so it stays where it is put and a missed push
    bounces off the wall of its own box rather than sliding the target away.
    """
    spec = _prop_spec()
    body = spec.worldbody.add_body(name="dunk_hoop", pos=[0.0, 0.0, 0.0])
    inner, t, h = HOOP_INNER, HOOP_WALL_T, HOOP_RIM_H
    _prop_geom(body, name="hoop_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
               size=[inner + t, inner + t, t / 2], pos=[0.0, 0.0, t / 2],
               rgba=list(HOOP_RGBA), mass=0.4)
    for nm, sx, sy, px, py, wh in (
        ("hoop_wall_xp", t / 2, inner + t, inner + t / 2, 0.0, h),
        ("hoop_wall_xm", t / 2, inner + t, -inner - t / 2, 0.0, HOOP_SILL_H),
        ("hoop_wall_yp", inner + t, t / 2, 0.0, inner + t / 2, h),
        ("hoop_wall_ym", inner + t, t / 2, 0.0, -inner - t / 2, h),
    ):
        _prop_geom(body, name=nm, type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[sx, sy, wh / 2], pos=[px, py, t + wh / 2],
                   rgba=list(HOOP_RGBA), mass=0.1)
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
