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
#: **v10.5 (operator review, 2026-11-08): 80 mm.** Two reasons beyond "the
#: box is hard to enter": a bigger ball rolls over the sill with a smaller
#: push (step-climb force ~ sqrt(2Rh)/(R−h) × mg, and R is now 3× the sill
#: step), and its **equator sits at 40 mm — exactly the height the stowed
#: palm hangs at**. At 64 mm the mouth could only ever be *under* the ball;
#: at 80 mm the jaws can be across its sides, so the pinch is back within
#: reach as an emergent skill, not just the shove. The claw opens past 80 mm
#: only at finger < -0.75 (measured 82.3 mm at -0.75 in `claw.py`), which
#: the v7 finger action covers — the policy has to *choose* to pre-open.
BALL_R = 0.040
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
#: v11 (operator review, 2026-11-08): the square box is now a **circular
#: cup** — a golf hole at the end of the fairway, as asked ("用前臂推小球
#: 进洞，类似打高尔夫"). Scoring radius: one ball + a 12 mm annular margin
#: (`HOOP_INNER` = 0.052): tighter than the box's 2-ball mouth, because a
#: contained cup — not a doorstep — is what holds the ball once it is in.
HOOP_INNER = round(BALL_R + 0.012, 3)   # 0.052 — the cup, ~1 ball + margin
#: v10 (operator review): the post/arm/backboard/rim stand is **gone** — ten
#: runs proved the claw cannot get the ball off the floor at all, so nothing
#: overhead was ever part of the task. The cup takes the whole role, and its
#: wall rises 35 mm. The entrance toward the robot stays a 6 mm sill.
HOOP_WALL_T, HOOP_RIM_H = 0.006, 0.035
#: v10.5: sill 8 -> 6 mm with the bigger ball — the entrance step is now
#: 12 mm total (floor + sill) against a 40 mm radius: a push the walk easily
#: produces (the lip takes F >= mg*sqrt(2Rh-h^2)/(R-h) ~ 0.35 mg, and a
#: walking hexapod pushes at several times the ball's weight), and once the
#: ball is inside, the 41 mm wall top is **above the ball's own centre
#: height** — no rolling ball climbs a step taller than its radius, so the
#: cup holds. Enter over the sill; the trap is one-way.
HOOP_SILL_H = 0.006   # the approach-side lip, kept climbable
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
    """The scoring cup: a circle of touching vertical capsules sunk around a
    floor disc, with a low lip across the one open arc the robot pushes over
    (v11; see the `HOOP_*` notes for the one-way geometry).

    The wall is one ring of capsules at `inner + t` laid end to end with a
    deliberate gap on the -x side — the mouth of the cup, ~2 ball-widths
    wide, spanned by a `HOOP_SILL_H` lip. MuJoCo has no boolean subtraction,
    so "hole" is built the only way it can be: a rim *on* the floor, small
    enough inside that a ball at rest in the centre reads as sunk.

    Static — no freejoint — so it stays where it is put and a missed push
    bounces off the wall of its own cup rather than sliding the target away.
    """
    spec = _prop_spec()
    body = spec.worldbody.add_body(name="dunk_hoop", pos=[0.0, 0.0, 0.0])
    inner, t, h = HOOP_INNER, HOOP_WALL_T, HOOP_RIM_H
    _prop_geom(body, name="hoop_floor", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
               size=[inner + t, t / 2, 0.0], pos=[0.0, 0.0, t / 2],
               rgba=list(HOOP_RGBA), mass=0.4)
    # The cup mouth faces -x (the robot's side): the arc of capsules whose
    # angle from +x is within `mouth` of 180 deg is left out, and the
    # opening is spanned at sill height by one thin bar. Wall capsule
    # centres sit a little outside `inner + t` so neighbours touch across
    # their curved faces; the two flanking the gap are kept just clear of
    # the lip's ends instead of blocking them.
    import math as _math
    r_wall = inner + t + 0.002
    cap_r = 0.005
    nseg = 28
    mouth = _math.asin(min(1.0, 0.048 / r_wall))      # half-angle of the gap
    for i in range(nseg):
        a = 2.0 * _math.pi * i / nseg
        da = abs((a - _math.pi + _math.pi) % (2 * _math.pi) - _math.pi)  # to 180°
        if da < mouth + 2 * _math.pi / nseg:
            continue                                    # inside the mouth
        _prop_geom(body, name=f"hoop_wall_{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                   size=[cap_r, h / 2, 0.0],
                   pos=[r_wall * _math.cos(a), r_wall * _math.sin(a), t + h / 2],
                   rgba=list(HOOP_RGBA), mass=0.02)
    _prop_geom(body, name="hoop_sill", type=mujoco.mjtGeom.mjGEOM_BOX,
               size=[t / 2, 0.046, HOOP_SILL_H / 2],
               pos=[-inner - t / 2, 0.0, t + HOOP_SILL_H / 2],
               rgba=list(HOOP_RGBA), mass=0.05)
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
