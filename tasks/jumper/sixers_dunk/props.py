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
#: v12.1 (operator, 2026-11-08): **three balls AND three cups**, each ball
#: paired with its own hole, the stations staggered across the court ("三组
#: 小球和圆洞要有位置上的落差") so that after sinking one the robot has to
#: **run to the next firing position** — the reward follows the nearest
#: undelivered ball, so the run between stations is the task's spine, not an
#: afterthought. Ball i scores in cup i (`BALLS_XY[i]` ↔ `CUPS_XY[i]`); the
#: pairings below are chosen so no fairway pushes a ball into a *neighbour's*
#: cup (every cross distance stays above `SCORE_R` + ball radius once the
#: ball is short of its own mouth).
BALLS_XY: tuple[tuple[float, float], ...] = (
    (0.35, -0.35),      # station 0: right of centre, the short shot
    (0.50,  0.55),      # station 1: far left, reached after the first sink
    (0.95,  0.05),      # station 2: centre, the deep one up the fairway
)
CUPS_XY: tuple[tuple[float, float], ...] = (
    (1.15, -0.55),      # cup 0: 0.85 m off station 0's ball
    (1.05,  0.85),      # cup 1: 0.95 m off station 1's ball
    (1.85,  0.05),      # cup 2: 0.90 m, dead down the middle
)
BALL_NAMES: tuple[str, ...] = tuple(f"dunk_ball_{i}"
                                    for i in range(len(BALLS_XY)))
CUP_NAMES: tuple[str, ...] = tuple(f"dunk_cup_{i}"
                                   for i in range(len(CUPS_XY)))
#: 30 g — a superball. Held across two jaws at mu_s 0.35 the pinch needs only
#: N >= mg / 2mu ~= 0.42 N, and the actuator tops out at `STIFFNESS *
#: SQUEEZE_LEAD` = 0.6 N·m at the fingertip — the grip has room to spare.
#: (The 160 g of v1-v7 was sized for the old 150 mm ball.)
BALL_MASS = 0.03
#: leather orange, the same hue family as the arena's court markings read
#: against; the arena map's own game ball was removed at v10 so this is the
#: only ball in the video.
BALL_RGBA = (0.82, 0.33, 0.10, 1.0)

#: v12.1: **three cups, one per ball**, at `CUPS_XY`. Each is the v11.2
#: design — a continuous arc wall standing flush on the terrain with the
#: mouth facing the robot's side of the court; sized to hold its own ball
#: (and, come a happy accident, all three).
#: v12: the cup interior is `HOOP_INNER` = 0.115, a container three balls
#: could share (triangle packing needs `BALL_R * (1 + 2/sqrt(3)) ~= 0.105`),
#: so no cup is too tight for a stray.
HOOP_INNER = 0.115
#: v11 (operator review, 2026-11-08): the square box became a **circular
#: cup** — a golf hole at the end of the fairway, as asked ("用前臂推小球
#: 进洞，类似打高尔夫").
#: v10 (operator review): the post/arm/backboard/rim stand is **gone** — ten
#: runs proved the claw cannot get the ball off the floor at all, so nothing
#: overhead was ever part of the task. The cup takes the whole role.
#: v11.1 (operator): **no raised floor, no sill** — just a ring wall standing
#: on the terrain, its open mouth flush with the ground: the ball rolls in
#: with nothing to climb ("丝滑"). Hold is not lost: lifting a 40 mm ball
#: over a 35 mm step needs ~8x the ball's weight in push (F = mg*sqrt(2Rh-
#: h^2)/(R-h)), which the walk never exerts sideways — in through the mouth,
#: stays in.
HOOP_WALL_T, HOOP_RIM_H = 0.006, 0.035
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


def _hoop_spec(body_name: str = "dunk_hoop"):
    """The scoring cup: a **continuous arc wall** standing directly on the
    terrain, with a flush mouth on the robot's side (v11.2; see the `HOOP_*`
    notes).

    No floor disc, no sill, no posts. MuJoCo has no hollow-cylinder or torus
    primitive in this binding, so the ring is built the way a real curved
    retaining wall is built: **`nseg` thin panels laid along the tangents of
    one circle**, each spanning its chord and overlapping its neighbours by a
    few millimetres so the joints are solid — one continuous 6 mm-thick,
    35 mm-tall wall, flat on the inside and outside, that a pushed ball can
    neither slip through nor ride over. The mouth is simply the one arc of
    panels left out at -x.

    Static — no freejoint — so it stays where it is put and a missed push
    bounces off the wall of its own cup rather than sliding the target away.
    """
    spec = _prop_spec()
    body = spec.worldbody.add_body(name=body_name, pos=[0.0, 0.0, 0.0])
    inner, t, h = HOOP_INNER, HOOP_WALL_T, HOOP_RIM_H
    import math as _math
    r_wall = inner + t / 2                 # panel mid-surface radius
    nseg = 24
    step = 2.0 * _math.pi / nseg
    half_chord = r_wall * _math.tan(step / 2) + 0.004   # + overlap at joints
    mouth = _math.asin(min(0.95, 0.060 / r_wall))       # half-angle of the gap
    for i in range(nseg):
        a = step * i
        da = abs((a - _math.pi + _math.pi) % (2 * _math.pi) - _math.pi)  # to 180°
        if da < mouth:
            continue                                    # inside the mouth
        # the panel is a box whose +x axis lies on the circle's TANGENT at its
        # midpoint: a tangent at angle `a` is the +x axis turned by a + 90 deg
        # (quaternion w,x,y,z of that z-rotation).
        ang = a + _math.pi / 2
        qw, qs = _math.cos(ang / 2), _math.sin(ang / 2)
        _prop_geom(body, name=f"hoop_wall_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[half_chord, t / 2, h / 2],
                   pos=[r_wall * _math.cos(a), r_wall * _math.sin(a), h / 2],
                   quat=[qw, 0.0, 0.0, qs],
                   rgba=list(HOOP_RGBA), mass=0.03)
    return spec


def dunk_entities() -> dict[str, EntityCfg]:
    """The three balls and their three cups, at their scene positions
    (relative to each env origin, the way five_foot's row places its props).
    Entity names are unique, so every body inside a spec may keep the plain
    name it was authored with: mjlab attaches each entity under a
    `"{entity}/"` prefix (`rl/mjlab/scene/scene.py`)."""
    entities = {
        name: EntityCfg(
            spec_fn=_ball_spec,
            init_state=EntityCfg.InitialStateCfg(
                pos=(x, y, BALL_R + 0.002)))   # a hair above rest:
        #   settles, never starts buried (objects.py: burial ejects the prop)
        for name, (x, y) in zip(BALL_NAMES, BALLS_XY)
    }
    entities.update({
        name: EntityCfg(
            spec_fn=lambda n=name: _hoop_spec(n),
            init_state=EntityCfg.InitialStateCfg(pos=(x, y, 0.0)))
        for name, (x, y) in zip(CUP_NAMES, CUPS_XY)
    })
    return entities
