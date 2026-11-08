"""Staged dunk terms for the THREE-STAION task (v12.1), plus observations.

The operator's program of 2026-11-08: three balls, three cups, each ball
paired with its own hole — sink one, then **run to the next firing position**
("每完成一次进球，要主动跑到下一个击球点"), and after all three, the crown
celebration is a separate task (`sixers_crown`).

There is no scripted order. The live job per environment is the **nearest
ball not yet delivered**:

    target        argmin over balls of robot→ball distance, skipping balls
                  already in *their own* cup (ball i scores in cup i only —
                  `props.CUPS_XY` is laid out so no fairway can sink into a
                  neighbour's cup by accident);

and the stages follow that one target:

    stage 1   approach_ball      exp cone on robot→target;
    stage 1b  ball_progress      per-step credit for closing that distance;
    stage 2   carry_to_hoop      exp cone on target→own cup, paid only while
                                 the claw **touches the target** (the v9/v10
                                 bulldozer gate — a kicked ball earns nothing);
    stage 2b  hoop_progress      per-step credit for the target closing on
                                 its cup under the same contact gate;
    score     ball_in_hoop       fraction of the three balls delivered —
                                 dense: each sink raises the payout by 1/3
                                 and holds it.

So the run between stations pays on its own: standing at a sunk cup earns
approach only from wherever the next target happens to be, and the gradient
of `ball_progress` points straight at it. Per-ball contact comes from one
`ContactSensorCfg` per ball (env_cfg wires `claw_ball_contact_{i}`), stacked
into a [B, N] matrix and gathered at the target index.

No camera anywhere and none needed: every ball's offset, every cup's offset,
and the delivered count are fed to the policy as observations directly.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.manager_base import ManagerTermBase

from ..props import BALL_R, HOOP_INNER, BALL_NAMES, CUP_NAMES
from ...five_foot.claw import FINGER_JOINT

#: The task's world, as `props` authored it. Entity names are unique, so
#: mjlab's per-entity prefixing (`scene.py::attach(prefix="{name}/")`) keeps
#: the identically-named bodies inside each prop spec apart.
N_BALLS = len(BALL_NAMES)
BALL_SENSORS: tuple[str, ...] = tuple(f"claw_ball_contact_{i}"
                                      for i in range(N_BALLS))

#: A ball counts as **delivered** when its centre lies inside this radius of
#: *its own* cup's centre. The wall's inner face is `HOOP_INNER`; a ball
#: resting against any wall has its centre `HOOP_INNER − BALL_R` away, and
#: one waiting just outside the mouth is `HOOP_INNER + t + BALL_R` or more
#: away, so `HOOP_INNER` cleanly contains every in-and-at-rest centre and
#: excludes every out-and-stopped one (verified on the v12.1 layout: the
#: tightest legitimate outside wait is ~0.20 m against a 0.115 m line).
SCORE_R = round(HOOP_INNER, 3)


# ── world reads ──────────────────────────────────────────────────────────────

# `root_link_pose_w` is already world frame (`rl/mjlab/entity/data.py`:
# `data.xpos[:, root_body_id]`, "All inputs/outputs use world frame"), and each
# env's props and robot live in that env's world — positions are directly
# comparable, with no origin math. five_foot's `env_origins` uses subtract the
# origin only to get a *height above the floor*.

def _ball_pos(env: ManagerBasedRlEnv, name: str = BALL_NAMES[0]) -> torch.Tensor:
    return env.scene[name].data.root_link_pose_w[:, :3]


def _balls_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Every ball's centre, stacked [B, N, 3] in `BALL_NAMES` order."""
    return torch.stack([_ball_pos(env, name) for name in BALL_NAMES], dim=1)


def _hoop_pos(env: ManagerBasedRlEnv, i: int = 0) -> torch.Tensor:
    return env.scene[CUP_NAMES[i]].data.root_link_pose_w[:, :3]


def _cups_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Every cup's centre, stacked [B, N, 3] in `CUP_NAMES` order."""
    return torch.stack([_hoop_pos(env, i) for i in range(N_BALLS)], dim=1)


def _robot_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_link_pose_w[:, :3]


def _ball_lift(env: ManagerBasedRlEnv, name: str = BALL_NAMES[0]) -> torch.Tensor:
    """Height of one ball's centre above its resting height on this env's
    floor — `env_origins[:, 2]` is subtracted the way five_foot's height
    terms do, so a tilted terrain patch cannot fake a lift."""
    return _ball_pos(env, name)[:, 2] - env.scene.env_origins[:, 2] - BALL_R


_SHUT_APERTURE = -0.2   # rad; between GRIPPER_OPEN -0.65 and CLOSED 0.10,
#                        past halfway means the mouth has taken the ball


def _aperture(env: ManagerBasedRlEnv) -> torch.Tensor:
    """The finger joint's own position [B] — the aperture the policy drives
    since v7 put LF_J4 in the action. Resolved by the entity's name API."""
    robot = env.scene["robot"]
    idx = getattr(env, "_dunk_finger_idx", None)
    if idx is None:
        ids, _ = robot.find_joints([FINGER_JOINT])
        idx = int(ids[0])
        env._dunk_finger_idx = idx
    return robot.data.joint_pos[:, idx]


def _claw_touch(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """[B] bool: any jaw geom of the carried claw is touching one ball.

    v9's answer to eight runs of evidence that the **pinch is not this
    claw's way with a floor-level ball**: the palm hangs ~40 mm off the
    ground (`five_foot/env_cfg.py` says so outright), so contact happens on
    the under-jaw, not across the mouth. The task accepts the mechanic the
    hardware actually offers — **bulldozing**: the claw face presses the ball
    and walks it. This sensor says the walk is *controlled*, not a kick.
    """
    found = env.scene[sensor_name].data.found
    return found.abs().sum(dim=tuple(range(1, found.dim()))) > 0


def _touches(env: ManagerBasedRlEnv) -> torch.Tensor:
    """[B, N] bool: which balls the claw is touching right now."""
    return torch.stack([_claw_touch(env, s) for s in BALL_SENSORS], dim=1)


def delivered_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
    """[B, N] bool: ball i is **in its own cup** — centre within `SCORE_R`
    of cup i. Height is not tested: the cup is flush with the terrain (v11.1)
    and its 35 mm wall is unclimbable, so every centre inside the ring is a
    ball that has arrived and stayed."""
    b, c = _balls_pos(env), _cups_pos(env)
    d = torch.linalg.norm(b[:, :, :2] - c[:, :, :2], dim=2)
    return d < SCORE_R


def target_ball(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """The live job per environment: (`idx` [B] long, `active` [B] bool).

    The **nearest ball not yet delivered to its own cup** — no scripted
    order, one job at a time. `active` says the environment still owes at
    least one sink; once all three balls are in, every stage term goes
    silent and only the score keeps paying, which is the hand-off point to
    the crown choreography (a separate task).
    """
    dv = delivered_mask(env)
    d = torch.linalg.norm(_balls_pos(env)[:, :, :2]
                          - _robot_pos(env)[:, None, :2], dim=2)
    d = d.masked_fill(dv, float("inf"))
    idx = torch.argmin(d, dim=1)
    active = ~dv.all(dim=1)
    return idx, active


def _pick(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather row `idx[b]` of `t[b, ...]` — [B, ...K] from [B, N, ...K] (or
    [B, N] for a per-element pick)."""
    shape = [-1, 1] + [1] * (t.dim() - 2)
    view = idx.view(*shape).expand(-1, 1, *t.shape[2:])
    return torch.gather(t, 1, view).squeeze(1)


# ── observations ─────────────────────────────────────────────────────────────


def ball_rel_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    """All three balls' offsets from the robot's base, flattened [B, 3N].

    Deliberately the *offset from the robot's base*, not the raw world pos:
    the policy's other inputs are egocentric, and a world coordinate would
    make the same skill look different on env 0 and env 1024. Every ball is
    shown, not just the target — the policy may line up the next shot early,
    and the target rule simply pays whichever ball it is closing on.
    """
    return (_balls_pos(env) - _robot_pos(env)[:, None, :]).flatten(1)


def ball_score_count(env: ManagerBasedRlEnv) -> torch.Tensor:
    """[B, 1] in [0, 1]: fraction of the three balls delivered. The reward
    broadcasts the same fact (it IS the score), but observations let the
    critic value a finished green without replaying the sinks."""
    return delivered_mask(env).float().mean(dim=1, keepdim=True)


def hoop_rel_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    """All three cups' offsets from the robot, flattened [B, 3N] — paired
    with `ball_rel_pos` by index, so the policy can see which hole belongs
    to which ball."""
    return (_cups_pos(env) - _robot_pos(env)[:, None, :]).flatten(1)


# ── per-step progress (stateful) ─────────────────────────────────────────────


class _TargetProgress(ManagerTermBase):
    """Shared engine: credit the per-step closing of ONE distance measured
    to the *current target*, which may change identity between steps.

    Subclasses define the distance (robot→ball, or ball→own cup). Two
    guards, both learned the hard way in earlier versions:

      * **the step after any reset returns 0** — the placement events
        teleport every ball back to its spawn, and a raw prev-minus would
        print that as metres of free progress (stored distance NaNs out);
      * **a step where the target changes identity returns 0** — when the
        target ball drops into its cup the job jumps to the next ball,
        whose distance may happen to be *smaller*; without the identity
        guard that swap itself would pay out.

    Values are clipped to [0, 1] by `v_full` m/s of closing: standing pays
    0, walking pays partial, a full-speed bulldozer push pays a full step.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        params = getattr(cfg, "params", {}) or {}
        self.v_full = float(params.get("v_full", 0.12))
        self._prev = None
        self._prev_idx = None

    def reset(self, env_ids):
        if self._prev is None:
            return
        if env_ids is None or isinstance(env_ids, slice):
            self._prev[:] = float("nan")
            self._prev_idx[:] = -1
        else:
            self._prev[env_ids] = float("nan")
            self._prev_idx[env_ids] = -1

    def distance(self, env: ManagerBasedRlEnv, idx: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def gate(self, env: ManagerBasedRlEnv, idx: torch.Tensor,
             active: torch.Tensor) -> torch.Tensor:
        return active.float()

    def __call__(self, env: ManagerBasedRlEnv, **_params) -> torch.Tensor:
        idx, active = target_ball(env)
        d = self.distance(env, idx)
        if self._prev is None:
            self._prev = d.clone()
            self._prev_idx = idx.clone()
            return torch.zeros_like(d)
        gain = ((self._prev - d) / self.v_full).clamp(0.0, 1.0)
        stale = self._prev.isnan() | (idx != self._prev_idx) | ~active
        gain = torch.where(stale, torch.zeros_like(gain), gain)
        gain = gain * self.gate(env, idx, active)
        self._prev = d
        self._prev_idx = idx
        return gain


class BallProgress(_TargetProgress):
    """Stage 1b: closing on the **target ball** itself (robot→ball). The
    run-from-station term: after one sink the new target is a different ball
    0.8–1.3 m away, and every step toward it pays."""

    def distance(self, env, idx):
        b = _pick(_balls_pos(env), idx)
        r = _robot_pos(env)
        return torch.linalg.norm(b[:, :2] - r[:, :2], dim=1)


class BallToHoopProgress(_TargetProgress):
    """Stage 2b: the **target ball** closing on *its own* cup (ball→cup),
    paid only while the claw is touching that ball — the answer to the
    operator's "the claw taps the ball, it rolls away, what then?": a kick
    earns 0, walking it earns every centimetre, and stage 1's gradient
    already drags the robot back to whatever ball is nearest and undelivered."""

    def distance(self, env, idx):
        b = _pick(_balls_pos(env), idx)
        c = _pick(_cups_pos(env), idx)
        return torch.linalg.norm(b[:, :2] - c[:, :2], dim=1)

    def gate(self, env, idx, active):
        touch = _pick(_touches(env), idx)
        return (active & touch).float()


# ── rewards ──────────────────────────────────────────────────────────────────


def approach_ball(env: ManagerBasedRlEnv, sigma: float = 0.35,
                  leave_r: float = 0.30) -> torch.Tensor:
    """ exp(−d²/σ²) on the horizontal robot→**target-ball** distance.

    Three switches, each a lesson from v6–v10: silent while the claw is on
    the ball walking it (stage 2 owns that), silent once the target sits
    near *its own* cup (`leave_r`, handing the distance budget to stage 2 so
    a parked ball cannot feed stage 1 forever), and silent when no ball is
    pending (`active` — all three delivered).
    """
    idx, active = target_ball(env)
    b = _pick(_balls_pos(env), idx)
    c = _pick(_cups_pos(env), idx)
    d_xy = torch.linalg.norm(b[:, :2] - _robot_pos(env)[:, :2], dim=1)
    near_hoop = torch.linalg.norm(b[:, :2] - c[:, :2], dim=1) <= leave_r
    walking = _pick(_touches(env), idx)
    return torch.exp(-(d_xy / sigma) ** 2) * (active & ~near_hoop & ~walking).float()


def ball_lifted(env: ManagerBasedRlEnv, cap: float = 0.12) -> torch.Tensor:
    """RETIRED (weight 0 since v10), kept so old replay layouts resolve:
    ball height above rest. Ten runs proved the pinch-lift is not this
    claw's mechanic; the three-station task scores pushes, not lifts."""
    lift = _ball_lift(env).clamp(min=0.0)
    return (lift / cap).clamp(max=1.0)


def carry_to_hoop(env: ManagerBasedRlEnv, sigma: float = 0.35) -> torch.Tensor:
    """Stage 2 cone: **target ball→its own cup** distance paid while the
    claw touches the target — the coarse "arrive with it" shaping; the dense
    half of the same skill is `BallToHoopProgress`. A ball kicked free rolls
    without earning; a ball walked earns the whole fairway.

    (v12 dropped the `gate="grip"/"lift"` variants for good: they gated on
    a pinch this claw has never once performed — see the module docstring.)
    """
    idx, active = target_ball(env)
    b = _pick(_balls_pos(env), idx)
    c = _pick(_cups_pos(env), idx)
    d = torch.linalg.norm(b - c, dim=1)
    held = _pick(_touches(env), idx)
    return torch.exp(-(d / sigma) ** 2) * (active & held).float()


def ball_in_hoop(env: ManagerBasedRlEnv) -> torch.Tensor:
    """The score: **fraction of the three balls in their cups** — 1/3 per
    delivered ball, held for the rest of the episode. Re-read every step,
    so the mask is honest in both directions (a ball nudged back out of the
    mouth costs the fraction; the wall is unclimbable, so this is a rare
    self-inflicted accident, not a design case).

    15.0 weight means a finished round pays 15/step against ~9/step of
    approach+carry for a single ball mid-fairway: three sinks is the whole
    point of the task, and the reward design exists to make exactly that
    posture the cheapest way to earn it."""
    return delivered_mask(env).float().mean(dim=1)
