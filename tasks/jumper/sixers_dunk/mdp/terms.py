"""Staged dunk terms: approach → grip → carry → score, plus the observations.

Each stage is a term the policy can feel *before* the next one unlocks, the
same ladder the command curriculum uses for walking: nothing here pays for
the whole trick at once.

    stage 1  approach_ball   exp(-d^2/sigma^2) while the claw is not closed
    stage 1b ball_progress   per-step credit for shrinking that distance
    stage 2  ball_lifted     ball height above rest (only a grip can hold it)
    stage 3  carry_to_hoop   -dist(ball, hoop) while the ball is GRIPPED
    stage 4  ball_in_hoop    ball centre inside the rim footprint, under the
                             rim line — the score

v7 moved the grip inside the policy: `env_cfg` puts LF_J4 in the action, and
the carry stage gates on `grip` — the same reading the claw's own teleop
uses (aperture shut *and* ball off the floor), so a rolled or flicked ball
no longer unlocks stage 3. The aperture is the finger joint's own position,
read from the state every observation term already lives on.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.manager_base import ManagerTermBase

from ..props import BALL_R, HOOP_INNER
from ...five_foot.claw import FINGER_JOINT

def _ball_lift(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Height of the ball centre above its resting height on this env's
    floor — `env_origins[:, 2]` is subtracted the way five_foot's height
    terms do, so a tilted terrain patch cannot fake a lift."""
    return _ball_pos(env)[:, 2] - env.scene.env_origins[:, 2] - BALL_R


# `root_link_pose_w` is already world frame (`rl/mjlab/entity/data.py`:
# `data.xpos[:, root_body_id]`, "All inputs/outputs use world frame"), and each
# env's props and robot live in that env's world — positions are directly
# comparable, with no origin math. five_foot's `env_origins` uses subtract the
# origin only to get a *height above the floor*.

def _ball_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.scene["dunk_ball"].data.root_link_pose_w[:, :3]


def _hoop_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.scene["dunk_hoop"].data.root_link_pose_w[:, :3]


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


def _robot_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_link_pose_w[:, :3]


def _claw_touch(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """[B] bool: any jaw geom of the carried claw is touching the ball.

    v9's answer to eight runs of evidence that the **pinch is not this
    claw's way with a floor-level ball**: the palm hangs ~40 mm off the
    ground (`five_foot/env_cfg.py` says so outright) while a 64 mm ball's
    centre sits at 32 mm, so contact happens on the under-jaw, not across
    the mouth, and the lift reward never left chance level. The task
    therefore accepts the mechanic the hardware actually offers —
    **bulldozing**: the claw face presses the ball and walks it. This
    sensor is what says the walk is *controlled* and not a kick.
    """
    found = env.scene[sensor_name].data.found
    return found.abs().sum(dim=tuple(range(1, found.dim()))) > 0


# ── observations ─────────────────────────────────────────────────────────────


def ball_rel_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Ball position in the robot's frame-free world offset — [B, 3].

    Deliberately the *offset from the robot's base*, not the raw world pos:
    the policy's other inputs are egocentric, and a world coordinate would
    make the same skill look different on env 0 and env 1024.
    """
    return _ball_pos(env) - _robot_pos(env)


def hoop_rel_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    return _hoop_pos(env) - _robot_pos(env)


class BallProgress(ManagerTermBase):
    """v4: shape the *motion toward* the ball, not the distance to it.

    Per control step: clip((prev_d - d) / v_full, 0, 1). Walking toward the
    ball at v_full m/s pays 1.0 per step; standing pays **0** -- which is the
    point. The v3 GPU run (2026-11-06, 1000 iterations) showed why distance
    shaping alone fails: standing collects ~13.7 per step of inherited
    locomotion shaping and the spawn-to-ball gap is only ~1.2, so once
    `Policy/mean_std` collapsed (0.11) no policy could discover the walk.
    Progress pays for the walk itself, every step of it, from wherever it
    starts.

    State lives per environment like the crown's pump counter: `reset`
    re-arms on episode end so a reset-prop's teleport (the placement event
    moves the ball back to its spawn) is not read as a huge free progress
    gain -- the step after any reset returns 0.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        params = getattr(cfg, "params", {}) or {}
        self.v_full = float(params.get("v_full", 0.24))
        self._prev = None

    def reset(self, env_ids):
        if self._prev is None:
            return
        if env_ids is None or isinstance(env_ids, slice):
            self._prev[:] = float("nan")
        else:
            self._prev[env_ids] = float("nan")

    def __call__(self, env: ManagerBasedRlEnv, **_params) -> torch.Tensor:
        b, r = _ball_pos(env), _robot_pos(env)
        d = torch.linalg.norm(b[:, :2] - r[:, :2], dim=1)
        if self._prev is None:
            self._prev = d.clone()
            return torch.zeros_like(d)
        gain = ((self._prev - d) / self.v_full).clamp(0.0, 1.0)
        gain = torch.where(self._prev.isnan(), torch.zeros_like(gain), gain)
        self._prev = d
        return gain


# ── rewards ──────────────────────────────────────────────────────────────────


def approach_ball(env: ManagerBasedRlEnv, sigma: float = 0.35,
                  leave_r: float = 0.30) -> torch.Tensor:
    """ exp(−d²/σ²) on the horizontal claw→ball distance.

    v9 kill switch: stops paying once the ball is *near the hoop* (closer
    than `leave_r`), handing the distance budget over to `carry_to_hoop`
    so a parked ball at the box cannot feed stage 1 forever. (v6-v8's
    `~lifted` switch never fired — the pinch lift never happened; see
    `_claw_touch` for why the task stopped waiting for it.)
    """
    d_xy = torch.linalg.norm(_ball_pos(env)[:, :2] - _robot_pos(env)[:, :2], dim=1)
    d_hoop = torch.linalg.norm(_ball_pos(env)[:, :2] - _hoop_pos(env)[:, :2], dim=1)
    return torch.exp(-(d_xy / sigma) ** 2) * (d_hoop > leave_r).float()


class BallToHoopProgress(ManagerTermBase):
    """v10.5: per-step ball→hoop closing, gated on the claw touching the ball.

    The answer to the operator's question of 2026-11-08: *"the claw taps the
    ball, it rolls away, and there is no camera — how does the robot keep
    pushing?"* A cone on distance (v9's `carry_to_hoop`) does get the robot
    back to a lost ball — `ball_rel_pos`/`hoop_rel_pos` are fed to the policy
    as observations directly, no vision needed — but a cone cannot tell a
    **controlled push** from a kick that happens to land well. This term pays
    the ball's own velocity component toward the hoop, one step at a time,
    and only while a jaw is on it: kick it and the reward goes silent, walk
    it and every 12 mm of the 1.2 m trip pays a full step. That is the
    dense gradient the bulldozer skill needs, and it is the stage-2 term now
    — `carry_to_hoop`'s cone stays as the coarser "arrive with it" shaping.

    Same per-env state discipline as `BallProgress`: the step after any reset
    returns 0 so the placement teleport cannot be farmed.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        params = getattr(cfg, "params", {}) or {}
        self.v_full = float(params.get("v_full", 0.12))
        self._prev = None

    def reset(self, env_ids):
        if self._prev is None:
            return
        if env_ids is None or isinstance(env_ids, slice):
            self._prev[:] = float("nan")
        else:
            self._prev[env_ids] = float("nan")

    def __call__(self, env: ManagerBasedRlEnv, sensor_name: str =
                 "claw_ball_contact", **_params) -> torch.Tensor:
        b, h = _ball_pos(env), _hoop_pos(env)
        d = torch.linalg.norm(b[:, :2] - h[:, :2], dim=1)
        if self._prev is None:
            self._prev = d.clone()
            return torch.zeros_like(d)
        gain = ((self._prev - d) / self.v_full).clamp(0.0, 1.0)
        gain = torch.where(self._prev.isnan(), torch.zeros_like(gain), gain)
        self._prev = d
        return gain * _claw_touch(env, sensor_name).float()


def ball_lifted(env: ManagerBasedRlEnv, cap: float = 0.12) -> torch.Tensor:
    """Ball height above rest, capped: reward *how high the object came*, a
    proxy for 'the grip held'. Beyond `cap` it is a fling, not a carry."""
    lift = _ball_lift(env).clamp(min=0.0)
    return (lift / cap).clamp(max=1.0)


def carry_to_hoop(env: ManagerBasedRlEnv, sigma: float = 0.35,
                  gate: str = "contact",
                  sensor_name: str = "claw_ball_contact") -> torch.Tensor:
    """Same shaping as approach, but ball→hoop.

    gate="contact" (v9, what env_cfg wires): the ball counts as carried only
    while a jaw is **touching it** — the claw bulldozing the ball across the
    floor is this machine's actual carry, and a ball that was kicked and
    rolls free must not unlock the stage. gate="grip" is the v7 behaviour
    (off-floor + shut aperture) and gate="lift" the v6 one, both kept only
    so old checkpoints replay with the terms they were trained on.
    """
    d = torch.linalg.norm(_ball_pos(env) - _hoop_pos(env), dim=1)
    if gate == "contact":
        held = _claw_touch(env, sensor_name)
    else:
        held = _ball_lift(env) > 0.02
        if gate == "grip":
            held = held & (_aperture(env) > _SHUT_APERTURE)
    return torch.exp(-(d / sigma) ** 2) * held.float()


def ball_in_hoop(env: ManagerBasedRlEnv, rim_h: float = 0.06) -> torch.Tensor:
    """The score. 1 while the ball centre is inside the rim footprint and
    below the rim line — holding it over the hoop is `carry_to_hoop`'s job;
    this only fires on a release into the tray. Settle in the tray keeps the
    reward alive until termination, so per-step weights stay small."""
    b, h = _ball_pos(env), _hoop_pos(env)
    inside = (torch.abs(b[:, 0] - h[:, 0]) < HOOP_INNER) \
        & (torch.abs(b[:, 1] - h[:, 1]) < HOOP_INNER)
    under_rim = _ball_lift(env) < rim_h
    return (inside & under_rim).float()
