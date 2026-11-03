"""Staged dunk terms: approach → lift → carry → score, plus the observations.

Each stage is a term the policy can feel *before* the next one unlocks, the
same ladder the command curriculum uses for walking: nothing here pays for
the whole trick at once.

    stage 1  approach_ball   −dist(ball, claw) while the claw is empty
    stage 2  ball_lifted     ball height above its rest, gated on the grip
    stage 3  carry_to_hoop   −dist(ball, hoop) while the ball is lifted
    stage 4  ball_in_hoop    one-shot: ball centre inside the rim footprint
                             and under the rim top — the score.

Stage 2's "lifted" gate is z-only because **the claw cannot see its own grip**
at this scaffold stage: five_foot knows it holds something through
`mdp/grasp.py::ClawHold`, whose support-pair test is calibrated against the
box/can/hull props in `jaws.py`. A sphere breaks that calibration (point
contact, no flat support faces), so until stage 2 on the GPU re-calibrates
`ClawHold` for `dunk_ball`, `ball_lifted` reads the ball itself — the policy
is rewarded for the ball being up, whatever mechanism achieved it. That is
honest shaping, not a lie about sensing: the z test is the ground truth the
grip test was always a proxy for.

TODO(stage-2, GPU): extend `jaws.py` grip rule + `ClawHold` support pair to
the sphere; then gate stage 3 on ClawHold rather than on z, so the policy
cannot farm `carry_to_hoop` by flinging the ball.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv

from ..props import BALL_R, HOOP_INNER

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


def _robot_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_link_pose_w[:, :3]


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


# ── rewards ──────────────────────────────────────────────────────────────────


def approach_ball(env: ManagerBasedRlEnv, sigma: float = 0.35) -> torch.Tensor:
    """ exp(−d²/σ²) on the horizontal claw→ball distance, killed once the
    ball is off the ground (stage 1 stops paying when stage 2 has begun)."""
    d_xy = torch.linalg.norm(_ball_pos(env)[:, :2] - _robot_pos(env)[:, :2], dim=1)
    lifted = _ball_lift(env) > 0.02
    return torch.exp(-(d_xy / sigma) ** 2) * (~lifted).float()


def ball_lifted(env: ManagerBasedRlEnv, cap: float = 0.12) -> torch.Tensor:
    """Ball height above rest, capped: reward *how high the object came*, a
    proxy for 'the grip held'. Beyond `cap` it is a fling, not a carry."""
    lift = _ball_lift(env).clamp(min=0.0)
    return (lift / cap).clamp(max=1.0)


def carry_to_hoop(env: ManagerBasedRlEnv, sigma: float = 0.35) -> torch.Tensor:
    """Same shaping as approach, but ball→hoop, and it only lives while the
    ball is lifted — carrying without holding scores nothing."""
    d = torch.linalg.norm(_ball_pos(env) - _hoop_pos(env), dim=1)
    lifted = _ball_lift(env) > 0.02
    return torch.exp(-(d / sigma) ** 2) * lifted.float()


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
