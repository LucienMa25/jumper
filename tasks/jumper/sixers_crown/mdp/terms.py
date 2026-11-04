"""Crown terms: heading away, claws above the head, three counted pumps.

Nothing here adds observations — the policy's state already carries every
joint, and the palm heights below are derived, not sensed. That keeps the
export path clean (sixers_dunk learned that lesson the hard way, in the
other direction).

Heads are geometric: a palm is *raised* when its world z clears the trunk's
by `margin`; a pump is a fall below the lowered line followed by a rise back
over the raised line, counted per environment. The counter lives in the term
instance (a class term: mjlab instantiates `func(cfg=…, env=…)` once and
calls it as `func(env, **params)` every step), and `reset` zeroes it on
episode end, so every attempt is three fresh pumps.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.manager_base import ManagerTermBase

PALM_BODIES = ("LF_palm_link", "RF_palm_link")
TRUNK = "base_link"


def _body_ids(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Indices of [LF_palm, RF_palm, trunk] on the body-pose axis, cached.

    `Entity.find_bodies` is the entity's own name→index resolution and
    `preserve_order=True` pins the column order to the lookup list, so what
    reads below is always (left palm, right palm, trunk).
    """
    ids = getattr(env, "_crown_body_ids", None)
    if ids is None:
        robot = env.scene["robot"]
        found, names = robot.find_bodies([*PALM_BODIES, TRUNK], preserve_order=True)
        assert names == [*PALM_BODIES, TRUNK], f"crown body lookup drifted: {names}"
        ids = torch.tensor(found, device=robot.data.device, dtype=torch.long)
        env._crown_body_ids = ids
    return ids


def _palm_lift(env: ManagerBasedRlEnv) -> torch.Tensor:
    """min(z_LF_palm, z_RF_palm) - z_trunk: how far the *lower* claw sits
    above the body. min is the point: one claw up is not a crown."""
    pose = env.scene["robot"].data.body_link_pose_w[:, _body_ids(env)]
    return torch.minimum(pose[:, 0, 2], pose[:, 1, 2]) - pose[:, 2, 2]


def _yaw_rad(env: ManagerBasedRlEnv) -> torch.Tensor:
    q = env.scene["robot"].data.root_link_quat_w      # (w, x, y, z)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def turn_away(env: ManagerBasedRlEnv, target_yaw: float = 3.14159265) -> torch.Tensor:
    """1 facing `target_yaw`; the spawn yaw is pinned at 0, so away = pi."""
    err = torch.remainder(_yaw_rad(env) - target_yaw + torch.pi, 2.0 * torch.pi) - torch.pi
    return 0.5 * (1.0 + torch.cos(err))


def claws_raised(env: ManagerBasedRlEnv, margin: float = 0.05) -> torch.Tensor:
    """Ramp of `palm_lift - margin` over margin, clamped 0..1.

    The raised line `margin` is the same number the pump term uses as `hi`:
    one shaping of the choreography, stated once per consumer.
    """
    lift = _palm_lift(env)
    return ((lift - margin) / margin).clamp(0.0, 1.0)


class CrownPumps(ManagerTermBase):
    """Count pumps: a fall below the lowered line followed by a rise over the
    raised line. 1.0 per pump, and nothing after the third.

    Hysteresis (up at `hi`, down at `lo`) keeps the crossings honest against
    jitter. The ceiling is the choreography: an unbounded oscillator is the
    cheaper thing to farm, so pay stops at `target` — that is what makes the
    robot *hold still* after three.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        params = getattr(cfg, "params", {}) or {}
        self.hi = float(params.get("hi", 0.05))
        self.lo = float(params.get("lo", 0.02))
        self.target = int(params.get("target", 3))
        n, dev = env.num_envs, env.device
        self._count = torch.zeros(n, device=dev, dtype=torch.long)
        self._was_up = torch.zeros(n, device=dev, dtype=torch.bool)

    def reset(self, env_ids):
        if isinstance(env_ids, slice) or env_ids is None:
            env_ids = torch.arange(self._count.shape[0], device=self._count.device)[
                env_ids if env_ids is not None else slice(None)]
        self._count[env_ids] = 0
        self._was_up[env_ids] = False

    def __call__(self, env: ManagerBasedRlEnv, **_params) -> torch.Tensor:
        lift = _palm_lift(env)
        fell = self._was_up & (lift < self.lo)     # a press-down completed
        self._was_up = lift > self.hi
        self._count += fell.long()
        return (fell & (self._count <= self.target)).float()
