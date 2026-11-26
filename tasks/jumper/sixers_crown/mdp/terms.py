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
    above the body. min is the point: one claw up is not a crown.

    v2 NOTE: kept for the module history and the video-check scripts, but the
    crown rewards no longer read it — the FK scan (2026-11-26, model sweep
    over the full J1 range at the deployment HOME pose) showed the palm tops
    out **6 mm above the trunk**: `margin` 0.05 was geometrically impossible,
    which is why 2000 iterations produced zero claws and zero pumps. The
    reachable signature is the shoulder-pitch joint itself; see `_j1_lift`.
    """
    pose = env.scene["robot"].data.body_link_pose_w[:, _body_ids(env)]
    return torch.minimum(pose[:, 0, 2], pose[:, 1, 2]) - pose[:, 2, 2]


#: v2 (FK-measured): the raised-claw shoulder-pitch target, per side. Both
#: J1 axes are (0,1,0) with the same range, and the palm-z sweep is the same
#: curve mirrored in sign per side, so ONE value raises both arms: J1 ~ -3.2
#: sits the palms at their global max (+0.006 above trunk, ~17 cm above home;
#: `assets/jumper/jumper_sixers.xml`, 0.4 deg sweep grid). The stow value is
#: ~-1.8, so the action target `(-3.2 - default)/scale` ≈ -0.7 lies well
#: inside [-1, 1] at the family's ~0.27 actuator scale — two orders of
#: magnitude more reachable than the z-line it replaces.
J1_RAISE = -3.20
J1_RAISE_LO = -3.05      # pump hysteresis: press below this counts as a down-stroke
# The sim's stow (measured 2026-11-26: both J1 default to -1.066 in
# `jumper_sixers.xml`; the hardware deployment HOME sits deeper at -1.84,
# which is hello's *raised* row in this action space, not its rest). The
# signal therefore starts at 0 at the sim stow and reaches 1.73 at the FK
# maximum — env_cfg's lines (1.3/1.7) sit on this scale, and the action
# scale raised to 1.5 is what makes the upper end COMMANDABLE at all
# (0.25 could only reach -1.316; see the v2b note in env_cfg).
J1_HOME_APPROX = -1.066


def _j1_positions(env: ManagerBasedRlEnv) -> torch.Tensor:
    """[B, 2] the two shoulder-pitch joint positions (LF_J1, RF_J1), cached
    column indices like `_body_ids` above."""
    ids = getattr(env, "_crown_j1_ids", None)
    robot = env.scene["robot"]
    if ids is None:
        found, names = robot.find_joints(["LF_J1_joint", "RF_J1_joint"],
                                         preserve_order=True)
        assert names == ["LF_J1_joint", "RF_J1_joint"], f"crown joint lookup drifted: {names}"
        ids = torch.tensor(found, device=robot.data.device, dtype=torch.long)
        env._crown_j1_ids = ids
    return robot.data.joint_pos[:, ids]


def _arm_signal(env: ManagerBasedRlEnv) -> torch.Tensor:
    """The crown's ONE observable, in joint space: mean shoulder-pitch
    advance past a mid-stow line, `(-J1 + J1_HOME_APPROX)` centred so the
    stow sits at ~0 and the raised pose at ~1.4.

    Mean (not min, as `_palm_lift` used): each claw pumps with its own joint,
    a half-raised pair is still visibly a celebration, and the mirrored
    symmetry augmentation (on for this task) reflects LF<->RF without
    touching a mean. The sign flip is because raising = J1 *decreasing*.
    """
    j1 = _j1_positions(env)
    return (-j1 + J1_HOME_APPROX).mean(dim=1)


def _yaw_rad(env: ManagerBasedRlEnv) -> torch.Tensor:
    q = env.scene["robot"].data.root_link_quat_w      # (w, x, y, z)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def turn_away(env: ManagerBasedRlEnv, target_yaw: float = 3.14159265) -> torch.Tensor:
    """1 facing `target_yaw`; the spawn yaw is pinned at 0, so away = pi."""
    err = torch.remainder(_yaw_rad(env) - target_yaw + torch.pi, 2.0 * torch.pi) - torch.pi
    return 0.5 * (1.0 + torch.cos(err))


def claws_raised(env: ManagerBasedRlEnv, lo: float = 0.90,
                 hi: float = 1.20) -> torch.Tensor:
    """v2: ramp of the joint-space arm signal (`_arm_signal`) from `lo` to
    `hi` — 0 with the claws at mid-stow, 1 once both shoulder pitches hold
    at/through the FK-measured raised value (-3.2 ± 0.15). The v1 palm-z
    cone it replaces could never fire (palm max +0.006 m vs the 0.05 line),
    which is the recorded reason the first crown run scored zero for
    2000 iterations. The name of the parameter pair mirrors `CrownPumps`
    so one raised line is stated once per consumer."""
    return ((_arm_signal(env) - lo) / (hi - lo)).clamp(0.0, 1.0)


class CrownPumps(ManagerTermBase):
    """Count pumps on the joint-space arm signal: a fall below the lowered
    line followed by a rise back over the raised line. 1.0 per pump, and
    nothing after the third.

    Hysteresis (up at `hi`, down at `lo`) keeps the crossings honest against
    jitter — in joint space the lines are 0.3 rad apart, an order of
    magnitude above stance noise. The ceiling is the choreography: an
    unbounded oscillator is the cheaper thing to farm, so pay stops at
    `target` — that is what makes the robot *hold still* after three.

    v2: signal switched from palm-z (unreachable, see `claws_raised`) to
    `_arm_signal`; `hi`/`lo` are now signal units (J1 -3.0 / -2.7).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        params = getattr(cfg, "params", {}) or {}
        self.hi = float(params.get("hi", 1.20))
        self.lo = float(params.get("lo", 0.90))
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
        lift = _arm_signal(env)
        fell = self._was_up & (lift < self.lo)     # a press-down completed
        self._was_up = lift > self.hi
        self._count += fell.long()
        return (fell & (self._count <= self.target)).float()
