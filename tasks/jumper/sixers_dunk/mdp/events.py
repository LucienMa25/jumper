"""Per-step world-protection events for the dunk task.

v12.6 exists because of how v12.5 died (Colab, iteration 1701):
`check_nan` caught NaN in the `actor` observation group and stopped the run.
The observations are physics state, so the NaN was the simulation's own —
the classic generator in this world is the 30 g ball being shoved hard into
the cup's wall seam: at warp's 50 Hz substeps a fast small sphere can
*tunnel* into a hull, the solver answers with an astronomical velocity, and
one poisoned environment drags the whole training batch into NaN.

`guard_ball` is the two-part answer, wired as a `mode="step"` event (runs
every policy step, after the physics that could blow up):

  * **cap** — any ball moving faster than `max_speed` is written back at
    `max_speed`. A bulldozed ball never legitimately exceeds ~1 m/s (the
    robot itself walks at 0.25), so the cap only ever bites on the
    tunnelling precursor; and
  * **repair** — a ball whose state has already gone non-finite is put back
    to its spawn pose with zero velocity in that one environment. The
    episode continues instead of the guard raising `ValueError` and killing
    the run: losing one env's steps is what the reset distribution is for.
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ..props import BALLS_XY, BALL_NAMES, BALL_R


def guard_ball(env: ManagerBasedRlEnv, env_ids=None,
               asset_cfg: SceneEntityCfg = SceneEntityCfg("dunk_ball_0"),
               max_speed: float = 2.5) -> None:
    """Cap the ball's speed; soft-reset it in the envs where physics NaNed.

    `env_ids` is the manager's per-fire subset and is deliberately ignored —
    the guard watches every env on every step. Silent in a healthy step:
    the velocity write happens only where the speed exceeds `max_speed` or
    the state is non-finite, so finite normal play never touches the solver
    through this door.
    """
    entity: Entity = env.scene[asset_cfg.name]
    i = BALL_NAMES.index(asset_cfg.name)
    rest_xy = torch.tensor(BALLS_XY[i], device=env.device)

    pos = entity.data.root_link_pos_w
    vel = entity.data.root_link_lin_vel_w
    finite = torch.isfinite(pos).all(dim=1) & torch.isfinite(vel).all(dim=1)

    # ── repair: this ball's state is poison — put it back on its spot ──
    broken = ~finite
    if bool(broken.any()):
        ids = broken.nonzero(as_tuple=False).flatten()
        origin = env.scene.env_origins[ids]
        pos_new = origin[:, :2] + rest_xy
        root_state = torch.zeros(ids.shape[0], 13, device=env.device)
        root_state[:, 0] = pos_new[:, 0]
        root_state[:, 1] = pos_new[:, 1]
        root_state[:, 2] = origin[:, 2] + BALL_R + 0.002   # its spawn height
        root_state[:, 3] = 1.0                            # quat (w,x,y,z)
        entity.write_root_state_to_sim(root_state, ids)

    # ── cap: tunneling precursor — a finite but absurd speed ──
    speed = torch.linalg.norm(vel, dim=1)
    fast = finite & (speed > max_speed)
    if bool(fast.any()):
        ids = fast.nonzero(as_tuple=False).flatten()
        scale = (max_speed / speed[ids].clamp(min=1e-6)).unsqueeze(1)
        spin = entity.data.root_link_ang_vel_w[ids]
        entity.write_root_link_velocity_to_sim(
            torch.cat([vel[ids] * scale, spin], dim=1), ids)
