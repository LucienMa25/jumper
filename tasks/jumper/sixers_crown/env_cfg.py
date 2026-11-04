"""Environment config for jumper.sixers_crown.

**Built by layering, not by copying.** `flat.env_cfg` is the base — six feet
grounded, the whole 20-joint action (both front arms included: they are joints
like any other), the stance and limits budget. This file adds the
celebration: turn away, both claws up, three counted pumps.

The choreography has no world and no operator:

  * commands are **neutral in training only** (the dunk's lesson, written down
    where it can be found): a policy that must turn and raise while a sampled
    twist command yanks at it has two masters; replay keeps the operator's
    path so a person can walk a crowned robot around afterwards;
  * the **spawn yaw is pinned at 0 in both modes**, because `turn_away` names
    a world direction (pi from the spawn's +x). Position stays randomised —
    where on the floor the bow happens is nobody's decision.

The weights are this task's decision:

  turn_away 2.0    — a posture task's heading is cheap to hold once turned;
  claws_raised 4.0 — the shape a video is judged on;
  crown_pumps 6.0  — the only term with a ceiling, so the loudest.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.managers.reward_manager import RewardTermCfg

from ..flat.env_cfg import env_cfg as _flat_env_cfg
from .mdp.terms import CrownPumps, claws_raised, turn_away


def env_cfg(asset: Path | None = None, play: bool = False):
    cfg = _flat_env_cfg(asset=asset, play=play)

    # ── the celebration terms, on top of the standing budget ──
    cfg.rewards["turn_away"] = RewardTermCfg(
        func=turn_away, weight=2.0, params={"target_yaw": 3.14159265})
    cfg.rewards["claws_raised"] = RewardTermCfg(
        func=claws_raised, weight=4.0, params={"margin": 0.05})
    cfg.rewards["crown_pumps"] = RewardTermCfg(
        func=CrownPumps, weight=6.0,
        params={"hi": 0.05, "lo": 0.02, "target": 3})

    # ── spawn: yaw pinned (the heading target is a world direction) ──
    reset = cfg.events.get("reset_base")
    if reset is None:
        raise ValueError("crown pins the spawn yaw, and this config has no 'reset_base'")
    pose = dict(reset.params.get("pose_range", {}))
    pose["yaw"] = (0.0, 0.0)
    reset.params["pose_range"] = pose

    # ── neutral commands, training only ──
    if not play:
        stilled = []
        for name, command in cfg.commands.items():
            for field_name in ("rel_standing_envs", "rel_neutral_envs"):
                if hasattr(command, field_name):
                    setattr(command, field_name, 1.0)
                    stilled.append(f"{name}.{field_name}")
        if not stilled:
            raise ValueError("crown wants neutral commands; no command term carries them")
    return cfg
