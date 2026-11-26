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
    # v2 (2026-11-26): claws and pumps moved from palm-z to the shoulder
    # pitch. The FK sweep showed the palm cannot clear the trunk by the old
    # 0.05 m line (global max +0.006), so v1's two loud terms could not fire
    # and the first 2000-iteration run learned nothing about them. The lines
    # below are signal units, measured from `assets/jumper/jumper_sixers.xml`
    # (raised pose: J1 = -3.2 on both sides, palm +0.006 / ~17 cm above home).
    # Lines on the reachable scale: stow 0, full raise 1.6 (J1 ~ -2.67 —
    # the FK max 2.13 is the asymptote, but the arm's PD settles toward it
    # over 3-4 s, so the lines sit where a 60-step pump stroke provably
    # crosses; measured live 2026-11-26: action -1 hits sig 1.89 at 150
    # steps, 2.20 at 200). Pump press to sig 0.9 = J1 -1.97 — a visible
    # half-swing down, and hello's own wave amplitude (1.6 rad) lives
    # comfortably inside this hysteresis band.
    cfg.rewards["claws_raised"] = RewardTermCfg(
        func=claws_raised, weight=4.0, params={"lo": 1.10, "hi": 1.60})
    cfg.rewards["crown_pumps"] = RewardTermCfg(
        func=CrownPumps, weight=6.0,
        params={"hi": 1.60, "lo": 0.90, "target": 3})

    # ── v2b: THE ARM JOINS THE ACTION PROPERLY ────────────────────────────
    # The FK fix alone was not enough: every actuator here runs the family's
    # position scale 0.25, so a policy action of ±1 moves J1 only ±0.25 rad
    # around its -1.066 default — the raised signature (J1 ~ -2.8, palm above
    # trunk) sat 1.7 rad OUTSIDE the commanded range. No reward could have
    # bought what the action space could not reach; this is the deeper root
    # of v1's silence, and the reason a term-only v2 would have failed too.
    # The dunk's v7 finger is the precedent (scale 0.75 = the travel the task
    # needs, stated where it can be found). 2.2 spans the FK maximum exactly:
    # target = -1.066 + 2.2*(-1) = -3.266, one hair past the palm-z optimum
    # (-3.2) — measured, then set, not guessed.
    action = cfg.actions["joint_pos"]
    for joint in ("LF_J1_joint", "RF_J1_joint"):
        action.scale = {**action.scale, joint: 2.2}

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
