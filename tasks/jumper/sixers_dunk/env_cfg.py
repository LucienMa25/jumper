"""Environment config for jumper.sixers_dunk.

**Built by layering, not by copying.** `five_foot.env_cfg` is the base: the
five-grounded legs, the stowed arm, the claw action/observation sets, the
payload curriculum, the walk-timeout termination — all of it stays exactly
where five_foot maintains it, and this file adds only the dunk world. If
five_foot fixes a gait reward tomorrow, this task inherits the fix rather
than a stale copy of it.

`--hoop` layers on:

  1. the two props (`props.dunk_entities`) and the contact capacity/friction
     model they need — the same `PROP_*` physics five_foot's row trains and
     replays under, imported not restated;
  2. two observation terms on both groups (`mdp/terms.py`), critic and actor:
     ball and hoop offset from the robot;
  3. four staged rewards: approach 1.0 / lift 1.5 / carry 1.0 / score 4.0.
     The budget is deliberately shallow early — three of the four are shaping
     a skill five_foot has never been asked for, and a dominant term would
     teach a lottery-ticket fling before a walk;
  4. a fixed spawn facing down-court. five_foot pins its spawn only under
     `--objects`; here the pin applies whenever the world is present,
     training included — an approach task whose ball starts behind the robot
     half the time is measuring randomisation, not skill.

Unlike five_foot's `--objects` (replay-only, "a policy replayed here walks on
friction it was not trained with"), **`--hoop` is meant for training and
replay alike**, and both entry points must pass it: it is the first dunk task
whose world is the training world.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ..five_foot.env_cfg import env_cfg as _five_foot_env_cfg
from .mdp.terms import approach_ball, ball_in_hoop, ball_lifted, carry_to_hoop, \
    hoop_rel_pos, ball_rel_pos
from .props import HOOP_RIM_H, dunk_entities


def env_cfg(
    asset: Path | None = None,
    play: bool = False,
    hoop: bool = False,
):
    cfg = _five_foot_env_cfg(asset=asset, play=play)   # objects=False
    if hoop:
        _apply_hoop_world(cfg)
    return cfg


def _apply_hoop_world(cfg) -> None:
    from .props import HOOP_X, HOOP_Y, BALL_X, BALL_Y

    # ── physics capacity and the prop friction model (five_foot's, imported) ──
    from ..five_foot.objects import (
        OBJECTS_NCONMAX, OBJECTS_NJMAX, PROP_CONE, PROP_IMPRATIO,
    )
    cfg.sim.nconmax = max(int(cfg.sim.nconmax or 0), OBJECTS_NCONMAX)
    cfg.sim.njmax = max(int(cfg.sim.njmax or 0), OBJECTS_NJMAX)
    cfg.sim.mujoco.cone = PROP_CONE
    cfg.sim.mujoco.impratio = PROP_IMPRATIO

    # ── entities ──
    entities = cfg.scene.entities
    clash = sorted({"dunk_ball", "dunk_hoop"} & set(entities))
    if clash:
        raise ValueError(f"--hoop would overwrite existing entities {clash}")
    entities.update(dunk_entities())

    # ── observations: both groups see the world they must act on ──
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        terms["ball_rel_pos"] = ObservationTermCfg(func=ball_rel_pos)
        terms["hoop_rel_pos"] = ObservationTermCfg(func=hoop_rel_pos)

    # ── staged rewards (weights are this task's decision, written down) ──
    cfg.rewards["approach_ball"] = RewardTermCfg(
        func=approach_ball, weight=1.0, params={"sigma": 0.35})
    cfg.rewards["ball_lifted"] = RewardTermCfg(
        func=ball_lifted, weight=1.5, params={"cap": 0.12})
    cfg.rewards["carry_to_hoop"] = RewardTermCfg(
        func=carry_to_hoop, weight=1.0, params={"sigma": 0.35})
    cfg.rewards["ball_in_hoop"] = RewardTermCfg(
        func=ball_in_hoop, weight=4.0, params={"rim_h": HOOP_RIM_H + 0.04})

    # ── fixed spawn and neutral commands ──
    # Pin by narrowing the task's own `reset_base` (five_foot's pattern):
    # keep every other key of `pose_range` — `z` is a settle, not a position —
    # and put x/y/yaw at a point so the ball is always ahead-left and the
    # hoop ahead-right.
    #
    # Commands go **neutral in training too**, which is the point of the
    # design: with the sampled command pinned at rest, the base tracking
    # reward is already maximised by standing, and `approach_ball` becomes
    # the only gradient that moves the robot. The policy learns *goal-directed*
    # walking with no operator; on hardware the same checkpoint runs under an
    # operator's zero command. A goal-command interface (joystick-to-ball
    # vector, or an autonomous planner on top of `jumper.posture`) is the
    # stage-2 upgrade; do not let sampled twist commands coexist with the
    # approach reward, or the robot is pulled by two masters and both
    # numbers on the dashboard look fine while the policy learns nothing.
    reset = cfg.events.get("reset_base")
    if reset is None:
        raise ValueError("--hoop pins the spawn, and this config has no 'reset_base'")
    pose = dict(reset.params.get("pose_range", {}))
    pose["x"] = (0.0, 0.0)
    pose["y"] = (0.0, 0.0)
    pose["yaw"] = (0.0, 0.0)
    reset.params["pose_range"] = pose

    stilled = []
    for name, command in cfg.commands.items():
        for field_name in ("rel_standing_envs", "rel_neutral_envs"):
            if hasattr(command, field_name):
                setattr(command, field_name, 1.0)
                stilled.append(f"{name}.{field_name}")
    if not stilled:
        raise ValueError("--hoop wants neutral commands; no command term carries them")

