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
  3. four staged rewards: approach 3.0 / lift 4.0 / carry 3.0 / score 15.0.
     The first budget (1.0/1.5/1.0/4.0) measured flatline — standing collects
     ~13.7 per step of inherited locomotion shaping and outshouts a 0.06-per-
     step approach term; the weights are now above that comfort zone;
  4. a fixed spawn facing down-court. five_foot pins its spawn only under
     `--objects`; here the pin applies whenever the world is present,
     training included — an approach task whose ball starts behind the robot
     half the time is measuring randomisation, not skill. Commands go neutral
     in **training only**: replay keeps five_foot's own operator behaviour so
     the fixed-command evaluation path still resolves `twist`.

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
from .mdp.terms import (BallProgress, approach_ball, ball_in_hoop, ball_lifted,
                        carry_to_hoop, hoop_rel_pos, ball_rel_pos)
from .props import HOOP_RIM_H, dunk_entities


def env_cfg(
    asset: Path | None = None,
    play: bool = False,
    hoop: bool = False,
):
    cfg = _five_foot_env_cfg(asset=asset, play=play)   # objects=False
    if hoop:
        _apply_hoop_world(cfg, play=play)
    return cfg


def _apply_hoop_world(cfg, *, play: bool) -> None:
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

    # ── drop the payload curriculum (v3, measured on the v2 run, 2026-11-05) ──
    # five_foot's payload ladder promotes when "the policy tracks the top
    # command rung" -- and this task pins the command at zero, so standing
    # perfectly *is* the top rung: v2 promoted at iteration ~921 and loaded
    # 0.6 kg onto the carried arm for no reason the task can use. The load
    # also makes reaching and gripping the ball physically harder while it
    # pays nothing toward it. The startup event (empty claw, rung 0) stays --
    # `requires_model_fields` is what makes body_mass writable per world --
    # only the ladder goes.
    if "payload" in cfg.curriculum:
        del cfg.curriculum["payload"]

    # ── the reset events that actually place the props ──
    # Adding the entity is not enough: mjlab applies `env_origins` inside
    # `reset_root_state_uniform`, and an entity with no reset event stays at
    # (0, 0, 0) in **every** environment (five_foot's `objects.py::_place_props`
    # states this outright). Measured on `model_299`, 2026-11-04: with no events
    # the ball spawned buried at the world origin and the hoop was not per-env
    # replicated, so `ball_in_hoop`'s distance test became a constant and the
    # run trained on a scoring bug. The pose range is **empty** — the entity's
    # own `init_state` supplies the offset, and passing it again would land the
    # prop at twice the distance.
    from mjlab.envs.mdp import events as mdp_events
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    for name in ("dunk_ball", "dunk_hoop"):
        cfg.events[f"reset_prop_{name}"] = EventTermCfg(
            func=mdp_events.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {},
                    "asset_cfg": SceneEntityCfg(name)},
        )


    # ── observations: both groups see the world they must act on ──
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        terms["ball_rel_pos"] = ObservationTermCfg(func=ball_rel_pos)
        terms["hoop_rel_pos"] = ObservationTermCfg(func=hoop_rel_pos)

    # ── staged rewards ──
    # The weights are this task's decision, and the first run measured why:
    # standing still collects ~13.7 per step of the inherited locomotion budget
    # while the ball at 0.585 m paid 1.0 x 0.06 = 0.06 per step -- the dunk
    # stages were never loud enough to move the policy (`model_1999`,
    # 2026-11-04: approach/carry/score all exactly 0.0000, action std collapsed
    # to 0.06). They are raised until the goal-directed gradient beats that
    # comfort zone; the sigma widening keeps early progress visible. If the
    # second run shows the same flatline, the budget is still wrong, not the
    # robot.
    cfg.rewards["approach_ball"] = RewardTermCfg(
        func=approach_ball, weight=3.0, params={"sigma": 0.60})
    # v4: the distance budget above plateaued at the standing baseline on GPU
    # (v3, 2026-11-06: 1.91 peak at ~500, back to 1.76 by 1000 as std
    # collapsed to 0.11). Progress pays for the walk itself -- 6.0 per step at
    # v_full against standing's 0.0 -- so the 1.2 comfort gap stops mattering.
    cfg.rewards["ball_progress"] = RewardTermCfg(
        func=BallProgress, weight=6.0, params={"v_full": 0.24})
    cfg.rewards["ball_lifted"] = RewardTermCfg(
        func=ball_lifted, weight=4.0, params={"cap": 0.12})
    cfg.rewards["carry_to_hoop"] = RewardTermCfg(
        func=carry_to_hoop, weight=3.0, params={"sigma": 0.50})
    cfg.rewards["ball_in_hoop"] = RewardTermCfg(
        func=ball_in_hoop, weight=15.0, params={"rim_h": HOOP_RIM_H + 0.04})

    # ── v5: zero the standing-comfort terms ────────────────────────────────
    # v4's curve (2026-11-07, 4090D, 4000 iters) ended exactly at the
    # standing baseline: track_linear 3.9 + track_angular 4.8 + feet_planted
    # 5.0 ≈ 13.7 per step paid to a robot that commands nothing and goes
    # nowhere, while the whole dunk budget paid ≤9. With commands pinned to
    # zero those tracking terms *are* a standing bonus — they no longer
    # describe the task's goal, they oppose it. Zeroed here, loudly; the
    # five_foot base keeps them at full weight for its own runs.
    # The safety floor stays: upright/pose/dof limits/self-collision still
    # penalise falling and flailing; `fell_over` still terminates.
    for comfort in ("track_linear_velocity", "track_angular_velocity",
                    "feet_planted", "five_foot_gait"):
        if comfort in cfg.rewards:
            cfg.rewards[comfort].weight = 0.0


    # ── fixed spawn ──
    # Pinned in train and replay alike: the ball and hoop sit at known env-
    # relative positions, so a random spawn would sometimes start them behind
    # the robot. Pin by narrowing the task's own `reset_base` (five_foot's
    # pattern) and keep every other key of `pose_range` -- `z` is a settle,
    # not a position.
    reset = cfg.events.get("reset_base")
    if reset is None:
        raise ValueError("--hoop pins the spawn, and this config has no 'reset_base'")
    pose = dict(reset.params.get("pose_range", {}))
    pose["x"] = (0.0, 0.0)
    pose["y"] = (0.0, 0.0)
    pose["yaw"] = (0.0, 0.0)
    reset.params["pose_range"] = pose

    # ── neutral commands: TRAINING ONLY ──
    # With the sampled command pinned at rest, the base tracking reward is
    # already maximised by standing, and `approach_ball` becomes the only
    # gradient that moves the robot. The policy learns *goal-directed* walking
    # with no operator; on hardware the same checkpoint runs under an operator's
    # zero command.
    #
    # Replay keeps five_foot's own behaviour (the operator drives; hands off,
    # the sampled command passes through) -- pinning replay too would leave the
    # `twist` command term with no operator bindings to drive and break the
    # fixed-command evaluation path. (`model_1999`'s notebook run hit exactly
    # that: a neutralised replay refused `--command` evaluation.)
    if not play:
        stilled = []
        for name, command in cfg.commands.items():
            for field_name in ("rel_standing_envs", "rel_neutral_envs"):
                if hasattr(command, field_name):
                    setattr(command, field_name, 1.0)
                    stilled.append(f"{name}.{field_name}")
        if not stilled:
            raise ValueError("--hoop wants neutral commands; no command term carries them")


