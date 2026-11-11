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
from .mdp.terms import (BALL_SENSORS, BallProgress, BallToHoopProgress,
                        approach_ball, ball_in_hoop, ball_lifted, ball_score_count,
                        carry_to_hoop, hoop_rel_pos, ball_rel_pos)
from ..five_foot.claw import FINGER_JOINT, GRIPPER_CLOSED, GRIPPER_OPEN
from ..five_foot.env_cfg import FIVE_FOOT_JOINTS
from .props import BALL_NAMES, CUP_NAMES, dunk_entities


def env_cfg(
    asset: Path | None = None,
    play: bool = False,
    hoop: bool = False,
):
    # five_foot is the locomotion base -- the five grounded legs, the stowed
    # arm, the walk-timeout termination. Its *claw is re-opened to the policy*
    # just below (v7), and the flat skeleton's 20-joint gait set stays intact
    # except for the carried arm, exactly as five_foot grounds it.
    cfg = _five_foot_env_cfg(asset=asset, play=play)   # objects=False
    if hoop:
        _apply_hoop_world(cfg, play=play)
    return cfg


def _apply_hoop_world(cfg, *, play: bool) -> None:
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
    clash = sorted((set(BALL_NAMES) | set(CUP_NAMES)) & set(entities))
    if clash:
        raise ValueError(f"--hoop would overwrite existing entities {clash}")
    entities.update(dunk_entities())

    # ── v7: the finger joins the action ──────────────────────────────────
    # v6 saturated the approach cone (6.5/8 standing over the ball) while
    # `ball_lifted` sat at luck-level 0.011: the policy could not close the
    # claw. five_foot keeps LF_J4 out of the action because *its* policy was
    # never rewarded for what it does with it and its checkpoints predate the
    # idea; dunk reverses that judgement by its own contract -- lift, carry
    # and score all require a held ball, so the finger is the task's terminal
    # effector and the network must own it.
    # Mapping, calibrated on the probe (2026-11-07): the position action sweeps
    # `center + scale*action` with the center at the asset default GRIPPER_OPEN
    # (-0.65 measured: action 0 sits the finger open, +1 at scale 0.375 only
    # reached -0.27). scale = the full 0.75 travel closes exactly onto
    # GRIPPER_CLOSED at action +1 and opens toward the joint's own -1.6 at -1
    # (harmless: past -0.75 there is no wider mouth, `claw.py` measured that
    # wide-open pinches props out -- the policy learns not to camp there).
    # The action column order is the model's actuator order, LF_J4 first --
    # the MLP is permutation-agnostic, and the contract records the order from
    # the env, so export reads it as trained.
    action = cfg.actions["joint_pos"]
    action.actuator_names = list(action.actuator_names) + [FINGER_JOINT]
    action.scale = {**action.scale, FINGER_JOINT: GRIPPER_CLOSED - GRIPPER_OPEN}
    action.offset = {**(action.offset if isinstance(action.offset, dict) else {}),
                     FINGER_JOINT: 0.0}

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

    for name in (*BALL_NAMES, *CUP_NAMES):
        cfg.events[f"reset_prop_{name}"] = EventTermCfg(
            func=mdp_events.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {},
                    "asset_cfg": SceneEntityCfg(name)},
        )

    # ── v9/v12: one claw→ball contact sensor per ball ─────────────
    # Eight runs (v1-v8) settled it: this claw does not pinch a floor-level
    # ball. `five_foot/env_cfg.py` states the geometry outright — the stowed
    # claw hangs **40 mm off the floor** — so every contact is the under-jaw
    # shoving the ball along the ground. Rather than keep paying a `lift`
    # term the hardware cannot collect, the task rewards that shove as the
    # carry: stage 2 gates on body-on-ball contact, and a ball that was
    # kicked and rolls free earns nothing. The primary is the whole robot
    # subtree — see the v12.2 note below for why not just the jaws.
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    # v12.2 (4000-iteration log, 2026-11-09): the jaw-only gate never fired.
    # `ball_in_hoop` grew a nonzero tail while `hoop_progress` sat at 0.0002:
    # the robot does bulldoze the ball — with its **whole front body**, not
    # the 40 mm-high palm. The gate therefore watches any robot geom on any
    # ball (`subtree` from `base_link`, the dance task's pattern), and the
    # control discipline moves to where it can bite: `hoop_progress` only
    # pays while the contact is ON and the ball is CLOSING, so a kicked ball
    # rolling free still earns nothing.
    for ball, sensor in zip(BALL_NAMES, BALL_SENSORS):
        claw_ball = ContactSensorCfg(
            name=sensor,
            primary=ContactMatch(mode="subtree", pattern="base_link",
                                 entity="robot"),
            secondary=ContactMatch(mode="geom", pattern="dunk_ball",
                                   entity=ball),
            fields=("found",),
            reduce="none",
            num_slots=1,
            history_length=cfg.decimation,
        )
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (claw_ball,)


    # ── observations: both groups see the world they must act on ──
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        terms["ball_rel_pos"] = ObservationTermCfg(func=ball_rel_pos)
        terms["hoop_rel_pos"] = ObservationTermCfg(func=hoop_rel_pos)
        # v12: which of the three balls are already in, so the critic can
        # value a finished green (and the policy is never blind-sided by
        # the target switching when one sinks).
        terms["ball_score_count"] = ObservationTermCfg(func=ball_score_count)

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
        func=approach_ball, weight=8.0,
        params={"sigma": 0.35, "leave_r": 0.30})
    # v9 stop rule: the cone goes quiet when the ball is near the hoop, so
    # the push stage owns the end of the court; `ball_lifted` below is the
    # only stage that still cares about height.
    # v6: sigma 0.35/weight 8.0 make *staying at the ball* the profitable
    # posture (spawn sits inside the cone: exp(-(0.19/0.35)^2)*8 ≈ 6.0/step),
    # so the first thing the policy needs to learn is "don't leave", from
    # which progress and lift are one fling away. v5 proved an empty world of
    # 0.43 m is one the policy will never bridge on its own.
    # v4: the distance budget above plateaued at the standing baseline on GPU
    # (v3, 2026-11-06: 1.91 peak at ~500, back to 1.76 by 1000 as std
    # collapsed to 0.11). Progress pays for the walk itself -- 6.0 per step at
    # v_full against standing's 0.0 -- so the 1.2 comfort gap stops mattering.
    cfg.rewards["ball_progress"] = RewardTermCfg(
        func=BallProgress, weight=6.0, params={"v_full": 0.15})
    # v9: the lift stage is **retired** (weight 0). Its premise failed eight
    # times over: `five_foot/env_cfg.py` measures the stowed claw hanging
    # 40 mm off the floor, and no gait in this machine lowers the palm to a
    # 32 mm ball centre. What the hardware does have is a jaw face that
    # presses the ball against the ground and walks — so `carry_to_hoop`'s
    # new contact gate (below) *is* stage 2 now, and this slot stays wired
    # at zero only so old checkpoints keep their observation/reward layout.
    cfg.rewards["ball_lifted"] = RewardTermCfg(
        func=ball_lifted, weight=0.0, params={"cap": 0.12})
    cfg.rewards["carry_to_hoop"] = RewardTermCfg(
        func=carry_to_hoop, weight=6.0,
        params={"sigma": 0.50})   # v12: gates on touching the TARGET ball
    # v10.5: the dense half of stage 2. `carry_to_hoop`'s cone says "be near
    # the hoop with the ball"; this says "the ball is CLOSING on the hoop
    # right now, under jaw contact" — the per-step push a bulldozer actually
    # performs, and the term that answers the operator's "the ball rolls away
    # and there is no camera" objection (the policy reads ball_rel_pos /
    # hoop_rel_pos directly as observations; vision is not required).
    cfg.rewards["hoop_progress"] = RewardTermCfg(
        func=BallToHoopProgress, weight=6.0, params={"v_full": 0.12})
    cfg.rewards["ball_in_hoop"] = RewardTermCfg(
        func=ball_in_hoop, weight=15.0)

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
                    "feet_planted", "five_foot_gait", "feet_still"):
        if comfort in cfg.rewards:
            cfg.rewards[comfort].weight = 0.0
    # v12.4: `feet_still` joins the zeroed family. The test-drive of the
    # v12.3 checkpoints showed the deterministic policy walking *toward* the
    # ball and stopping dead at ~0.44 m, and the arithmetic says why: its
    # −2.0 charge fires on **every step that is not planted**, so walking
    # the fairway cost ~2.7/step against an approach cone that pays for
    # *being* near the ball, not moving. v5 zeroed the four tracking
    # comfort terms and missed this one — it is the same trap, priced per
    # footfall. The safety floor (tilt/height/limits/self-collision) is
    # untouched: a robot that cannot afford to walk is exactly the failure
    # those terms are for, and it will still show up there.
    # v10.5 (operator question of 2026-11-08: "why is the second pair of feet
    # always off the ground?"): zeroing `feet_planted` removed *all* pressure
    # on stance quality, and the policy found a comfortable three-legged lean
    # it keeps in replay. Bring it back at a **fraction** of five_foot's full
    # weight — enough to charge the weird stance, not enough to rebuild the
    # standing-comfort sinkhole (five_foot's full weight made standing worth
    # ~5/step, which is what starved the dunk stages in v1-v4).
    if "feet_planted" in cfg.rewards:
        cfg.rewards["feet_planted"].weight = 1.5


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

    # ── v12: a three-station round needs a longer clock ──
    # one sink is a 0.6–0.9 m push plus a 0.8–1.3 m run to the next firing
    # position; five_foot's 20 s episode ends barely past the first green.
    # 45 s fits a full round with room for one recovery; replay keeps
    # five_foot's own unpinned clock (and `play` already raised it to 1e9).
    if not play:
        cfg.episode_length_s = 45.0

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


