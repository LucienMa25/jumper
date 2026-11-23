"""Hyper-parameters for jumper.sixers_dunk.

The network and PPO scalars are the family's — five_foot's exact set, copied
deliberately rather than inherited, because agreement written down survives
one of them changing.

**Symmetry is off** for the same correctness reason five_foot states: the
robot walks on five legs with one clawed arm; its left-right mirror is a
machine that does not exist.

One scalar is not the family's and the reason is this task's:
`max_iterations=20_000`. The dunk is a longer-horizon skill than five_foot
walking — the staged rewards only compose after the policy first relearns
walking *around its own object manipulation*, and five_foot's own converged
checkpoint sits at 86,600. 20,000 is the first GPU run's budget, chosen for
"finish a stage ladder", not copied from anywhere; the ladder's stages will
say whether it was enough.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline

NUM_STEPS_PER_ENV = 24


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return jumper_ppo_baseline(
        experiment_name="jumper.sixers_dunk",
        # ── Network ──
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=1.0,
        # ── PPO ──
        # 0.005 was the family's; the v2 run collapsed action std to 0.09 by
        # iteration ~900 and the policy stopped exploring toward the ball
        # entirely (approach_ball flat at its spawn baseline), so v7 raised it
        # to 0.03 to keep exploration alive. Exploration has now *done its
        # job* (v12.9 @9299: ball_in_hoop pulses every few iterations) but the
        # mean policy cannot score on its own (deterministic eval: 1.6% vs the
        # noise-carried ~10% deliveries) and std sits stubbornly at 1.77 from
        # 7300 to 9299. v12.10 harvested at 0.005. Verdict (11/23): std fell
        # 1.77 -> 0.59 exactly as intended and training kept scoring through
        # the residual noise (pulses to 4.4 near iter 11290), but deterministic
        # in-cup went 1.6% -> 0.0% at BOTH 10500 and 11298 — the harvest
        # drained the noise faster than the mean absorbed the push. The
        # binding constraint is push *precision* at the 120 mm mouth, not the
        # exploration budget; v12.11 pairs the family midpoint 0.01 (keep the
        # scoring trajectories alive) with the widened 150 mm mouth in props.
        entropy_coef=0.01,
        learning_rate=1.0e-3,
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        # ── Runner ──
        num_steps_per_env=NUM_STEPS_PER_ENV,
        max_iterations=20_000,
        # ── Symmetry: off — see module docstring and five_foot's long note ──
        symmetry=False,
    )


def runner_cls() -> type:
    """Curriculum-aware velocity runner, the family's."""
    from ..common.runner import CurriculumRunner

    return CurriculumRunner
