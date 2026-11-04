"""Hyper-parameters for jumper.sixers_crown.

**Only this file changes this task's training hyper-parameters**; nothing this
task does not state is inherited silently.

The numbers are the family's — the same scalars `flat` starts from, stated
rather than reached through, so agreement is written down. **Symmetry stays
on** (the family default): unlike `five_foot`, nothing here is one-sided —
both claws rise together, and the mirror of a robot facing yaw pi is a robot
still facing yaw pi. The pump counter lives in the term instance, not in any
observation, so the mirror cannot contradict it.

One scalar is the task's own: `max_iterations=4000`. The celebration is a
standing ritual — no terrain, no object, no approach — so its horizon is the
shortest of the two new tasks; the gesture family converges under 1000, and
4000 is that budget with room for the three-count ceiling to be discovered
the hard way (pumped a fourth time, paid nothing, stopped).
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return jumper_ppo_baseline(
        experiment_name="jumper.sixers_crown",
        # ── Network ──
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=1.0,
        # ── PPO ──
        entropy_coef=0.005,
        learning_rate=1.0e-3,
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        # ── Runner ──
        num_steps_per_env=24,
        max_iterations=4_000,
    )
