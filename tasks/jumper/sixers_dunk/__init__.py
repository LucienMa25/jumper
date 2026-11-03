"""jumper.sixers_dunk — approach, grasp, carry and release a ball into a hoop.

Scaffolded on top of `jumper.five_foot` (the same five-feet locomotion with the
left-front arm carried as a claw). What this task adds is a **dunk world**:
one leather ball on the floor ahead of the robot and one low hoop further on,
plus the staged rewards that say what the job is — walk to the ball, lift it,
carry it to the hoop, let go inside. The operator then turns around and the
crown celebration is trained as its own short task (see TODOs).

The claw's physics, payload curriculum and controls conventions are inherited,
not copied: `env_cfg` calls `five_foot.env_cfg` and layers this world onto the
returned config. Where five_foot's row of props is `--objects`, this task's
dunk world is `--hoop`.

**Honest state of the scaffold (2026-11-03):** the staged rewards are written
against the entity positions and are trainable, but the *grasp itself* is not
yet a trained term — five_foot trains grasping through its payload curriculum
and convex jaw geometry (`jaws.py`), and the ball is a harder grip than any
prop in that row (sphere, point contact). Stage 2 on a GPU must close that
gap; see `mdp/terms.py` module docstring for the plan.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.sixers_dunk",
    assets=JUMPER_ASSETS,
    description="jumper hexapod on five feet with a claw: approach a ball, carry it to a hoop, release",
    tags=("locomotion", "jumper", "flat", "five_foot", "manipulation", "dunk"),
)


def cli_args(group):
    """`--hoop` — the dunk world. Replay or training, unlike five_foot's
    `--objects` this flag is designed to be trained with (see README below)."""
    group.add_argument(
        "--hoop",
        action="store_true",
        help="put the dunk world in front of the robot: a leather ball, a low "
        "hoop, and the staged approach/lift/carry/score rewards. Physical, so "
        "train and replay must both pass it (the friction model changes for "
        "every contact)",
    )
    return ("hoop",)
