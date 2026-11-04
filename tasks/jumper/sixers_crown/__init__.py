"""jumper.sixers_crown — the celebration: turn away, raise both claws, pump three times.

The dunk program's second half. The robot starts from standing (the scored
ball is behind it), turns to face away from the hoop, lifts both front claws
above its head, and drives exactly three crown-pumps — raise, press down,
raise — before holding still. Trained as its own short task; a bundle mode
switch (`jump`-style, `A` or Space in a later `deploy/manifests.json` entry)
hands it the robot when the dunk succeeds.

Built on `jumper.flat`, not `five_foot`: **all six feet stay grounded** and
the 20-joint action already owns every arm joint, so nothing about the walk
skeleton is touched — the claws rise from a standing base the family has
trained a thousand times, the same posture vocabulary `gesture_*` uses.

The pump is counted, not cycled: the term counts rising edges of the raised
state and pays one pump per completed up-down-up, up to three, and stops
paying after the third — a policy that pumps forever collects nothing extra,
and one that pumps twice collects two.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.sixers_crown",
    assets=JUMPER_ASSETS,
    description="jumper hexapod celebration: turn around, raise both claws, three counted crown-pumps",
    tags=("jumper", "flat", "gesture", "celebration", "crown"),
)
