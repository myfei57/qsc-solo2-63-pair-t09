"""Arbitration: who owns a barcode when two readings claim different flights.

The plant has two ways to react to a barcode conflict, and they fail in
opposite directions:

* ``last_read_wins`` overwrites the mapping immediately. It never stops the
  belt, so it never misses a sort, but when the code is genuinely shared by two
  bags it re-routes the first bag onto the wrong aircraft: a *mis-sort*.
* ``freeze_pending_review`` freezes the barcode and holds every piece behind
  it at the review station. A mis-sort becomes impossible, but an unattended
  queue delays bags that would have sorted fine: a *missed sort*.

Mis-sorting luggage onto the wrong flight is the irreversible, safety-bearing
failure; a delayed sort is recovered by the review desk. The plant therefore
freezes. The choice and this rationale are stamped onto every adjudication so
a reviewer sees the rule a case was decided under.
"""

from __future__ import annotations

POLICY_LAST_READ = "last_read_wins"
POLICY_FREEZE = "freeze_pending_review"

POLICY_RATIONALE = (
    "freeze on conflict: taking the later read would silently re-route the "
    "earlier bag onto the wrong flight (mis-sort, effectively irreversible); "
    "freezing only delays sorting until the review desk attributes the code, "
    "which trades miss-sort risk for the smaller, recoverable kind"
)

# External evidence the arbiter trusts to decide without a human.
EVIDENCE_DCS_REROUTE = "dcs.reroute"
EVIDENCE_DUAL_TRACK = "dual_physical_track"
