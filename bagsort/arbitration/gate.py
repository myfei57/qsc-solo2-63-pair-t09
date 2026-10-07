"""The sortation gate: baggage with unsettled ownership never reaches a chute.

The sorter asks the gate before dispatching. A bag is sortable only when its
barcode is not frozen, it is not one side of an open conflict case, and its
bag-to-flight mapping exists. Keeping the rule in one place means neither the
sorter nor the injection belt can route a bag whose ownership the arbitrator
has not settled.
"""

from __future__ import annotations

from bagsort.arbitration.enums import ClaimStatus
from bagsort.errors import OrderingError
from bagsort.model.entities import Bag
from bagsort.model.enums import BagState
from bagsort.store.store import Store


class SortationGate:
    """Decides whether a bag is allowed onto the sorting line."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def _claim_for(self, bag_id: str):
        return self._store.state().read_claims.get(f"{bag_id}:read")

    def open_case_for(self, bag: Bag) -> str:
        """The id of the open conflict involving the bag, or an empty string."""

        state = self._store.state()
        for case in state.conflict_cases.values():
            if not case.is_open or case.barcode != bag.barcode:
                continue
            involved = {
                state.read_claims[case.prior_claim_id].bag_id,
                state.read_claims[case.latter_claim_id].bag_id,
            }
            if bag.id in involved:
                return case.id
        return ""

    def reason_blocked(self, bag: Bag) -> str:
        """Why a bag must not be sorted; empty when it may proceed."""

        state = self._store.state()
        if bag.barcode in state.frozen_barcodes:
            return "barcode_frozen_pending_review"
        open_case = self.open_case_for(bag)
        if open_case:
            return f"open_conflict_case:{open_case}"
        claim = self._claim_for(bag.id)
        if claim is not None and claim.status is ClaimStatus.PENDING:
            return "read_ownership_unsettled"
        if bag.id not in state.mappings:
            return "no_flight_mapping"
        if bag.state is BagState.DIVERTED:
            return "bag_diverted"
        return ""

    def is_sortable(self, bag: Bag) -> bool:
        return not self.reason_blocked(bag)

    def require_sortable(self, bag: Bag) -> None:
        reason = self.reason_blocked(bag)
        if reason:
            raise OrderingError(f"bag {bag.id!r} is not sortable: {reason}")
