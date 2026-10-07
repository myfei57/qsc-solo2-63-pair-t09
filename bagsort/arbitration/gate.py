"""Sort gate: a bag whose ownership is not settled never reaches a chute.

The gate is the single chokepoint the sorter consults before dispatch. A bag
is *sortable* only when its barcode is not frozen by an open conflict. Every
refusal is recorded as an audit entry so a denied sort is as visible as a
granted one.
"""

from __future__ import annotations

from dataclasses import dataclass

from bagsort.audit.chain import AuditChain
from bagsort.model.entities import Bag
from bagsort.model.errors import unknown_bag
from bagsort.store.store import Store


@dataclass(frozen=True)
class GateVerdict:
    """Whether one bag may be sorted, and why."""

    allowed: bool
    reason: str
    case_id: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"allowed": self.allowed, "reason": self.reason, "case_id": self.case_id}


class SortGate:
    """Authorises dispatch against the set of frozen barcodes."""

    def __init__(self, store: Store, audit: AuditChain | None = None) -> None:
        self._store = store
        self._audit = audit

    def case_for(self, bag: Bag) -> str:
        state = self._store.state()
        if bag.barcode not in state.frozen_barcodes:
            return ""
        for case in state.cases.values():
            if case.barcode == bag.barcode:
                return case.id
        return ""

    def check(self, bag: Bag, record: bool = True) -> GateVerdict:
        stored = self._store.state().bags.get(bag.id)
        if stored is None:
            raise unknown_bag(bag.id)
        case_id = self.case_for(stored)
        if case_id:
            verdict = GateVerdict(
                allowed=False,
                reason="barcode frozen by open conflict; review desk must attribute it",
                case_id=case_id,
            )
            if record and self._audit is not None:
                self._audit.record(
                    "sort_blocked",
                    stored.id,
                    "sorter",
                    {"barcode": stored.barcode, "case_id": case_id},
                )
            return verdict
        return GateVerdict(allowed=True, reason="ownership settled")
