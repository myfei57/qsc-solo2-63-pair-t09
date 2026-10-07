"""Correspondence invariants between reads, mappings, cases and the queues.

The arbitration contract is "read commit, bag-to-flight mapping and review
queue line up one to one". This module turns that sentence into checkable
invariants over the committed state, and :func:`replay_projection` rebuilds an
attribution projection straight from the journal so the audit trail can be
replayed by time and compared against the live materialised state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bagsort.model.entities import ConflictCase, ReadRecord
from bagsort.model.enums import ConflictState, ReadStatus
from bagsort.store.journal import JournalEntry
from bagsort.store.store import Store


@dataclass
class Projection:
    """Attribution folded from journal commands, independent of the live state."""

    reads: dict[str, ReadRecord] = field(default_factory=dict)
    mappings: dict[str, str] = field(default_factory=dict)
    cases: dict[str, ConflictCase] = field(default_factory=dict)
    frozen: set[str] = field(default_factory=set)
    review_queue: list[str] = field(default_factory=list)
    recheck_queue: list[str] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)

    def at(self, instant: str) -> "Projection":
        """Fold only commands whose event time is at or before ``instant``."""

        return fold_timeline([item for item in self.timeline if item["at"] <= instant])


def apply_command(
    projection: Projection,
    kind: str,
    payload: dict[str, Any],
    at: str = "",
    sequence: int = 0,
) -> None:
    if kind == "read.put":
        read = ReadRecord.from_dict(payload["read"])
        projection.reads[read.id] = read
        projection.timeline.append(
            {"at": at or read.at, "sequence": sequence, "event": "reading_committed",
             "read": read.as_dict()}
        )
    elif kind == "mapping.put":
        projection.mappings[str(payload["bag_id"])] = str(payload["flight_id"])
    elif kind == "case.put":
        case = ConflictCase.from_dict(payload["case"])
        projection.cases[case.id] = case
        projection.timeline.append(
            {"at": at or case.created_at, "sequence": sequence,
             "event": case.state.value, "case_id": case.id,
             "decision": case.decision.value, "barcode": case.barcode}
        )
    elif kind in ("freeze.add", "freeze.remove"):
        barcode = str(payload["barcode"])
        if kind == "freeze.add":
            projection.frozen.add(barcode)
        else:
            projection.frozen.discard(barcode)
        projection.timeline.append(
            {"at": at, "sequence": sequence, "event": kind,
             "case_id": "", "decision": "", "barcode": barcode}
        )
    elif kind == "freeze.add":
        projection.frozen.add(str(payload["barcode"]))
    elif kind == "freeze.remove":
        projection.frozen.discard(str(payload["barcode"]))
    elif kind == "review.enqueue":
        case_id = str(payload["case_id"])
        if case_id not in projection.review_queue:
            projection.review_queue.append(case_id)
    elif kind == "review.dequeue":
        case_id = str(payload["case_id"])
        projection.review_queue = [item for item in projection.review_queue if item != case_id]
    elif kind == "recheck.enqueue":
        bag_id = str(payload["bag_id"])
        if bag_id not in projection.recheck_queue:
            projection.recheck_queue.append(bag_id)
    elif kind == "recheck.dequeue":
        bag_id = str(payload["bag_id"])
        projection.recheck_queue = [item for item in projection.recheck_queue if item != bag_id]


def fold_entries(entries: list[JournalEntry]) -> Projection:
    projection = Projection()
    for entry in entries:
        apply_command(projection, entry.kind, entry.payload, entry.at, entry.sequence)
    return projection


def fold_timeline(items: list[dict[str, Any]]) -> Projection:
    """Re-fold timeline rows (used for time scoped replay)."""

    projection = Projection()
    seen_reads: set[str] = set()
    for item in sorted(items, key=lambda row: (row["at"], row["sequence"])):
        if item["event"] == "reading_committed":
            read = item["read"]
            # Keep the first commit of a reading; later rows are status rewrites.
            if read["id"] in seen_reads:
                continue
            seen_reads.add(read["id"])
            apply_command(projection, "read.put", {"read": read}, item["at"], item["sequence"])
        elif item["event"] in ("freeze.add", "freeze.remove"):
            apply_command(projection, item["event"], {"barcode": item["barcode"]},
                          item["at"], item["sequence"])
        else:
            apply_command(
                projection,
                "case.put",
                {"case": {"id": item["case_id"], "state": item["event"],
                          "decision": item["decision"], "barcode": item["barcode"]}},
                item["at"],
                item["sequence"],
            )
    return projection


def replay_projection(store: Store, instant: str | None = None) -> Projection:
    """Rebuild attribution from the committed journal, optionally as-of a time."""

    projection = fold_entries(store.entries())
    return projection.at(instant) if instant else projection


def verify_correspondence(store: Store) -> list[str]:
    """Return a list of broken one-to-one correspondences; empty means healthy."""

    state = store.state()
    violations: list[str] = []

    # 1. Every open case freezes exactly one barcode and sits in the queue once.
    for case in state.cases.values():
        queued = state.review_queue.count(case.id)
        if case.state is ConflictState.OPEN and queued != 1:
            violations.append(
                f"case {case.id!r} is open but appears {queued}x in the review queue"
            )
        if case.state is not ConflictState.OPEN and case.id in state.review_queue:
            violations.append(f"settled case {case.id!r} still occupies the review queue")
        if case.state is ConflictState.OPEN and case.barcode not in state.frozen_barcodes:
            violations.append(f"open case {case.id!r} does not freeze barcode {case.barcode!r}")

    # 2. Every frozen barcode has exactly one open case; an unfrozen one has none.
    for barcode in state.frozen_barcodes:
        owners = [c.id for c in state.cases.values()
                  if c.barcode == barcode and c.state is ConflictState.OPEN]
        if len(owners) != 1:
            violations.append(f"frozen barcode {barcode!r} has {len(owners)} open cases")

    # 3. Every attributed read resolves to a bag that exists and is mapped.
    for read in state.reads:
        if read.status in (ReadStatus.ATTRIBUTED, ReadStatus.REROUTED) and read.bag_id:
            if read.bag_id not in state.bags:
                violations.append(f"read {read.id!r} points at missing bag {read.bag_id!r}")
            elif state.mappings.get(read.bag_id) is None:
                violations.append(f"read {read.id!r} bag {read.bag_id!r} has no flight mapping")

    # 4. A reroute case's bag is mapped to the last claimed flight; a distinct
    #    case's second bag exists and is mapped.
    for case in state.cases.values():
        if case.decision.value == "reroute" and case.state is not ConflictState.OPEN:
            mapped = state.mappings.get(case.bag_ids[0])
            if mapped != case.claimed_flights[-1]:
                violations.append(
                    f"reroute case {case.id!r} maps to {mapped!r}, "
                    f"expected {case.claimed_flights[-1]!r}"
                )
        if case.decision.value == "distinct":
            if len(case.bag_ids) < 2:
                violations.append(f"distinct case {case.id!r} never produced a second bag id")
            elif state.mappings.get(case.bag_ids[-1]) is None:
                violations.append(f"distinct case {case.id!r} second bag has no mapping")

    # 5. Replay projection must equal the materialised arbitration state.
    projection = fold_entries(store.entries())
    if set(projection.frozen) != set(state.frozen_barcodes):
        violations.append("replay disagrees with the live frozen set")
    if projection.review_queue != state.review_queue:
        violations.append("replay disagrees with the live review queue")
    if {bid: flight for bid, flight in projection.mappings.items()
            if bid in state.mappings} != {bid: state.mappings[bid] for bid in projection.mappings
                                          if bid in state.mappings}:
        violations.append("replay disagrees with live mappings")
    if set(projection.reads) != {read.id for read in state.reads}:
        violations.append("replay disagrees with the live read ledger")
    else:
        for read_id, folded in projection.reads.items():
            live = next(item for item in state.reads if item.id == read_id)
            if folded.as_dict() != live.as_dict():
                violations.append(f"replay disagrees with read {read_id!r}")

    # 6. Nothing frozen is permitted near a chute.
    for bag in state.bags.values():
        if bag.barcode in state.frozen_barcodes and bag.state.value == "sorted":
            violations.append(f"frozen bag {bag.id!r} has a sort record")

    return violations


def correspondence_holds(store: Store) -> bool:
    return not verify_correspondence(store)
