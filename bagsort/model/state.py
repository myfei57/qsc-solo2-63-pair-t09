"""The folded document every subsystem reads and the reducer that builds it.

The store keeps an append-only journal of committed commands. This module turns
that journal into the materialised state and knows how to apply one command,
which is what makes restart-by-replay possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .entities import (
    AuditEntry,
    Bag,
    ChuteAssignment,
    ConflictCase,
    Evidence,
    Flight,
    LoadRecord,
    ReadRecord,
    Sequence,
    SortRecord,
    TransferCommand,
)
from bagsort.errors import ValidationError


@dataclass
class State:
    """Materialised view of every committed command."""

    flights: dict[str, Flight] = field(default_factory=dict)
    bags: dict[str, Bag] = field(default_factory=dict)
    mappings: dict[str, str] = field(default_factory=dict)
    assignments: dict[str, ChuteAssignment] = field(default_factory=dict)
    sort_records: list[SortRecord] = field(default_factory=list)
    transfer_commands: list[TransferCommand] = field(default_factory=list)
    load_records: list[LoadRecord] = field(default_factory=list)
    audit: list[AuditEntry] = field(default_factory=list)
    quota_limits: dict[str, int] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    sequence: Sequence = field(default_factory=Sequence)
    recheck_queue: list[str] = field(default_factory=list)
    reads: list[ReadRecord] = field(default_factory=list)
    evidence: dict[tuple[str, str, str], Evidence] = field(default_factory=dict)
    cases: dict[str, ConflictCase] = field(default_factory=dict)
    frozen_barcodes: set[str] = field(default_factory=set)
    review_queue: list[str] = field(default_factory=list)
    arbitration_policy: str = ""
    arbitration_policy_rationale: str = ""
    belt_carousel: str = ""
    belt_batch: int = 0
    belt_on_carousel: list[str] = field(default_factory=list)

    def copy(self) -> "State":
        return State(
            flights=dict(self.flights),
            bags=dict(self.bags),
            mappings=dict(self.mappings),
            assignments=dict(self.assignments),
            sort_records=list(self.sort_records),
            transfer_commands=list(self.transfer_commands),
            load_records=list(self.load_records),
            audit=list(self.audit),
            quota_limits=dict(self.quota_limits),
            counters=dict(self.counters),
            sequence=self.sequence,
            recheck_queue=list(self.recheck_queue),
            reads=list(self.reads),
            evidence=dict(self.evidence),
            cases=dict(self.cases),
            frozen_barcodes=set(self.frozen_barcodes),
            review_queue=list(self.review_queue),
            arbitration_policy=self.arbitration_policy,
            arbitration_policy_rationale=self.arbitration_policy_rationale,
            belt_carousel=self.belt_carousel,
            belt_batch=self.belt_batch,
            belt_on_carousel=list(self.belt_on_carousel),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "flights": {key: value.as_dict() for key, value in sorted(self.flights.items())},
            "bags": {key: value.as_dict() for key, value in sorted(self.bags.items())},
            "mappings": dict(sorted(self.mappings.items())),
            "assignments": {
                key: value.as_dict() for key, value in sorted(self.assignments.items())
            },
            "sort_records": [record.as_dict() for record in self.sort_records],
            "transfer_commands": [command.as_dict() for command in self.transfer_commands],
            "load_records": [record.as_dict() for record in self.load_records],
            "audit": [entry.as_dict() for entry in self.audit],
            "quota_limits": dict(sorted(self.quota_limits.items())),
            "counters": dict(sorted(self.counters.items())),
            "sequence": self.sequence.as_dict(),
            "recheck_queue": list(self.recheck_queue),
            "reads": [record.as_dict() for record in self.reads],
            "evidence": [
                self.evidence[key].as_dict()
                for key in sorted(self.evidence, key=lambda item: tuple(str(part) for part in item))
            ],
            "cases": {key: value.as_dict() for key, value in sorted(self.cases.items())},
            "frozen_barcodes": sorted(self.frozen_barcodes),
            "review_queue": list(self.review_queue),
            "arbitration_policy": self.arbitration_policy,
            "arbitration_policy_rationale": self.arbitration_policy_rationale,
            "belt": {
                "carousel": self.belt_carousel,
                "batch": self.belt_batch,
                "on_carousel": list(self.belt_on_carousel),
            },
        }

def apply_entry(state: State, kind: str, payload: dict[str, Any]) -> None:
    """Apply one journal command to ``state`` in place."""

    if kind == "flight.put":
        flight = Flight.from_dict(payload["flight"])
        state.flights[flight.id] = flight
    elif kind == "flight.remove":
        state.flights.pop(str(payload["flight_id"]), None)
        state.mappings = {
            bag_id: flight_id
            for bag_id, flight_id in state.mappings.items()
            if flight_id != str(payload["flight_id"])
        }
    elif kind == "chute.assign":
        assignment = ChuteAssignment.from_dict(payload)
        state.assignments[assignment.flight_id] = assignment
    elif kind == "bag.put":
        bag = Bag.from_dict(payload["bag"])
        state.bags[bag.id] = bag
    elif kind == "bag.remove":
        state.bags.pop(str(payload["bag_id"]), None)
        state.mappings.pop(str(payload["bag_id"]), None)
    elif kind == "mapping.put":
        state.mappings[str(payload["bag_id"])] = str(payload["flight_id"])
    elif kind == "sequence.put":
        state.sequence = Sequence.from_dict(payload)
    elif kind == "sort.record":
        state.sort_records.append(SortRecord.from_dict(payload["record"]))
    elif kind == "transfer.append":
        state.transfer_commands.append(TransferCommand.from_dict(payload["command"]))
    elif kind == "transfer.clear_before":
        floor = int(payload["batch"])
        state.transfer_commands = [
            command for command in state.transfer_commands if command.batch >= floor
        ]
    elif kind == "load.append":
        state.load_records.append(LoadRecord.from_dict(payload["record"]))
    elif kind == "audit.append":
        state.audit.append(AuditEntry.from_dict(payload["entry"]))
    elif kind == "quota.set":
        state.quota_limits[str(payload["flight_id"])] = int(payload["limit"])
    elif kind == "quota.adjust":
        flight_id = str(payload["flight_id"])
        base = state.quota_limits.get(flight_id, 0)
        state.quota_limits[flight_id] = base + int(payload["delta"])
    elif kind == "counter.add":
        name = str(payload["name"])
        state.counters[name] = state.counters.get(name, 0) + int(payload["delta"])
    elif kind == "recheck.enqueue":
        bag_id = str(payload["bag_id"])
        if bag_id not in state.recheck_queue:
            state.recheck_queue.append(bag_id)
    elif kind == "recheck.dequeue":
        bag_id = str(payload["bag_id"])
        state.recheck_queue = [item for item in state.recheck_queue if item != bag_id]
    elif kind == "recheck.clear":
        state.recheck_queue = []
    elif kind == "belt.set":
        state.belt_carousel = str(payload.get("carousel", ""))
        state.belt_batch = int(payload.get("batch", 0))
    elif kind == "belt.place":
        bag_id = str(payload["bag_id"])
        if bag_id not in state.belt_on_carousel:
            state.belt_on_carousel.append(bag_id)
    elif kind == "belt.clear":
        state.belt_on_carousel = []
        state.belt_carousel = str(payload.get("carousel", state.belt_carousel))
        state.belt_batch = int(payload.get("batch", state.belt_batch))
    elif kind == "arbitration.policy":
        state.arbitration_policy = str(payload["policy"])
        state.arbitration_policy_rationale = str(payload.get("rationale", ""))
    elif kind == "evidence.put":
        evidence = Evidence.from_dict(payload["evidence"])
        state.evidence[
            (evidence.barcode, evidence.kind, evidence.to_flight_id)
        ] = evidence
    elif kind == "read.put":
        read = ReadRecord.from_dict(payload["read"])
        state.reads = [item for item in state.reads if item.id != read.id]
        state.reads.append(read)
        state.reads.sort(key=lambda item: (item.ordinal, item.id))
    elif kind == "case.put":
        case = ConflictCase.from_dict(payload["case"])
        state.cases[case.id] = case
    elif kind == "freeze.add":
        state.frozen_barcodes.add(str(payload["barcode"]))
    elif kind == "freeze.remove":
        state.frozen_barcodes.discard(str(payload["barcode"]))
    elif kind == "review.enqueue":
        case_id = str(payload["case_id"])
        if case_id not in state.review_queue:
            state.review_queue.append(case_id)
    elif kind == "review.dequeue":
        case_id = str(payload["case_id"])
        state.review_queue = [item for item in state.review_queue if item != case_id]
    elif kind == "review.clear":
        state.review_queue = []
    else:  # pragma: no cover - guards against a typo in a new command kind
        raise ValidationError(f"unknown journal command {kind!r}")


def fold(entries: list[tuple[str, dict[str, Any]]]) -> State:
    """Rebuild a :class:`State` from a sequence of ``(kind, payload)`` pairs."""

    state = State()
    for kind, payload in entries:
        apply_entry(state, kind, payload)
    return state
