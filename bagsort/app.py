"""Assembly of the handling line.

Every subsystem is constructed here and every operator visible action is a
method on :class:`Plant`, which is what keeps the console handlers thin and
lets the CLI drive the same operations the HTTP API exposes.
"""

from __future__ import annotations

from dataclasses import dataclass

from bagsort.audit.chain import AuditChain
from bagsort.arbitration.arbitrator import Arbitrator
from bagsort.arbitration.gate import SortationGate
from bagsort.belt.belt import Belt
from bagsort.checkin.desk import Desk
from bagsort.clock import Clock
from bagsort.errors import ConflictError
from bagsort.events.bus import EventBus
from bagsort.flight.book import FlightBook
from bagsort.flight.close import FlightCloser
from bagsort.flight.mapping import BagMapping
from bagsort.flight.schedule import ScheduleBoard
from bagsort.ids import IdFactory
from bagsort.inject.injector import Injector
from bagsort.model.entities import Bag, Flight, Reading, SortRecord
from bagsort.model.errors import unknown_bag
from bagsort.ns.chutes import ChuteRegistry
from bagsort.ns.namespace import Namespace
from bagsort.ns.topology import handling_line
from bagsort.quota.quota import QuotaBook
from bagsort.recheck.queue import RecheckQueue
from bagsort.reporting.report import PlantReport
from bagsort.sequence.verdict import VerdictTracker
from bagsort.sorter.sorter import Sorter
from bagsort.store.store import Store
from bagsort.tag.history import ReadHistory
from bagsort.tag.reader import TagReader
from bagsort.chute.assigner import ChuteAssigner
from bagsort.chute.history import AssignmentHistory
from bagsort.console.metrics import Metrics
from bagsort.console.telemetry import Telemetry

DEFAULT_TERMINAL = "T1"


@dataclass
class Plant:
    """The wired plant plus the operator level operations the console calls."""

    store: Store
    clock: Clock
    namespace: Namespace
    registry: ChuteRegistry
    bus: EventBus
    telemetry: Telemetry
    metrics: Metrics
    reader: TagReader
    reads: ReadHistory
    book: FlightBook
    mapping: BagMapping
    schedule: ScheduleBoard
    closer: FlightCloser
    chutes: ChuteAssigner
    chute_history: AssignmentHistory
    sorter: Sorter
    injector: Injector
    desk: Desk
    recheck: RecheckQueue
    belt: Belt
    quota: QuotaBook
    arbitrator: Arbitrator
    gate: SortationGate
    audit: AuditChain
    report: PlantReport

    @property
    def topology(self):
        return handling_line()

    # -- construction ------------------------------------------------------

    @classmethod
    def build(
        cls,
        store: Store,
        terminal: str = DEFAULT_TERMINAL,
        clock: Clock | None = None,
    ) -> "Plant":
        clock = clock or Clock()
        namespace = Namespace(terminal)
        registry = ChuteRegistry()
        bus = EventBus()
        telemetry = Telemetry()
        metrics = Metrics()
        bus.subscribe("*", lambda event: telemetry.observe(event.topic))

        reader = TagReader(store, namespace, clock)
        reads = ReadHistory(store)
        book = FlightBook(store, clock)
        mapping = BagMapping(store, clock)
        schedule = ScheduleBoard(store, clock)
        chutes = ChuteAssigner(store, clock)
        chute_history = AssignmentHistory(store)
        sorter = Sorter(store, mapping, chutes, clock, VerdictTracker())
        injector = Injector(store, book, sorter, clock)
        quota = QuotaBook(store, clock)
        gate = SortationGate(store)
        arbitrator = Arbitrator(store, namespace, clock, quota, injector)
        desk = Desk(
            store, reader, book, mapping, injector, quota, clock, handling_line(), arbitrator
        )
        closer = FlightCloser(book)
        recheck = RecheckQueue(store, book, clock)
        belt = Belt(store, clock, IdFactory("cmd"))

        audit = AuditChain(store, clock, handling_line())
        audit_subscriber = _AuditSubscriber(audit)
        for topic in (
            "bag.checked_in",
            "bag.sorted",
            "bag.diverted",
            "bag.recheck_enqueued",
            "flight.closed",
            "chute.assigned",
            "belt.switched",
            "bag.remapped",
            "barcode.conflict_detected",
            "barcode.decided",
            "barcode.escalated",
            "barcode.unfrozen",
            "barcode.reflight_recorded",
        ):
            bus.subscribe(topic, audit_subscriber)

        report = PlantReport(
            store, book, chutes, belt, quota, audit, namespace, handling_line()
        )
        return cls(
            store=store,
            clock=clock,
            namespace=namespace,
            registry=registry,
            bus=bus,
            telemetry=telemetry,
            metrics=metrics,
            reader=reader,
            reads=reads,
            book=book,
            mapping=mapping,
            schedule=schedule,
            closer=closer,
            chutes=chutes,
            chute_history=chute_history,
            sorter=sorter,
            injector=injector,
            desk=desk,
            recheck=recheck,
            belt=belt,
            quota=quota,
            arbitrator=arbitrator,
            gate=gate,
            audit=audit,
            report=report,
        )

    # -- operator operations ----------------------------------------------

    def register_flight(self, code: str, carousel: str = "", chute: str = "") -> Flight:
        flight = Flight(
            id=self.namespace.flight_id(code),
            code=code.strip().upper(),
            terminal=self.namespace.terminal,
            carousel=self.namespace.carousel_id(carousel) if carousel else "",
        )
        self.book.register(flight)
        if chute:
            self.assign_chute(flight.id, self.namespace.chute_id(chute))
        return self.book.get(flight.id)

    def assign_chute(self, flight_id: str, chute_id: str):
        assignment = self.chutes.assign(flight_id, chute_id)
        self.bus.publish("chute.assigned", flight_id=flight_id, chute_id=chute_id)
        return assignment

    def read_barcode(self, barcode: str) -> Reading:
        return self.reader.read(barcode)

    def check_in(self, barcode: str, flight_id: str, weight: float = 0.0) -> Bag:
        bag = self.desk.check_in(barcode, flight_id, weight)
        self.bus.publish("bag.checked_in", bag_id=bag.id, flight_id=flight_id)
        return bag

    def check_in_adjudicated(
        self, barcode: str, flight_id: str, weight: float = 0.0
    ):
        """Submit a read and return the outcome together with its adjudication.

        Publishes the conflict/decision/escalation events exactly once, only
        when the adjudication actually created them.
        """

        outcome = self.arbitrator.check_in(barcode, flight_id, weight)
        self.bus.publish("bag.checked_in", bag_id=outcome.bag.id, flight_id=flight_id)
        if outcome.case_id:
            case = self.arbitrator.case(outcome.case_id)
            self.bus.publish(
                "barcode.conflict_detected",
                barcode=case.barcode,
                case_id=case.id,
            )
            if outcome.decision_id:
                decision = self.arbitrator.decision_for(case.id)
                self.bus.publish(
                    "barcode.decided",
                    barcode=case.barcode,
                    case_id=case.id,
                    decision_id=decision.id,
                    verdict=decision.verdict.value,
                    decided_by=decision.decided_by,
                )
            if outcome.escalated:
                self.bus.publish(
                    "barcode.escalated",
                    barcode=case.barcode,
                    case_id=case.id,
                    recheck_item_id=case.recheck_item_id,
                )
        return outcome

    def record_reflight(self, barcode: str, from_flight_id: str, to_flight_id: str):
        directive = self.arbitrator.record_reflight(barcode, from_flight_id, to_flight_id)
        self.bus.publish(
            "barcode.reflight_recorded",
            barcode=directive.barcode,
            from_flight_id=directive.from_flight_id,
            to_flight_id=directive.to_flight_id,
        )
        return directive

    def report_conflict(self, prior_claim_id: str, latter_claim_id: str):
        """Idempotently (re)report a conflict between two committed reads."""

        before = {case.id: case for case in self.arbitrator.cases()}
        case = self.arbitrator.report_conflict(prior_claim_id, latter_claim_id)
        if case.id not in before:
            self.bus.publish("barcode.conflict_detected", barcode=case.barcode, case_id=case.id)
            decision = self.arbitrator.decision_for(case.id)
            if decision is not None:
                self.bus.publish(
                    "barcode.decided",
                    barcode=case.barcode,
                    case_id=case.id,
                    decision_id=decision.id,
                    verdict=decision.verdict.value,
                    decided_by=decision.decided_by,
                )
            if case.recheck_item_id:
                self.bus.publish(
                    "barcode.escalated",
                    barcode=case.barcode,
                    case_id=case.id,
                    recheck_item_id=case.recheck_item_id,
                )
        return case

    def resolve_conflict(self, case_id: str, verdict: str, decided_by: str, note: str = ""):
        decision = self.arbitrator.resolve_case(case_id, verdict, decided_by, note)
        self.bus.publish(
            "barcode.decided",
            barcode=decision.barcode,
            case_id=decision.case_id,
            decision_id=decision.id,
            verdict=decision.verdict.value,
            decided_by=decision.decided_by,
        )
        self.bus.publish("barcode.unfrozen", barcode=decision.barcode, case_id=decision.case_id)
        return decision

    def next_conflict_recheck(self):
        case = self.arbitrator.next_recheck()
        return case

    def is_sortable(self, bag_id: str) -> bool:
        return self.gate.is_sortable(self.bag(bag_id))

    def sort_block_reason(self, bag_id: str) -> str:
        return self.gate.reason_blocked(self.bag(bag_id))

    def arbitration_consistency(self) -> dict:
        return self.arbitrator.consistency()

    def cancel_check_in(self, bag_id: str) -> Bag:
        bag = self.desk.cancel(bag_id)
        self.bus.publish("bag.diverted", bag_id=bag.id, reason="checkin_cancelled")
        return bag

    def bag(self, bag_id: str) -> Bag:
        bag = self.store.state().bags.get(bag_id)
        if bag is None:
            raise unknown_bag(bag_id)
        return bag

    def push_bag(self, bag_id: str, chute_id: str) -> SortRecord:
        bag = self.bag(bag_id)
        record = self.injector.push(bag, chute_id)
        self.bus.publish("bag.sorted", bag_id=bag.id, chute_id=record.chute_id)
        return record

    def retry_bag(self, bag_id: str) -> bool:
        bag = self.bag(bag_id)
        sorted_now = self.injector.retry(bag)
        if sorted_now:
            self.bus.publish("bag.sorted", bag_id=bag.id, retry=True)
        return sorted_now

    def divert_bag(self, bag_id: str) -> Bag:
        bag = self.bag(bag_id)
        updated = self.sorter.divert(bag)
        self.bus.publish("bag.diverted", bag_id=bag.id)
        return updated

    def remap_bag(self, bag_id: str, flight_id: str) -> Bag:
        self.book.require_open(flight_id)
        self.mapping.remap(bag_id, flight_id)
        self.bus.publish("bag.remapped", bag_id=bag_id, flight_id=flight_id)
        return self.bag(bag_id)

    def revise_schedule(self, flight_id: str, cutoff_at: str):
        return self.schedule.revise(flight_id, cutoff_at)

    def close_flight(self, flight_id: str) -> Flight:
        flight = self.closer.close(flight_id, self.injector)
        self.bus.publish("flight.closed", flight_id=flight_id)
        return flight

    def enqueue_recheck(self, bag_id: str) -> Bag:
        bag = self.bag(bag_id)
        self.recheck.enqueue(bag)
        self.bus.publish("bag.recheck_enqueued", bag_id=bag_id)
        return bag

    def process_recheck(self) -> Bag | None:
        return self.recheck.dequeue()

    def switch_belt(self, carousel: str, batch: int) -> int:
        cleared = self.belt.switch(carousel, batch)
        self.bus.publish("belt.switched", carousel=carousel, batch=batch)
        return cleared

    def rotate_belt(self) -> int:
        return self.belt.rotate()

    def append_transfer(self, flight_id: str, carousel: str, batch: int):
        return self.belt.append_transfer(flight_id, carousel, batch)

    def set_quota(self, flight_id: str, limit: int) -> int:
        return self.quota.set_limit(flight_id, limit)

    def verify(self) -> bool:
        return self.audit.verify()


class _AuditSubscriber:
    """Turns published operations into linked audit entries."""

    _STAGE = {
        "bag.checked_in": ("checkin", "checkin"),
        "bag.sorted": ("sort", "sorter"),
        "bag.diverted": ("diversion", "sorter"),
        "bag.recheck_enqueued": ("recheck", "recheck"),
        "flight.closed": ("flight_close", "audit"),
        "chute.assigned": ("chute_assign", "chute"),
        "belt.switched": ("belt_switch", "belt"),
        "bag.remapped": ("remap", "checkin"),
        "barcode.conflict_detected": ("barcode_conflict", "recheck"),
        "barcode.decided": ("barcode_decision", "recheck"),
        "barcode.escalated": ("barcode_escalation", "recheck"),
        "barcode.unfrozen": ("barcode_unfreeze", "recheck"),
        "barcode.reflight_recorded": ("reflight_evidence", "checkin"),
    }

    def __init__(self, audit: AuditChain) -> None:
        self._audit = audit

    def __call__(self, event) -> None:
        mapping = self._STAGE.get(event.topic)
        if mapping is None:
            raise ConflictError(f"no audit mapping for topic {event.topic!r}")
        kind, stage = mapping
        subject = str(
            event.payload.get("bag_id")
            or event.payload.get("flight_id")
            or event.payload.get("barcode")
            or event.payload.get("carousel")
            or ""
        )
        detail = {
            key: value
            for key, value in event.payload.items()
            if key not in {"bag_id", "flight_id", "barcode"}
        }
        self._audit.record(kind, subject, stage, detail)
