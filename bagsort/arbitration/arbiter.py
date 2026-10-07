"""The arbiter: commit reads, adjudicate conflicts, settle them at review.

Every mutation is a single journal transaction, so a reading commit, the
mapping it produces and the review/queue linkage it needs either all land
together or none do. Re-running the same report or the same resolution is
idempotent: the natural keys (read ordinal, case id) make a repeat return the
records already produced instead of creating a second adjudication.

Adjudication uses only two kinds of trusted external evidence:

* a DCS record saying one physical bag moved flight -> *reroute*, resolved
  automatically;
* two physical tracks observed at once for one code -> *distinct*, frozen and
  queued for a relabel.

With neither, the case is *inconclusive*: frozen and queued for a human.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bagsort.clock import Clock
from bagsort.errors import NotFoundError, OrderingError, ValidationError
from bagsort.model.entities import Bag, ConflictCase, Evidence, ReadRecord
from bagsort.model.enums import BagState, ConflictDecision, ConflictState, ReadStatus
from bagsort.model.validate import require_text
from bagsort.sequence.allocator import allocate
from bagsort.store.store import Store
from . import policy


@dataclass(frozen=True)
class ReportResult:
    """What one :meth:`Arbiter.report_reading` produced."""

    read: ReadRecord
    outcome: str
    case: ConflictCase | None = None
    bag: Bag | None = None
    duplicate: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "read": self.read.as_dict(),
            "outcome": self.outcome,
            "case": self.case.as_dict() if self.case else None,
            "bag": self.bag.as_dict() if self.bag else None,
            "duplicate": self.duplicate,
        }


class Arbiter:
    """Owns the read ledger, conflict cases and the manual review queue."""

    def __init__(self, store: Store, clock: Clock, namespace: object) -> None:
        self._store = store
        self._clock = clock
        self._ns = namespace

    # -- reads -------------------------------------------------------------

    def report_reading(self, barcode: str, flight_id: str) -> ReportResult:
        """Commit one reading and adjudicate it against the ledger.

        Outcomes:

        * ``first`` -- the code is new, bag and mapping are created;
        * ``duplicate`` -- the code re-claimed the flight it already maps to;
        * ``reroute`` -- one bag changing flight, proven by DCS, auto-settled;
        * ``distinct`` -- two physical bags share the code, frozen + queued;
        * ``inconclusive`` -- cannot tell, frozen + queued;
        * ``rereported`` -- repeat report against an open case, no new ruling.
        """

        code = require_text(barcode, "barcode")
        flight = require_text(flight_id, "flight_id")
        state = self._store.state()
        self._require_flight(flight)

        prior = _latest_attributed(state, code)
        reads_of_code = [read for read in state.reads if read.barcode == code]
        ordinal = len(reads_of_code) + 1

        # A repeat report of the exact same (code, flight) pair is a duplicate
        # read of the already attributed bag: nothing new is adjudicated and no
        # command is appended, so retrying the report leaves state untouched.
        # This holds whether the bag's current flight came from a fresh read or
        # from an adjudicated reroute.
        if prior is not None and prior.flight_id == flight and not _open_case(state, code):
            return ReportResult(read=prior, outcome="duplicate", duplicate=True,
                                bag=state.bags[prior.bag_id])

        open_case = _open_case(state, code)
        if open_case is not None:
            # Re-submitting the identical conflict report must not produce a
            # second adjudication; only a genuinely new claimed flight adds
            # another attached reading.
            if flight in open_case.claimed_flights:
                existing = self.reads(code)
                read = next(
                    (item for item in reversed(existing) if item.flight_id == flight),
                    existing[-1],
                )
                return ReportResult(read=read, outcome="rereported", case=open_case,
                                    duplicate=True)
            return self._attach_report(open_case, code, flight, ordinal)

        if prior is None:
            return self._commit_first(state, code, flight, ordinal)
        return self._open_conflict(state, code, prior, flight, ordinal)

    def reads(self, barcode: str | None = None) -> list[ReadRecord]:
        reads = self._store.state().reads
        if barcode is None:
            return list(reads)
        return [read for read in reads if read.barcode == barcode]

    # -- evidence ----------------------------------------------------------

    def record_evidence(
        self,
        barcode: str,
        kind: str,
        *,
        from_flight_id: str = "",
        to_flight_id: str = "",
        source: str = "",
        at: str = "",
    ) -> Evidence:
        if kind not in (policy.EVIDENCE_DCS_REROUTE, policy.EVIDENCE_DUAL_TRACK):
            raise ValidationError(f"unknown evidence kind {kind!r}")
        evidence = Evidence(
            barcode=require_text(barcode, "barcode"),
            kind=kind,
            from_flight_id=from_flight_id,
            to_flight_id=to_flight_id,
            source=source,
            at=at or self._clock.now_iso(),
        )
        with self._store.session() as session:
            self._ensure_policy(session.state)
            key = (evidence.barcode, evidence.kind, evidence.to_flight_id)
            session.state.evidence[key] = evidence
            session.record("evidence.put", {"evidence": evidence.as_dict()}, evidence.at)
        return evidence

    # -- cases and review --------------------------------------------------

    def cases(self) -> list[ConflictCase]:
        return [self._store.state().cases[key] for key in sorted(self._store.state().cases)]

    def case(self, case_id: str) -> ConflictCase:
        case = self._store.state().cases.get(case_id)
        if case is None:
            raise NotFoundError(f"conflict case {case_id!r} does not exist")
        return case

    def review_queue(self) -> list[ConflictCase]:
        state = self._store.state()
        return [state.cases[case_id] for case_id in state.review_queue if case_id in state.cases]

    def resolve(
        self,
        case_id: str,
        decision: str,
        *,
        operator: str = "review_desk",
        new_flight_id: str = "",
        note: str = "",
    ) -> ConflictCase:
        """Human resolution at the review desk; idempotent on a settled case.

        The case must be the head of the review queue: rulings happen in FIFO
        arrival order. Ruling freezes/unfreezes ownership; :meth:`complete_review`
        releases the queue slot afterwards.
        """

        case = self.case(case_id)
        if case.state is not ConflictState.OPEN:
            # Re-submitting the same conflict produces no second adjudication.
            return case
        queue = self._store.state().review_queue
        if not queue or queue[0] != case.id:
            raise OrderingError(
                f"case {case.id!r} is not at the head of the review queue"
            )
        chosen = ConflictDecision(decision)
        if chosen not in (ConflictDecision.REROUTE, ConflictDecision.DISTINCT):
            raise ValidationError("manual resolution must be 'reroute' or 'distinct'")
        state = self._store.state()
        at = self._clock.now_iso()
        reason = note or f"manual review desk ruling: {chosen.value}"
        if chosen is ConflictDecision.REROUTE:
            target = new_flight_id or case.claimed_flights[-1]
            self._require_flight(target)
            updated = self._apply_reroute(state, case, target, operator, reason, at, manual=True)
        else:
            updated = self._apply_distinct(state, case, operator, reason, at, manual=True)
        return updated

    def next_review(self) -> ConflictCase | None:
        queue = self.review_queue()
        return queue[0] if queue else None

    def complete_review(self, case_id: str) -> ConflictCase:
        """A review operator has finished a case: dequeue it FIFO."""

        case = self.case(case_id)
        state = self._store.state()
        if case.id not in state.review_queue:
            raise OrderingError(f"case {case.id!r} is not waiting at the review desk")
        with self._store.session() as session:
            session.state.review_queue = [
                item for item in session.state.review_queue if item != case.id
            ]
            session.record("review.dequeue", {"case_id": case.id}, self._clock.now_iso())
        return self.case(case.id)

    def history(self, case_id: str | None = None) -> list[dict[str, Any]]:
        """The full, replayable trail for one case (or every case)."""

        cases = [self.case(case_id)] if case_id is not None else self.cases()
        entries = {entry.sequence: entry for entry in self._store.entries()}
        trails: list[dict[str, Any]] = []
        for case in cases:
            wanted = set(case.read_ids)
            trail: list[dict[str, Any]] = []
            seen_reads: set[str] = set()
            for sequence, entry in sorted(entries.items()):
                # A read row is rewritten when its status changes (frozen,
                # rerouted); show the first commit only, so a reading is not
                # replayed as two distinct observations.
                if entry.kind == "read.put":
                    read_id = entry.payload.get("read", {}).get("id")
                    if read_id in wanted and read_id not in seen_reads:
                        seen_reads.add(read_id)
                        trail.append({"at": entry.at, "event": "reading_committed",
                                      "sequence": sequence, "detail": entry.payload})
                if entry.kind == "case.put" and entry.payload.get("case", {}).get("id") == case.id:
                    stage = self._case_stage(ConflictCase.from_dict(entry.payload["case"]))
                    trail.append({"at": entry.at, "event": stage,
                                  "sequence": sequence, "detail": entry.payload})
            trails.append({"case": case.as_dict(), "trail": trail})
        return trails

    # -- internals: first/duplicate reads ----------------------------------

    def _commit_first(self, state, code: str, flight: str, ordinal: int) -> ReportResult:
        allocation = allocate(state.sequence)
        bag_id = self._ns.bag_id(code)
        at = self._clock.now_iso()
        bag = Bag(
            id=bag_id, barcode=code, sequence=allocation.value,
            epoch=allocation.sequence.epoch, flight_id=flight,
            state=BagState.CHECKED_IN,
        )
        read = ReadRecord(
            id=f"{bag_id}#1", barcode=code, bag_id=bag_id, flight_id=flight,
            ordinal=1, epoch=allocation.sequence.epoch, at=at,
        )
        with self._store.session() as session:
            self._ensure_policy(session.state)
            session.state.sequence = allocation.sequence
            session.state.bags[bag_id] = bag
            session.state.mappings[bag_id] = flight
            session.state.reads.append(read)
            self._record_first(session, allocation, bag, read, at)
        return ReportResult(read=read, outcome="first", bag=bag)

    # -- internals: conflict opening ---------------------------------------

    def _open_conflict(self, state, code, prior: ReadRecord, flight: str, ordinal: int) -> ReportResult:
        allocation = allocate(state.sequence)
        at = self._clock.now_iso()
        case_id = f"CF-{code}-{allocation.sequence.epoch}-{allocation.value}"
        new_read = ReadRecord(
            id=f"{self._ns.bag_id(code)}#{ordinal}", barcode=code, bag_id="",
            flight_id=flight, ordinal=ordinal, epoch=allocation.sequence.epoch,
            case_id=case_id, status=ReadStatus.FROZEN, at=at,
        )
        base = ConflictCase(
            id=case_id, barcode=code, claimed_flights=[prior.flight_id, flight],
            read_ids=[prior.id, new_read.id], bag_ids=[prior.bag_id], reports=1,
            created_at=at, policy=policy.POLICY_FREEZE,
            policy_rationale=policy.POLICY_RATIONALE,
        )
        reroute_evidence = self._find_reroute_evidence(state, code, prior.flight_id, flight)
        if reroute_evidence is not None:
            case, bag = self._settle_reroute(
                state, base, prior, flight, new_read, allocation, reroute_evidence, at
            )
            return ReportResult(read=new_read, outcome="reroute", case=case, bag=bag)

        distinct_evidence = self._find_dual_track(state, code)
        with self._store.session() as session:
            self._ensure_policy(session.state)
            session.state.sequence = allocation.sequence
            self._freeze(session, prior, new_read, base)
            if distinct_evidence is not None:
                case, bag = self._adjudicate_distinct(
                    session, base, new_read, allocation, distinct_evidence, at
                )
                read = next(item for item in session.state.reads if item.id == new_read.id)
                outcome = "distinct"
            else:
                case = base.evolved(
                    state=ConflictState.OPEN,
                    decision=ConflictDecision.INCONCLUSIVE,
                    reason="no trusted evidence distinguishes a reroute from a shared code",
                )
                session.state.cases[case.id] = case
                session.state.review_queue.append(case.id)
                session.record("review.enqueue", {"case_id": case.id}, at)
                session.record("case.put", {"case": case.as_dict()}, at)
                bag = None
                read = new_read
                outcome = "inconclusive"
            self._counter(session, "arbitration.conflict", at)
        return ReportResult(read=read, outcome=outcome, case=case, bag=bag)

    def _attach_report(self, open_case: ConflictCase, code, flight, ordinal: int) -> ReportResult:
        state = self._store.state()
        at = self._clock.now_iso()
        epoch = state.sequence.epoch
        read_id = f"{self._ns.bag_id(code)}#{ordinal}-r{open_case.reports + 1}"
        read = ReadRecord(
            id=read_id, barcode=code, bag_id="", flight_id=flight,
            ordinal=ordinal, epoch=epoch, case_id=open_case.id,
            status=ReadStatus.FROZEN, at=at,
        )
        claimed = list(open_case.claimed_flights)
        if flight not in claimed:
            claimed.append(flight)
        with self._store.session() as session:
            self._ensure_policy(session.state)
            updated = open_case.evolved(reports=open_case.reports + 1, claimed_flights=claimed)
            session.state.cases[updated.id] = updated
            session.state.reads.append(read)
            session.record("read.put", {"read": read.as_dict()}, at)
            session.record("case.put", {"case": updated.as_dict()}, at)
            self._counter(session, "arbitration.rereported", at)
        return ReportResult(read=read, outcome="rereported", case=updated, duplicate=True)

    # -- internals: settlement ---------------------------------------------

    def _settle_reroute(self, state, base, prior, flight, new_read, allocation, evidence_rec, at):
        """One bag changed flight: the second read owns the same bag id."""

        bag_id = prior.bag_id
        old_bag = state.bags[bag_id]
        read = new_read.evolved(bag_id=bag_id, status=ReadStatus.REROUTED)
        updated_bag = old_bag.evolved(flight_id=flight, state=BagState.CHECKED_IN)
        case = base.evolved(
            state=ConflictState.AUTO_ADJUDICATED,
            decision=ConflictDecision.REROUTE,
            evidence=[evidence_rec.as_dict()],
            decided_by=f"auto:{evidence_rec.source or 'dcs'}",
            reason="DCS records one physical bag moving between the two flights",
            bag_ids=[bag_id],
        )
        with self._store.session() as session:
            self._ensure_policy(session.state)
            session.state.sequence = allocation.sequence
            prior_frozen = prior.evolved(status=ReadStatus.REROUTED)
            session.state.reads = [
                item.evolved(status=ReadStatus.REROUTED, case_id=base.id) if item.id == prior.id else item
                for item in session.state.reads
            ]
            session.state.reads.append(read)
            session.state.bags[bag_id] = updated_bag
            session.state.mappings[bag_id] = flight
            session.state.cases[case.id] = case
            session.record("read.put", {"read": prior_frozen.as_dict()}, at)
            session.record("read.put", {"read": read.as_dict()}, at)
            session.record("mapping.put", {"bag_id": bag_id, "flight_id": flight}, at)
            session.record("bag.put", {"bag": updated_bag.as_dict()}, at)
            session.record("case.put", {"case": case.as_dict()}, at)
            self._counter(session, "arbitration.reroute_auto", at)
        return case, updated_bag

    def _freeze(self, session, prior: ReadRecord, new_read: ReadRecord, case: ConflictCase) -> None:
        session.state.frozen_barcodes.add(case.barcode)
        session.state.reads = [
            item.evolved(status=ReadStatus.FROZEN, case_id=case.id)
            if item.id == prior.id else item
            for item in session.state.reads
        ]
        old_bag = session.state.bags[prior.bag_id]
        session.state.bags[prior.bag_id] = old_bag.evolved(state=BagState.RECHECKED)
        session.state.reads.append(new_read)
        session.record("freeze.add", {"barcode": case.barcode}, case.created_at)
        session.record("read.put", {"read": prior.evolved(
            status=ReadStatus.FROZEN, case_id=case.id).as_dict()}, case.created_at)
        session.record("read.put", {"read": new_read.as_dict()}, case.created_at)
        session.record("bag.put", {
            "bag": old_bag.evolved(state=BagState.RECHECKED).as_dict()}, case.created_at)
        session.state.recheck_queue = [
            item for item in session.state.recheck_queue if item != prior.bag_id
        ]
        session.state.recheck_queue.append(prior.bag_id)
        session.record("recheck.enqueue", {"bag_id": prior.bag_id}, case.created_at)

    def _adjudicate_distinct(self, session, base: ConflictCase, new_read, allocation,
                             evidence_rec: Evidence, at: str):
        """Two physical bags share one code: mint a new bag, keep both frozen+queued."""

        second_bag_id = f"{self._ns.bag_id(base.barcode)}-D{base.id.rsplit('-', 1)[-1]}"
        bag = Bag(
            id=second_bag_id, barcode=base.barcode, sequence=allocation.value,
            epoch=allocation.sequence.epoch, flight_id=base.claimed_flights[-1],
            state=BagState.RECHECKED,
        )
        read = new_read.evolved(bag_id=second_bag_id, status=ReadStatus.DISTINCT)
        # Two physical pieces share the code, so the machine cannot label them:
        # the case stays OPEN and frozen until the review desk re-labels one
        # piece. The evidence is recorded, the second bag is minted, and the
        # case waits at the review queue -- it is not silently settled.
        case = base.evolved(
            state=ConflictState.OPEN,
            decision=ConflictDecision.DISTINCT,
            evidence=[evidence_rec.as_dict()],
            decided_by=f"auto:{evidence_rec.source or 'tracking'}",
            reason="two physical tracks observed for one barcode: the code is shared; "
                   "awaiting re-label at the review desk",
            bag_ids=base.bag_ids + [second_bag_id],
            review_id=second_bag_id,
        )
        session.state.bags[second_bag_id] = bag
        session.state.mappings[second_bag_id] = bag.flight_id
        session.state.reads = [
            item.evolved(bag_id=second_bag_id, status=ReadStatus.DISTINCT)
            if item.id == new_read.id else item
            for item in session.state.reads
        ]
        session.state.recheck_queue.append(second_bag_id)
        session.state.review_queue.append(case.id)
        session.state.cases[case.id] = case
        session.record("bag.put", {"bag": bag.as_dict()}, at)
        session.record("mapping.put", {"bag_id": second_bag_id, "flight_id": bag.flight_id}, at)
        session.record("read.put", {"read": read.as_dict()}, at)
        session.record("recheck.enqueue", {"bag_id": second_bag_id}, at)
        session.record("review.enqueue", {"case_id": case.id}, at)
        session.record("case.put", {"case": case.as_dict()}, at)
        self._counter(session, "arbitration.distinct_auto", at)
        return case, bag

    def _apply_reroute(self, state, case: ConflictCase, target_flight, operator, reason, at,
                       manual: bool) -> ConflictCase:
        bag_id = case.bag_ids[0]
        old_bag = state.bags[bag_id]
        updated_bag = old_bag.evolved(flight_id=target_flight, state=BagState.CHECKED_IN)
        settled = case.evolved(
            state=ConflictState.RESOLVED,
            decision=ConflictDecision.REROUTE,
            decided_by=operator,
            reason=reason,
            resolved_at=at,
            claimed_flights=_with(case.claimed_flights, target_flight),
        )
        with self._store.session() as session:
            self._ensure_policy(session.state)
            session.state.frozen_barcodes.discard(case.barcode)
            session.state.mappings[bag_id] = target_flight
            session.state.bags[bag_id] = updated_bag
            new_ledger: list[ReadRecord] = []
            rewritten: list[ReadRecord] = []
            for item in session.state.reads:
                if item.barcode == case.barcode and item.status is ReadStatus.FROZEN:
                    item = item.evolved(
                        status=ReadStatus.REROUTED,
                        case_id=item.case_id or case.id,
                        bag_id=item.bag_id or bag_id,
                    )
                    rewritten.append(item)
                new_ledger.append(item)
            session.state.reads = new_ledger
            session.state.recheck_queue = [
                item for item in session.state.recheck_queue if item not in case.bag_ids
            ]
            session.state.cases[settled.id] = settled
            session.record("freeze.remove", {"barcode": case.barcode}, at)
            session.record("mapping.put", {"bag_id": bag_id, "flight_id": target_flight}, at)
            session.record("bag.put", {"bag": updated_bag.as_dict()}, at)
            for read in rewritten:
                session.record("read.put", {"read": read.as_dict()}, at)
            session.record("case.put", {"case": settled.as_dict()}, at)
            self._counter(
                session,
                "arbitration.reroute_manual" if manual else "arbitration.reroute_auto",
                at,
            )
        return settled

    def _apply_distinct(self, state, case, operator, reason, at, manual: bool) -> ConflictCase:
        """Shared code confirmed at review: mint/restore both pieces after relabel."""

        second_bag_id = next((bid for bid in case.bag_ids if bid != case.bag_ids[0]), "")
        fresh_bag: Bag | None = None
        if not second_bag_id:
            second_bag_id = f"{self._ns.bag_id(case.barcode)}-D{case.id.rsplit('-', 1)[-1]}"
            if second_bag_id not in state.bags:
                fresh_bag = Bag(
                    id=second_bag_id, barcode=case.barcode, sequence=0, epoch=0,
                    flight_id=case.claimed_flights[-1], state=BagState.RECHECKED,
                )
        settled = case.evolved(
            state=ConflictState.RESOLVED,
            decision=ConflictDecision.DISTINCT,
            decided_by=operator,
            reason=reason,
            resolved_at=at,
        )
        with self._store.session() as session:
            self._ensure_policy(session.state)
            session.state.frozen_barcodes.discard(case.barcode)
            if fresh_bag is not None:
                session.state.bags[fresh_bag.id] = fresh_bag
                session.state.mappings[fresh_bag.id] = fresh_bag.flight_id
                session.record("bag.put", {"bag": fresh_bag.as_dict()}, at)
                session.record(
                    "mapping.put",
                    {"bag_id": fresh_bag.id, "flight_id": fresh_bag.flight_id},
                    at,
                )
            new_ledger: list[ReadRecord] = []
            rewritten: list[ReadRecord] = []
            for item in session.state.reads:
                if item.barcode == case.barcode and item.status is ReadStatus.FROZEN:
                    item = item.evolved(
                        status=ReadStatus.DISTINCT,
                        bag_id=item.bag_id or second_bag_id,
                    )
                    rewritten.append(item)
                new_ledger.append(item)
            session.state.reads = new_ledger
            # A human has re-labelled the shared code: both pieces are now
            # distinct physical bags and may resume the line.
            settled_bag_ids = list(dict.fromkeys(case.bag_ids + [second_bag_id]))
            for bag_id in settled_bag_ids:
                held = session.state.bags.get(bag_id)
                if held is not None and held.state is BagState.RECHECKED:
                    restored = held.evolved(state=BagState.CHECKED_IN)
                    session.state.bags[bag_id] = restored
                    session.record("bag.put", {"bag": restored.as_dict()}, at)
            settled = settled.evolved(bag_ids=settled_bag_ids, review_id=second_bag_id)
            session.state.recheck_queue = [
                item for item in session.state.recheck_queue if item not in settled_bag_ids
            ]
            session.state.cases[settled.id] = settled
            session.record("freeze.remove", {"barcode": case.barcode}, at)
            for read in rewritten:
                session.record("read.put", {"read": read.as_dict()}, at)
            session.record("case.put", {"case": settled.as_dict()}, at)
            self._counter(session, "arbitration.distinct_manual", at)
        return settled

    # -- helpers -----------------------------------------------------------

    def _ensure_policy(self, session_state) -> None:
        if not session_state.arbitration_policy:
            session_state.arbitration_policy = policy.POLICY_FREEZE
            session_state.arbitration_policy_rationale = policy.POLICY_RATIONALE
            # Stamp the policy into the journal inside the caller's transaction
            # so a restart rebuilds the rule every ruling was made under.
            self._store.journal.append(
                "arbitration.policy",
                {"policy": policy.POLICY_FREEZE, "rationale": policy.POLICY_RATIONALE},
                self._clock.now_iso(),
            )

    def _require_flight(self, flight_id: str) -> None:
        if flight_id not in self._store.state().flights:
            raise NotFoundError(f"flight {flight_id!r} is not registered")

    def _find_reroute_evidence(self, state, code, old_flight, new_flight) -> Evidence | None:
        for evidence in state.evidence.values():
            if (
                evidence.barcode == code
                and evidence.kind == policy.EVIDENCE_DCS_REROUTE
                and evidence.from_flight_id == old_flight
                and evidence.to_flight_id == new_flight
            ):
                return evidence
        return None

    def _find_dual_track(self, state, code) -> Evidence | None:
        for evidence in state.evidence.values():
            if evidence.barcode == code and evidence.kind == policy.EVIDENCE_DUAL_TRACK:
                return evidence
        return None

    def _record_first(self, session, allocation, bag: Bag, read: ReadRecord, at: str) -> None:
        session.record("sequence.put", allocation.sequence.as_dict(), at)
        session.record("bag.put", {"bag": bag.as_dict()}, at)
        session.record("mapping.put", {"bag_id": bag.id, "flight_id": bag.flight_id}, at)
        session.record("read.put", {"read": read.as_dict()}, at)
        session.record("counter.add", {"name": "arbitration.reads", "delta": 1}, at)

    def _counter(self, session, name: str, at: str) -> None:
        session.record("counter.add", {"name": name, "delta": 1}, at)

    def _case_stage(self, case: ConflictCase) -> str:
        if case.state is ConflictState.RESOLVED:
            return "resolved"
        if case.state is ConflictState.AUTO_ADJUDICATED:
            return "auto_adjudicated"
        return "case_opened"


def _latest_attributed(state, code: str) -> ReadRecord | None:
    candidates = [
        read for read in state.reads
        if read.barcode == code and read.status is not ReadStatus.FROZEN and read.bag_id
    ]
    return candidates[-1] if candidates else None


def _open_case(state, code: str) -> ConflictCase | None:
    for case in state.cases.values():
        if case.barcode == code and case.state is ConflictState.OPEN:
            return case
    return None


def _with(items: list[str], value: str) -> list[str]:
    return items if value in items else items + [value]
