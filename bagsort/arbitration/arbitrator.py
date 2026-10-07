"""Barcode conflict arbitration.

A physical barcode can be read twice for two different flights for two very
different reasons: the same piece was officially re-flighted, or two physical
pieces are sharing one (misprinted) code. This subsystem is the single owner
of that decision.

Every check-in goes through :meth:`Arbitrator.check_in`, which commits one
immutable :class:`ReadClaim` per read — the read is never silently overwritten.
When two reads of one barcode point at different flights a :class:`ConflictCase`
opens. Evidence that settles the question (an official re-flight directive, or
distinct weights proving two pieces share the code) is applied automatically;
anything the evidence cannot settle freezes the barcode and escalates to the
recheck station. Nothing with an unresolved case can be sorted.

The policy taken on an undecidable conflict is chosen once,
:data:`~bagsort.arbitration.entities.POLICY_CODE`, and the rejected alternative
with its opposite risk profile is recorded inside every decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bagsort.clock import Clock
from bagsort.errors import ConflictError, NotFoundError, OrderingError, ValidationError
from bagsort.model.entities import Bag
from bagsort.model.enums import BagState
from bagsort.ns.namespace import Namespace
from bagsort.sequence.allocator import allocate
from bagsort.store.store import Store
from bagsort.arbitration.entities import (
    ConflictCase,
    DecisionRecord,
    ReadClaim,
    ReflightDirective,
    policy_reason,
)
from bagsort.arbitration.enums import CaseStatus, ClaimStatus, Verdict

# Two reads whose stated weights differ by at least this much cannot be the
# same physical piece, so the code is shared by two pieces.
WEIGHT_EPSILON = 0.5

CASE_COUNTER = "arbitration.case"
DECISION_COUNTER = "arbitration.decision"


@dataclass(frozen=True)
class CheckInOutcome:
    """What one check-in produced, including any conflict adjudication."""

    bag: Bag
    duplicate: bool
    case_id: str
    decision_id: str
    escalated: bool
    verdict: str

    @property
    def settled(self) -> bool:
        """Whether the bag may enter the sorting line right now."""

        return not self.case_id


class Arbitrator:
    """Commits reads, detects barcode conflicts, adjudicates and escalates."""

    def __init__(
        self,
        store: Store,
        namespace: Namespace,
        clock: Clock,
        quota: object,
        injector: object,
        weight_epsilon: float = WEIGHT_EPSILON,
    ) -> None:
        self._store = store
        self._ns = namespace
        self._clock = clock
        self._quota = quota
        self._injector = injector
        self._epsilon = weight_epsilon

    # ------------------------------------------------------------------
    # evidence ledger
    # ------------------------------------------------------------------

    def record_reflight(self, barcode: str, from_flight_id: str, to_flight_id: str) -> ReflightDirective:
        """Register official evidence that a barcode changed flights."""

        text = (barcode or "").strip()
        if not text:
            raise ValidationError("barcode must not be empty")
        if not from_flight_id or not to_flight_id or from_flight_id == to_flight_id:
            raise ValidationError("a re-flight directive needs two different flights")
        directive = ReflightDirective(
            barcode=text,
            from_flight_id=from_flight_id,
            to_flight_id=to_flight_id,
            recorded_at=self._clock.now_iso(),
        )
        key = (text, from_flight_id, to_flight_id)
        existing = self._store.state().reflight_directives.get(key)
        if existing is not None:
            return existing
        with self._store.session() as session:
            session.state.reflight_directives[key] = directive
            session.record("reflight.record", {"directive": directive.as_dict()}, directive.recorded_at)
        return directive

    # ------------------------------------------------------------------
    # read submission and adjudication
    # ------------------------------------------------------------------

    def check_in(
        self, barcode: str, flight_id: str, weight: float = 0.0
    ) -> CheckInOutcome:
        """Submit one barcode read and settle any conflict it creates.

        The read, the bag, the mapping, the case and the recheck escalation all
        commit in one transaction, so the read record and the bag-to-flight
        mapping can never describe different worlds. Repeating the same
        submission returns the existing result without writing a second claim.
        """

        text = (barcode or "").strip()
        if not text:
            raise ValidationError("barcode must not be empty")
        state = self._store.state()
        if flight_id not in state.flights:
            raise NotFoundError(f"flight {flight_id!r} is not registered")

        duplicate = self._find_claim(text, flight_id)
        if duplicate is not None:
            return self._outcome_for(duplicate, duplicate=True)

        prior = self._prior_claim(text)
        at = self._clock.now_iso()

        if prior is None:
            bag = self._commit_plain(text, flight_id, weight, at)
            claim = self._store.state().read_claims[self._claim_id(bag.id)]
            return CheckInOutcome(bag, False, "", "", False, "")

        if self._is_frozen(text) and not self._directive_covers(
            prior.flight_id, flight_id, text
        ):
            raise ConflictError(
                f"barcode {text!r} is frozen pending manual review; "
                "resolve it at the recheck station"
            )

        allocation = allocate(state.sequence)
        occurrence = len([c for c in state.read_claims.values() if c.barcode == text]) + 1
        bag_id = f"{self._ns.bag_id(text)}#{occurrence}"
        bag = Bag(
            id=bag_id,
            barcode=text,
            sequence=allocation.value,
            epoch=allocation.sequence.epoch,
            flight_id=flight_id,
            state=BagState.CHECKED_IN,
            weight=weight,
        )
        claim = ReadClaim(
            id=self._claim_id(bag_id),
            barcode=text,
            bag_id=bag_id,
            flight_id=flight_id,
            sequence=allocation.value,
            epoch=allocation.sequence.epoch,
            weight=weight,
            status=ClaimStatus.PENDING,
            at=at,
        )

        evidence = self._evaluate(text, prior, claim)
        case = self._open_case(prior, claim, at, evidence)
        if evidence["verdict"] == Verdict.SAME_BAG_REFLIGHT.value:
            decision, owner_bag = self._auto_reflight(case, prior, claim, bag, evidence, at, allocation)
            return CheckInOutcome(
                owner_bag,
                False,
                case.id,
                decision.id,
                False,
                Verdict.SAME_BAG_REFLIGHT.value,
            )
        # Shared barcode and indeterminate both freeze and escalate.
        return self._escalate(case, prior, claim, bag, evidence, at, allocation)

    # ------------------------------------------------------------------
    # manual resolution at the recheck station
    # ------------------------------------------------------------------

    def resolve_case(
        self,
        case_id: str,
        verdict: str,
        decided_by: str,
        note: str = "",
    ) -> DecisionRecord:
        """Apply a human adjudication to an open conflict case.

        Idempotent: resolving a case that already carries this verdict returns
        the stored decision instead of writing a second one.
        """

        wanted = self._require_verdict(verdict)
        state = self._store.state()
        case = state.conflict_cases.get(case_id)
        if case is None:
            raise NotFoundError(f"conflict case {case_id!r} does not exist")
        if case.status is CaseStatus.RESOLVED:
            existing = state.decisions[case.decision_id]
            if existing.verdict is wanted:
                return existing
            raise OrderingError(f"case {case_id!r} is already resolved")
        if wanted is Verdict.INDETERMINATE:
            raise ValidationError("manual resolution must pick same_bag_reflight or shared_barcode")

        prior = state.read_claims[case.prior_claim_id]
        latter = state.read_claims[case.latter_claim_id]
        at = self._clock.now_iso()
        decision = self._decision(
            case, prior, latter, wanted, decided_by, "manual_review", case.evidence, at, note
        )

        with self._store.session() as session:
            if wanted is Verdict.SAME_BAG_REFLIGHT:
                prior_bag = self._mark_superseded(session, prior)
                latter_bag = self._rebuild_owner(session, latter)
                session.state.bags[prior_bag.id] = prior_bag
                session.state.bags[latter_bag.id] = latter_bag
            else:
                prior_bag = self._rebuild_shared(session, prior)
                latter_bag = self._rebuild_shared(session, latter)
                session.state.bags[prior_bag.id] = prior_bag
                session.state.bags[latter_bag.id] = latter_bag
            resolved = case.evolved(
                status=CaseStatus.RESOLVED,
                decision_id=decision.id,
                resolved_at=at,
                evidence={**case.evidence, "manual_note": note},
            )
            session.state.conflict_cases[case.id] = resolved
            session.state.decisions[decision.id] = decision
            self._unfreeze(session, case.barcode)
            if case.recheck_item_id:
                queue = [item for item in session.state.conflict_recheck_queue if item != case.recheck_item_id]
                session.state.conflict_recheck_queue = queue
                session.record(
                    "conflict_recheck.dequeue",
                    {"item_id": case.recheck_item_id, "case_id": case.id},
                    at,
                )
            session.record("case.put", {"case": resolved.as_dict()}, at)
            session.record("decision.put", {"decision": decision.as_dict()}, at)
            session.record("counter.add", {"name": "arbitration.manually_resolved", "delta": 1}, at)
        return decision

    def report_conflict(self, prior_claim_id: str, latter_claim_id: str) -> ConflictCase:
        """Idempotently (re)report a conflict between two committed reads.

        The same pair reported twice yields the same case and at most one
        decision and one escalation. If the pair agrees on the flight there is
        no conflict and the existing case, if any, is returned.
        """

        state = self._store.state()
        prior = state.read_claims.get(prior_claim_id)
        latter = state.read_claims.get(latter_claim_id)
        if prior is None or latter is None:
            raise NotFoundError("both reads must be committed before reporting a conflict")
        if prior.barcode != latter.barcode:
            raise ValidationError("a conflict needs two reads of the same barcode")
        existing = self._case_for(prior.id, latter.id)
        if existing is not None:
            return existing
        if prior.flight_id == latter.flight_id:
            raise ConflictError("the two reads belong to the same flight; nothing to arbitrate")
        at = self._clock.now_iso()
        evidence = self._evaluate(prior.barcode, prior, latter)
        case = self._open_case(prior, latter, at, evidence)
        if evidence["verdict"] == Verdict.SAME_BAG_REFLIGHT.value:
            self._auto_reflight_from_claims(case, prior, latter, evidence, at)
        else:
            self._escalate_existing(case, prior, latter, evidence, at)
        return self._store.state().conflict_cases[case.id]

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    def claims(self, barcode: str = "") -> list[ReadClaim]:
        claims = self._store.state().read_claims.values()
        if barcode:
            claims = [claim for claim in claims if claim.barcode == barcode]
        return sorted(claims, key=lambda claim: (claim.barcode, claim.sequence, claim.id))

    def claim(self, claim_id: str) -> ReadClaim:
        claim = self._store.state().read_claims.get(claim_id)
        if claim is None:
            raise NotFoundError(f"read {claim_id!r} is not committed")
        return claim

    def cases(self, open_only: bool = False) -> list[ConflictCase]:
        cases = self._store.state().conflict_cases.values()
        if open_only:
            cases = [case for case in cases if case.is_open]
        return sorted(cases, key=lambda case: case.detected_at + case.id)

    def case(self, case_id: str) -> ConflictCase:
        case = self._store.state().conflict_cases.get(case_id)
        if case is None:
            raise NotFoundError(f"conflict case {case_id!r} does not exist")
        return case

    def decisions(self) -> list[DecisionRecord]:
        return sorted(self._store.state().decisions.values(), key=lambda item: item.at + item.id)

    def decision_for(self, case_id: str) -> DecisionRecord | None:
        case = self._store.state().conflict_cases.get(case_id)
        if case is None or not case.decision_id:
            return None
        return self._store.state().decisions[case.decision_id]

    def history(self, barcode: str) -> list[dict[str, Any]]:
        """Every read, case, escalation and decision for a barcode, in order.

        Rows are ordered by timestamp first; rows sharing one instant are
        ordered by their place in the handling line (read, conflict
        detection, escalation, decision), which keeps the timeline legible on
        the fixed test/demo clock.
        """

        state = self._store.state()
        order = {
            "read": 0,
            "conflict_detected": 1,
            "escalated": 2,
            "decision": 3,
        }
        timeline: list[dict[str, Any]] = []
        for claim in self.claims(barcode):
            timeline.append({"kind": "read", "at": claim.at, "payload": claim.as_dict()})
        for case in state.conflict_cases.values():
            if case.barcode != barcode:
                continue
            timeline.append({"kind": "conflict_detected", "at": case.detected_at, "payload": case.as_dict()})
            if case.escalated_at:
                timeline.append(
                    {"kind": "escalated", "at": case.escalated_at, "payload": case.as_dict()}
                )
            if case.decision_id:
                decision = state.decisions[case.decision_id]
                timeline.append({"kind": "decision", "at": decision.at, "payload": decision.as_dict()})
        return sorted(timeline, key=lambda row: (row["at"], order.get(row["kind"], 9)))

    def frozen(self) -> list[str]:
        return list(self._store.state().frozen_barcodes)

    def is_frozen(self, barcode: str) -> bool:
        return barcode in self._store.state().frozen_barcodes

    def recheck_items(self) -> list[ConflictCase]:
        """Open escalations in FIFO order, one queue item per case."""

        state = self._store.state()
        return [
            state.conflict_cases[case_id]
            for case_id in state.conflict_recheck_queue
            if case_id in state.conflict_cases and state.conflict_cases[case_id].is_open
        ]

    def next_recheck(self) -> ConflictCase | None:
        items = self.recheck_items()
        return items[0] if items else None

    def consistency(self) -> dict[str, Any]:
        """Cross-check reads, mappings and queues agree; used by tests and health."""

        state = self._store.state()
        problems: list[str] = []
        for claim in state.read_claims.values():
            if claim.status is ClaimStatus.OWNER and claim.bag_id not in state.mappings:
                problems.append(f"owner read {claim.id} has no mapping")
            if claim.status is ClaimStatus.SHARED and claim.bag_id not in state.mappings:
                problems.append(f"shared read {claim.id} has no mapping")
        for case in state.conflict_cases.values():
            if case.recheck_item_id:
                queued = case.recheck_item_id in state.conflict_recheck_queue
                if case.is_open != queued:
                    problems.append(f"case {case.id} queue state disagrees with case state")
            if case.status is CaseStatus.RESOLVED:
                if not case.decision_id or case.decision_id not in state.decisions:
                    problems.append(f"case {case.id} resolved without a decision")
        for barcode in state.frozen_barcodes:
            open_for = [case for case in state.conflict_cases.values() if case.barcode == barcode and case.is_open]
            if not open_for:
                problems.append(f"barcode {barcode!r} frozen without an open case")
        return {"consistent": not problems, "problems": problems}

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _claim_id(self, bag_id: str) -> str:
        return f"{bag_id}:read"

    def _find_claim(self, barcode: str, flight_id: str) -> ReadClaim | None:
        for claim in self.claims(barcode):
            if claim.flight_id == flight_id:
                return claim
        return None

    def _prior_claim(self, barcode: str) -> ReadClaim | None:
        """The most recent committed read of a barcode, whatever its status.

        SHARED and SUPERSEDED reads still prove the code has been seen, so a
        further read must be adjudicated instead of treated as a first read.
        """

        claims = self.claims(barcode)
        return claims[-1] if claims else None

    def _is_frozen(self, barcode: str) -> bool:
        return barcode in self._store.state().frozen_barcodes

    def _directive_covers(self, from_flight_id: str, to_flight_id: str, barcode: str) -> bool:
        return (barcode, from_flight_id, to_flight_id) in self._store.state().reflight_directives

    def _evaluate(
        self, barcode: str, prior: ReadClaim, latter: ReadClaim
    ) -> dict[str, Any]:
        """Classify a conflict from the available evidence."""

        directive = self._store.state().reflight_directives.get(
            (barcode, prior.flight_id, latter.flight_id)
        )
        if directive is not None:
            return {
                "verdict": Verdict.SAME_BAG_REFLIGHT.value,
                "basis": "reflight_directive",
                "directive": directive.as_dict(),
                "weight_delta": round(abs(latter.weight - prior.weight), 3),
            }
        delta = round(abs(latter.weight - prior.weight), 3)
        if prior.weight and latter.weight and delta >= self._epsilon:
            return {
                "verdict": Verdict.SHARED_BARCODE.value,
                "basis": "weight_mismatch",
                "weight_delta": delta,
            }
        return {
            "verdict": Verdict.INDETERMINATE.value,
            "basis": "no_conclusive_evidence",
            "weight_delta": delta,
        }

    def _next_counter(self, state, name: str) -> int:
        return int(state.counters.get(name, 0)) + 1

    def _open_case(
        self, prior: ReadClaim, latter: ReadClaim, at: str, evidence: dict[str, Any]
    ) -> ConflictCase:
        state = self._store.state()
        existing = self._case_for(prior.id, latter.id)
        if existing is not None:
            return existing
        number = self._next_counter(state, CASE_COUNTER)
        case = ConflictCase(
            id=f"conf-{number:06d}",
            barcode=prior.barcode,
            prior_claim_id=prior.id,
            latter_claim_id=latter.id,
            status=CaseStatus.OPEN,
            detected_at=at,
            escalated_at="",
            recheck_item_id="",
            evidence=evidence,
        )
        with self._store.session() as session:
            session.state.conflict_cases[case.id] = case
            session.state.counters[CASE_COUNTER] = number
            session.record("case.put", {"case": case.as_dict()}, at)
            session.record("counter.add", {"name": CASE_COUNTER, "delta": 1}, at)
        return case

    def _case_for(self, prior_claim_id: str, latter_claim_id: str) -> ConflictCase | None:
        for case in self._store.state().conflict_cases.values():
            if case.prior_claim_id == prior_claim_id and case.latter_claim_id == latter_claim_id:
                return case
        return None

    def _commit_plain(
        self, barcode: str, flight_id: str, weight: float, at: str
    ) -> Bag:
        state = self._store.state()
        allocation = allocate(state.sequence)
        bag_id = self._ns.bag_id(barcode)
        if bag_id in state.bags:
            raise ConflictError(
                f"bag {bag_id!r} already exists from a commit that predates arbitration; "
                "it must be adjudicated before another read is accepted"
            )
        bag = Bag(
            id=bag_id,
            barcode=barcode,
            sequence=allocation.value,
            epoch=allocation.sequence.epoch,
            flight_id=flight_id,
            state=BagState.CHECKED_IN,
            weight=weight,
        )
        claim = ReadClaim(
            id=self._claim_id(bag_id),
            barcode=barcode,
            bag_id=bag_id,
            flight_id=flight_id,
            sequence=allocation.value,
            epoch=allocation.sequence.epoch,
            weight=weight,
            status=ClaimStatus.OWNER,
            at=at,
        )
        with self._store.session() as session:
            session.state.sequence = allocation.sequence
            session.state.bags[bag_id] = bag
            session.state.mappings[bag_id] = flight_id
            session.state.read_claims[claim.id] = claim
            session.record("sequence.put", allocation.sequence.as_dict(), at)
            session.record("bag.put", {"bag": bag.as_dict()}, at)
            session.record("mapping.put", {"bag_id": bag_id, "flight_id": flight_id}, at)
            session.record("claim.put", {"claim": claim.as_dict()}, at)
            session.record("counter.add", {"name": "tag.committed", "delta": 1}, at)
            session.record("counter.add", {"name": "arbitration.reads_committed", "delta": 1}, at)
            if self._quota.is_limited(flight_id):
                self._quota.stage_consume(session, flight_id)
        committed = self._store.state().bags[bag_id]
        with self._store.session() as session:
            injected = self._injector.stage_accept(session, committed)
        return injected

    def _auto_reflight(
        self,
        case: ConflictCase,
        prior: ReadClaim,
        latter: ReadClaim,
        latter_bag: Bag,
        evidence: dict[str, Any],
        at: str,
        allocation,
    ) -> tuple[DecisionRecord, Bag]:
        decision_number = self._next_counter(self._store.state(), DECISION_COUNTER)
        decision = self._decision_from_number(
            decision_number, case, prior, latter,
            Verdict.SAME_BAG_REFLIGHT, "arbitrator", evidence["basis"], evidence, at,
        )
        prior_bag = self._store.state().bags[prior.bag_id]
        prior_diverted = prior_bag.evolved(state=BagState.DIVERTED)
        owner_bag = latter_bag.evolved(state=BagState.INJECTED, flight_id=latter.flight_id)
        prior_claim = prior.evolved(status=ClaimStatus.SUPERSEDED)
        latter_claim = latter.evolved(status=ClaimStatus.OWNER)
        resolved = case.evolved(
            status=CaseStatus.RESOLVED, decision_id=decision.id, resolved_at=at
        )
        with self._store.session() as session:
            session.state.sequence = allocation.sequence
            session.state.bags[prior_diverted.id] = prior_diverted
            session.state.bags[owner_bag.id] = owner_bag
            session.state.read_claims[prior_claim.id] = prior_claim
            session.state.read_claims[latter_claim.id] = latter_claim
            session.state.mappings.pop(prior.bag_id, None)
            session.state.mappings[owner_bag.id] = latter.flight_id
            session.state.conflict_cases[case.id] = resolved
            session.state.decisions[decision.id] = decision
            session.record("sequence.put", allocation.sequence.as_dict(), at)
            session.record("bag.put", {"bag": prior_diverted.as_dict()}, at)
            session.record("bag.put", {"bag": owner_bag.as_dict()}, at)
            session.record("claim.put", {"claim": prior_claim.as_dict()}, at)
            session.record("claim.put", {"claim": latter_claim.as_dict()}, at)
            session.record("mapping.remove", {"bag_id": prior.bag_id}, at)
            session.record("mapping.put", {"bag_id": owner_bag.id, "flight_id": latter.flight_id}, at)
            session.record("case.put", {"case": resolved.as_dict()}, at)
            session.record("decision.put", {"decision": decision.as_dict()}, at)
            session.record("counter.add", {"name": "tag.committed", "delta": 1}, at)
            session.record("counter.add", {"name": "arbitration.auto_reflights", "delta": 1}, at)
            session.record("counter.add", {"name": DECISION_COUNTER, "delta": 1}, at)
            if self._quota.is_limited(latter.flight_id):
                self._quota.stage_consume(session, latter.flight_id)
        return decision, owner_bag

    def _auto_reflight_from_claims(
        self, case, prior, latter, evidence, at
    ) -> DecisionRecord:
        decision_number = self._next_counter(self._store.state(), DECISION_COUNTER)
        decision = self._decision_from_number(
            decision_number, case, prior, latter,
            Verdict.SAME_BAG_REFLIGHT, "arbitrator", evidence["basis"], evidence, at,
        )
        resolved = case.evolved(
            status=CaseStatus.RESOLVED, decision_id=decision.id, resolved_at=at
        )
        prior_bag = self._store.state().bags[prior.bag_id]
        latter_bag = self._store.state().bags[latter.bag_id]
        prior_diverted = prior_bag.evolved(state=BagState.DIVERTED)
        owner = latter_bag.evolved(state=BagState.INJECTED, flight_id=latter.flight_id)
        prior_claim = prior.evolved(status=ClaimStatus.SUPERSEDED)
        latter_claim = latter.evolved(status=ClaimStatus.OWNER)
        with self._store.session() as session:
            session.state.bags[prior_diverted.id] = prior_diverted
            session.state.bags[owner.id] = owner
            session.state.read_claims[prior_claim.id] = prior_claim
            session.state.read_claims[latter_claim.id] = latter_claim
            session.state.mappings.pop(prior.bag_id, None)
            session.state.mappings[owner.id] = latter.flight_id
            session.state.conflict_cases[case.id] = resolved
            session.state.decisions[decision.id] = decision
            session.record("bag.put", {"bag": prior_diverted.as_dict()}, at)
            session.record("bag.put", {"bag": owner.as_dict()}, at)
            session.record("claim.put", {"claim": prior_claim.as_dict()}, at)
            session.record("claim.put", {"claim": latter_claim.as_dict()}, at)
            session.record("mapping.remove", {"bag_id": prior.bag_id}, at)
            session.record("mapping.put", {"bag_id": owner.id, "flight_id": latter.flight_id}, at)
            session.record("case.put", {"case": resolved.as_dict()}, at)
            session.record("decision.put", {"decision": decision.as_dict()}, at)
            session.record("counter.add", {"name": "arbitration.auto_reflights", "delta": 1}, at)
            session.record("counter.add", {"name": DECISION_COUNTER, "delta": 1}, at)
        return decision

    def _escalate(
        self, case, prior, latter, latter_bag, evidence, at, allocation
    ) -> CheckInOutcome:
        pending_bag = latter_bag
        escalated = case.evolved(escalated_at=at, recheck_item_id=case.id)
        with self._store.session() as session:
            session.state.sequence = allocation.sequence
            session.state.bags[pending_bag.id] = pending_bag
            session.state.read_claims[latter.id] = latter
            session.state.mappings[pending_bag.id] = latter.flight_id
            session.state.conflict_cases[case.id] = escalated
            session.state.frozen_barcodes = [
                item for item in session.state.frozen_barcodes if item != case.barcode
            ] + [case.barcode]
            if case.id not in session.state.conflict_recheck_queue:
                session.state.conflict_recheck_queue.append(case.id)
            session.record("sequence.put", allocation.sequence.as_dict(), at)
            session.record("bag.put", {"bag": pending_bag.as_dict()}, at)
            session.record("claim.put", {"claim": latter.as_dict()}, at)
            session.record("mapping.put", {"bag_id": pending_bag.id, "flight_id": latter.flight_id}, at)
            session.record("case.put", {"case": escalated.as_dict()}, at)
            session.record("barcode.freeze", {"barcode": case.barcode}, at)
            session.record(
                "conflict_recheck.enqueue",
                {"item_id": case.id, "case_id": case.id, "barcode": case.barcode},
                at,
            )
            session.record("counter.add", {"name": "tag.committed", "delta": 1}, at)
            session.record("counter.add", {"name": "arbitration.escalations", "delta": 1}, at)
            if self._quota.is_limited(latter.flight_id):
                self._quota.stage_consume(session, latter.flight_id)
        return CheckInOutcome(self._store.state().bags[pending_bag.id], False, case.id, "", True, evidence["verdict"])

    def _escalate_existing(self, case, prior, latter, evidence, at) -> None:
        escalated = case.evolved(escalated_at=at, recheck_item_id=case.id, evidence=evidence)
        with self._store.session() as session:
            session.state.conflict_cases[case.id] = escalated
            if case.barcode not in session.state.frozen_barcodes:
                session.state.frozen_barcodes.append(case.barcode)
            if case.id not in session.state.conflict_recheck_queue:
                session.state.conflict_recheck_queue.append(case.id)
            session.record("case.put", {"case": escalated.as_dict()}, at)
            session.record("barcode.freeze", {"barcode": case.barcode}, at)
            session.record(
                "conflict_recheck.enqueue",
                {"item_id": case.id, "case_id": case.id, "barcode": case.barcode},
                at,
            )
            session.record("counter.add", {"name": "arbitration.escalations", "delta": 1}, at)

    # -- resolution helpers ---------------------------------------------

    def _mark_superseded(self, session, claim: ReadClaim) -> Bag:
        updated = claim.evolved(status=ClaimStatus.SUPERSEDED)
        session.state.read_claims[claim.id] = updated
        session.record("claim.put", {"claim": updated.as_dict()}, self._clock.now_iso())
        session.state.mappings.pop(claim.bag_id, None)
        session.record("mapping.remove", {"bag_id": claim.bag_id}, self._clock.now_iso())
        bag = session.state.bags[claim.bag_id].evolved(state=BagState.DIVERTED)
        session.state.bags[claim.bag_id] = bag
        session.record("bag.put", {"bag": bag.as_dict()}, self._clock.now_iso())
        return bag

    def _rebuild_owner(self, session, claim: ReadClaim) -> Bag:
        updated = claim.evolved(status=ClaimStatus.OWNER)
        session.state.read_claims[claim.id] = updated
        session.record("claim.put", {"claim": updated.as_dict()}, self._clock.now_iso())
        session.state.mappings[claim.bag_id] = claim.flight_id
        session.record(
            "mapping.put", {"bag_id": claim.bag_id, "flight_id": claim.flight_id}, self._clock.now_iso()
        )
        bag = session.state.bags[claim.bag_id].evolved(
            state=BagState.INJECTED, flight_id=claim.flight_id
        )
        session.state.bags[claim.bag_id] = bag
        session.record("bag.put", {"bag": bag.as_dict()}, self._clock.now_iso())
        return bag

    def _rebuild_shared(self, session, claim: ReadClaim) -> Bag:
        updated = claim.evolved(status=ClaimStatus.SHARED)
        session.state.read_claims[claim.id] = updated
        session.record("claim.put", {"claim": updated.as_dict()}, self._clock.now_iso())
        session.state.mappings[claim.bag_id] = claim.flight_id
        session.record(
            "mapping.put", {"bag_id": claim.bag_id, "flight_id": claim.flight_id}, self._clock.now_iso()
        )
        bag = session.state.bags[claim.bag_id].evolved(
            state=BagState.INJECTED, flight_id=claim.flight_id
        )
        session.state.bags[claim.bag_id] = bag
        session.record("bag.put", {"bag": bag.as_dict()}, self._clock.now_iso())
        return bag

    def _unfreeze(self, session, barcode: str) -> None:
        session.state.frozen_barcodes = [
            item for item in session.state.frozen_barcodes if item != barcode
        ]
        session.record("barcode.unfreeze", {"barcode": barcode}, self._clock.now_iso())

    def _require_verdict(self, verdict: str) -> Verdict:
        try:
            return Verdict(verdict)
        except ValueError as exc:
            raise ValidationError(
                f"verdict must be one of {[item.value for item in Verdict]}"
            ) from exc

    def _decision(
        self, case, prior, latter, verdict, decided_by, basis, evidence, at, note
    ) -> DecisionRecord:
        number = self._next_counter(self._store.state(), DECISION_COUNTER)
        decision = self._decision_from_number(
            number, case, prior, latter, verdict, decided_by, basis,
            {**evidence, "manual_note": note}, at,
        )
        return decision

    def _decision_from_number(
        self, number, case, prior, latter, verdict, decided_by, basis, evidence, at
    ) -> DecisionRecord:
        return DecisionRecord(
            id=f"dec-{number:06d}",
            case_id=case.id,
            barcode=case.barcode,
            verdict=verdict,
            decided_by=decided_by,
            basis=basis,
            evidence=evidence,
            policy=policy_reason(),
            prior_claim_id=prior.id,
            latter_claim_id=latter.id,
            prior_bag_id=prior.bag_id,
            latter_bag_id=latter.bag_id,
            prior_flight_id=prior.flight_id,
            latter_flight_id=latter.flight_id,
            at=at,
        )

    def _outcome_for(self, claim: ReadClaim, duplicate: bool) -> CheckInOutcome:
        state = self._store.state()
        bag = state.bags[claim.bag_id]
        cases = [
            case
            for case in state.conflict_cases.values()
            if case.latter_claim_id == claim.id or case.prior_claim_id == claim.id
        ]
        case = cases[-1] if cases else None
        decision_id = case.decision_id if case is not None else ""
        escalated = case.is_open and bool(case.recheck_item_id) if case is not None else False
        verdict = ""
        if case is not None and decision_id:
            verdict = state.decisions[decision_id].verdict.value
        return CheckInOutcome(bag, duplicate, case.id if case else "", decision_id, escalated, verdict)
