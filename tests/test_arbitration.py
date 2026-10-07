"""Barcode conflict arbitration: reads, conflicts, decisions, gate, audit."""

from __future__ import annotations

import pytest

from bagsort.arbitration.entities import POLICY_CODE
from bagsort.arbitration.enums import CaseStatus, ClaimStatus, PolicyChoice, Verdict
from bagsort.errors import ConflictError, OrderingError, ValidationError
from bagsort.model.enums import BagState
from bagsort.model.state import fold
from bagsort.store.replay import replay
from bagsort.store.store import Store
from tests.support import make_plant, register_flight


def _two_flights(plant):
    first = register_flight(plant, code="CA111", chute="CHA")
    second = register_flight(plant, code="CA222", chute="CHB")
    return first, second


# ---------------------------------------------------------------------------
# read submission and the basic mapping invariant
# ---------------------------------------------------------------------------


def test_first_read_creates_an_owner_claim_and_mapping(plant, flight):
    outcome = plant.check_in_adjudicated("BAG0001", flight.id)
    assert not outcome.case_id and not outcome.escalated
    claim = plant.arbitrator.claims("BAG0001")[0]
    assert claim.status is ClaimStatus.OWNER
    assert claim.bag_id == outcome.bag.id
    assert plant.mapping.flight_of(outcome.bag.id) == flight.id
    assert plant.arbitration_consistency()["consistent"]


def test_two_reads_never_overwrite_each_other(plant):
    first, second = _two_flights(plant)
    one = plant.check_in("BAG0007", first.id)
    outcome = plant.check_in_adjudicated("BAG0007", second.id)
    # the earlier piece still exists and still points at its own flight
    assert plant.bag(one.id).flight_id == first.id
    assert plant.mapping.flight_of(one.id) == first.id
    assert outcome.bag.id != one.id
    bags = {bag.barcode: bag for bag in plant.store.state().bags.values()}
    assert len(plant.arbitrator.claims("BAG0007")) == 2


def test_same_barcode_same_flight_is_an_idempotent_repeat(plant, flight):
    first = plant.check_in_adjudicated("DUP", flight.id)
    second = plant.check_in_adjudicated("DUP", flight.id)
    assert second.duplicate is True
    assert second.bag.id == first.bag.id
    assert len(plant.arbitrator.claims("DUP")) == 1
    assert plant.store.state().counters.get("tag.committed", 0) == 1


# ---------------------------------------------------------------------------
# evidence: official re-flight directive -> same bag, automatic
# ---------------------------------------------------------------------------


def test_reflight_directive_adjudicates_same_bag_automatically(plant):
    first, second = _two_flights(plant)
    plant.record_reflight("RX", first.id, second.id)
    prior = plant.check_in("RX", first.id)
    outcome = plant.check_in_adjudicated("RX", second.id, weight=14.0)
    assert outcome.verdict == Verdict.SAME_BAG_REFLIGHT.value
    assert outcome.decision_id and not outcome.escalated
    decision = plant.arbitrator.decision_for(outcome.case_id)
    assert decision.basis == "reflight_directive"
    assert decision.decided_by == "arbitrator"
    # the prior read is superseded, its mapping dropped, the piece diverted
    prior_claim = plant.arbitrator.claims("RX")[0]
    assert prior_claim.status is ClaimStatus.SUPERSEDED
    assert plant.bag(prior.id).state is BagState.DIVERTED
    # the new read owns the mapping and may be sorted to the later flight
    assert plant.mapping.flight_of(outcome.bag.id) == second.id
    assert plant.is_sortable(outcome.bag.id)
    assert plant.arbitrator.frozen() == []
    record = plant.push_bag(outcome.bag.id, "CHB")
    assert record.flight_id == second.id
    assert plant.arbitration_consistency()["consistent"]


def test_recording_a_directive_twice_is_idempotent(plant):
    first, second = _two_flights(plant)
    a = plant.record_reflight("RX2", first.id, second.id)
    b = plant.record_reflight("RX2", first.id, second.id)
    assert a == b
    assert len(plant.store.state().reflight_directives) == 1


def test_repeated_adjudicated_checkin_does_not_decide_twice(plant):
    first, second = _two_flights(plant)
    plant.record_reflight("RX3", first.id, second.id)
    plant.check_in("RX3", first.id)
    one = plant.check_in_adjudicated("RX3", second.id)
    two = plant.check_in_adjudicated("RX3", second.id)
    assert one.decision_id == two.decision_id
    assert len(plant.arbitrator.decisions()) == 1
    assert len(plant.arbitrator.cases()) == 1


# ---------------------------------------------------------------------------
# evidence: distinct weights -> two pieces share the code
# ---------------------------------------------------------------------------


def test_weight_mismatch_is_judged_shared_but_still_escalated(plant):
    first, second = _two_flights(plant)
    plant.check_in("WM", first.id, weight=10.0)
    outcome = plant.check_in_adjudicated("WM", second.id, weight=25.0)
    assert outcome.verdict == Verdict.SHARED_BARCODE.value
    assert outcome.escalated
    case = plant.arbitrator.case(outcome.case_id)
    assert case.evidence["basis"] == "weight_mismatch"
    # even with a verdict, ownership is not auto-asserted; both wait for review
    assert plant.arbitrator.is_frozen("WM")
    assert not plant.is_sortable(outcome.bag.id)


# ---------------------------------------------------------------------------
# no evidence -> freeze, block, escalate to the recheck station
# ---------------------------------------------------------------------------


def test_indeterminate_conflict_freezes_and_escalates(plant):
    first, second = _two_flights(plant)
    prior = plant.check_in("FZ", first.id)
    outcome = plant.check_in_adjudicated("FZ", second.id)
    assert outcome.verdict == Verdict.INDETERMINATE.value
    case = plant.arbitrator.case(outcome.case_id)
    assert case.status is CaseStatus.OPEN
    assert case.recheck_item_id == case.id
    assert case.escalated_at
    assert plant.arbitrator.recheck_items()[0].id == case.id


def test_frozen_barcode_blocks_a_further_unrelated_read(plant):
    first, second = _two_flights(plant)
    plant.check_in("FZ2", first.id)
    plant.check_in_adjudicated("FZ2", second.id)
    third = register_flight(plant, code="CA333", chute="CHC")
    with pytest.raises(ConflictError):
        plant.check_in("FZ2", third.id)


def test_unsettled_bag_cannot_be_sorted(plant):
    first, second = _two_flights(plant)
    prior = plant.check_in("BLK", first.id)
    outcome = plant.check_in_adjudicated("BLK", second.id)
    assert plant.sort_block_reason(prior.id) == "barcode_frozen_pending_review"
    with pytest.raises(OrderingError):
        plant.push_bag(prior.id, "CHA")
    with pytest.raises(OrderingError):
        plant.push_bag(outcome.bag.id, "CHB")


def test_recheck_queue_is_one_item_per_case_and_fifo(plant):
    a1, a2 = _two_flights(plant)
    plant.check_in("Q1", a1.id)
    c1 = plant.check_in_adjudicated("Q1", a2.id).case_id
    plant.check_in("Q2", a1.id)
    c2 = plant.check_in_adjudicated("Q2", a2.id).case_id
    assert [case.id for case in plant.arbitrator.recheck_items()] == [c1, c2]
    # duplicate reporting never adds a second queue item
    claims = plant.arbitrator.claims("Q1")
    plant.report_conflict(claims[0].id, claims[1].id)
    assert [case.id for case in plant.arbitrator.recheck_items()] == [c1, c2]


# ---------------------------------------------------------------------------
# manual resolution
# ---------------------------------------------------------------------------


def test_manual_reflight_resolution_releases_the_new_owner(plant):
    first, second = _two_flights(plant)
    prior = plant.check_in("MR", first.id)
    outcome = plant.check_in_adjudicated("MR", second.id)
    decision = plant.resolve_conflict(outcome.case_id, "same_bag_reflight", "clerk-1", "ok")
    assert decision.policy["chosen"] == POLICY_CODE
    assert decision.policy["rejected"] == PolicyChoice.LAST_READ_WINS.value
    assert decision.policy["chosen_risk"] == "missed_sortation"
    assert decision.policy["rejected_risk"] == "mis_sortation"
    assert plant.bag(prior.id).state is BagState.DIVERTED
    owner = plant.bag(outcome.bag.id)
    assert owner.state is BagState.INJECTED
    assert plant.mapping.flight_of(owner.id) == second.id
    assert not plant.arbitrator.is_frozen("MR")
    assert plant.arbitrator.recheck_items() == []
    # and the owner can finally be sorted
    plant.push_bag(owner.id, "CHB")


def test_manual_shared_resolution_keeps_both_pieces_but_their_own_flights(plant):
    first, second = _two_flights(plant)
    prior = plant.check_in("MS", first.id)
    outcome = plant.check_in_adjudicated("MS", second.id)
    plant.resolve_conflict(outcome.case_id, "shared_barcode", "clerk-2")
    for bag_id, flight in ((prior.id, first.id), (outcome.bag.id, second.id)):
        assert plant.mapping.flight_of(bag_id) == flight
        assert plant.is_sortable(bag_id)
    claims = {claim.bag_id: claim for claim in plant.arbitrator.claims("MS")}
    assert all(claim.status is ClaimStatus.SHARED for claim in claims.values())


def test_resolution_is_idempotent_but_rejects_a_contradiction(plant):
    first, second = _two_flights(plant)
    plant.check_in("ID", first.id)
    outcome = plant.check_in_adjudicated("ID", second.id)
    decision = plant.resolve_conflict(outcome.case_id, "shared_barcode", "c")
    again = plant.resolve_conflict(outcome.case_id, "shared_barcode", "c")
    assert again.id == decision.id
    assert len(plant.arbitrator.decisions()) == 1
    with pytest.raises(OrderingError):
        plant.resolve_conflict(outcome.case_id, "same_bag_reflight", "c")


def test_manual_resolution_must_pick_a_real_verdict(plant):
    first, second = _two_flights(plant)
    plant.check_in("IV", first.id)
    outcome = plant.check_in_adjudicated("IV", second.id)
    with pytest.raises(ValidationError):
        plant.resolve_conflict(outcome.case_id, "indeterminate", "c")
    with pytest.raises(ValidationError):
        plant.resolve_conflict(outcome.case_id, "nonsense", "c")


# ---------------------------------------------------------------------------
# conflict reporting
# ---------------------------------------------------------------------------


def test_report_conflict_is_idempotent_for_one_pair(plant):
    first, second = _two_flights(plant)
    plant.check_in("RP", first.id)
    outcome = plant.check_in_adjudicated("RP", second.id)
    claims = plant.arbitrator.claims("RP")
    reported = plant.report_conflict(claims[0].id, claims[1].id)
    assert reported.id == outcome.case_id
    assert len(plant.arbitrator.cases()) == 1
    assert len(plant.arbitrator.decisions()) <= 1


def test_report_conflict_rejects_unrelated_or_agreeing_reads(plant):
    first, second = _two_flights(plant)
    plant.check_in("AA", first.id)
    plant.check_in("BB", first.id)
    plant.check_in("CC", first.id)
    aa = plant.arbitrator.claims("AA")[0]
    bb = plant.arbitrator.claims("BB")[0]
    cc = plant.arbitrator.claims("CC")[0]
    # reads of different barcodes cannot form a barcode conflict
    with pytest.raises(ValidationError):
        plant.report_conflict(aa.id, bb.id)
    with pytest.raises(ValidationError):
        plant.report_conflict(aa.id, cc.id)
    # a repeated submission returning the same owner claim finds no pair:
    # reporting a claim against itself agrees on the flight and is a no-op
    with pytest.raises(ConflictError):
        plant.report_conflict(aa.id, aa.id)


def test_reporting_two_committed_reads_of_one_code_opens_the_case(plant):
    # Legacy data: two reads/bags for the same barcode were committed before
    # arbitration ran, so no case exists yet. An explicit report reconciles them.
    first, second = _two_flights(plant)
    prior_bag = plant.check_in("LEG", first.id)
    state = plant.store.state()
    from bagsort.arbitration.entities import ReadClaim
    from bagsort.arbitration.enums import ClaimStatus
    from bagsort.model.entities import Bag
    from bagsort.model.enums import BagState
    from bagsort.sequence.allocator import allocate

    allocation = allocate(state.sequence)
    legacy_id = "T1-LEG#2"
    legacy = Bag(
        id=legacy_id, barcode="LEG", sequence=allocation.value,
        epoch=allocation.sequence.epoch, flight_id=second.id, state=BagState.INJECTED,
    )
    claim = ReadClaim(
        id=f"{legacy_id}:read", barcode="LEG", bag_id=legacy_id, flight_id=second.id,
        sequence=allocation.value, epoch=allocation.sequence.epoch, weight=0.0,
        status=ClaimStatus.OWNER, at=plant.clock.now_iso(),
    )
    with plant.store.session() as session:
        session.state.sequence = allocation.sequence
        session.state.bags[legacy_id] = legacy
        session.state.read_claims[claim.id] = claim
        session.state.mappings[legacy_id] = second.id
        session.record("sequence.put", allocation.sequence.as_dict())
        session.record("bag.put", {"bag": legacy.as_dict()})
        session.record("claim.put", {"claim": claim.as_dict()})
        session.record("mapping.put", {"bag_id": legacy_id, "flight_id": second.id})

    prior_claim = plant.arbitrator.claims("LEG")[0]
    case = plant.report_conflict(prior_claim.id, claim.id)
    assert case.barcode == "LEG"
    assert case.is_open
    assert plant.arbitrator.is_frozen("LEG")
    assert not plant.is_sortable(prior_bag.id)
    assert plant.arbitration_consistency()["consistent"]


# ---------------------------------------------------------------------------
# replay, persistence, audit
# ---------------------------------------------------------------------------


def test_arbitration_state_rebuilds_from_the_journal(plant):
    first, second = _two_flights(plant)
    plant.check_in("RL", first.id)
    plant.check_in_adjudicated("RL", second.id)
    rebuilt = replay(plant.store.entries())
    assert rebuilt.as_dict() == plant.store.state().as_dict()
    assert set(rebuilt.frozen_barcodes) == {"RL"}
    assert rebuilt.conflict_recheck_queue


def test_arbitration_survives_a_restart():
    import tempfile
    from pathlib import Path

    workdir = Path(tempfile.mkdtemp())
    fresh = make_plant(workdir)
    first = register_flight(fresh, code="CA111", chute="CHA")
    second = register_flight(fresh, code="CA222", chute="CHB")
    fresh.check_in("PV", first.id)
    outcome = fresh.check_in_adjudicated("PV", second.id)
    fresh.store.persist()
    reopened = Store.open(fresh.store.path)
    rebuilt_plant = type(fresh).build(reopened, terminal="T1", clock=fresh.clock)
    assert rebuilt_plant.arbitrator.case(outcome.case_id).is_open
    assert rebuilt_plant.arbitrator.is_frozen("PV")
    assert rebuilt_plant.arbitrator.recheck_items()[0].id == outcome.case_id


def test_every_decision_is_audit_linked_and_the_chain_verifies(plant):
    first, second = _two_flights(plant)
    plant.record_reflight("AU", first.id, second.id)
    plant.check_in("AU", first.id)
    plant.check_in_adjudicated("AU", second.id)
    kinds = plant.audit.by_kind()
    assert kinds.get("barcode_conflict", 0) >= 1
    assert kinds.get("barcode_decision", 0) >= 1
    assert kinds.get("reflight_evidence", 0) >= 1
    assert plant.audit.verify()


def test_escalation_and_unfreeze_are_audited(plant):
    first, second = _two_flights(plant)
    plant.check_in("ES", first.id)
    outcome = plant.check_in_adjudicated("ES", second.id)
    assert plant.audit.by_kind().get("barcode_escalation", 0) == 1
    plant.resolve_conflict(outcome.case_id, "shared_barcode", "c")
    assert plant.audit.by_kind().get("barcode_unfreeze", 0) == 1
    assert plant.audit.verify()


def test_decision_history_can_be_reviewed_entry_by_entry(plant):
    first, second = _two_flights(plant)
    plant.check_in("H1", first.id)
    c1 = plant.check_in_adjudicated("H1", second.id).case_id
    plant.resolve_conflict(c1, "shared_barcode", "c")
    decisions = plant.arbitrator.decisions()
    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.case_id == c1
    assert decision.prior_flight_id == first.id
    assert decision.latter_flight_id == second.id
    assert decision.policy["chosen"]
    history = plant.arbitrator.history("H1")
    kinds = [row["kind"] for row in history]
    stage_order = ["read", "conflict_detected", "escalated", "decision"]
    assert kinds == sorted(kinds, key=lambda kind: stage_order.index(kind))
    assert kinds.count("read") == 2
    assert "decision" in kinds and "conflict_detected" in kinds


def test_fold_is_a_pure_reducer_for_arbitration_commands(plant):
    first, second = _two_flights(plant)
    plant.check_in("FD", first.id)
    plant.check_in_adjudicated("FD", second.id)
    pairs = [(entry.kind, entry.payload) for entry in plant.store.entries()]
    assert fold(pairs).as_dict() == plant.store.state().as_dict()


def test_a_third_read_after_a_shared_verdict_opens_a_new_case(plant):
    first = register_flight(plant, code="CA111", chute="CHA")
    second = register_flight(plant, code="CA222", chute="CHB")
    third = register_flight(plant, code="CA333", chute="CHC")
    plant.check_in("T3", first.id)
    first_conflict = plant.check_in_adjudicated("T3", second.id)
    plant.resolve_conflict(first_conflict.case_id, "shared_barcode", "c")
    assert not plant.arbitrator.is_frozen("T3")

    outcome = plant.check_in_adjudicated("T3", third.id)
    assert outcome.case_id
    assert outcome.case_id != first_conflict.case_id
    assert outcome.bag.id == "T1-T3#3"
    assert plant.arbitrator.is_frozen("T3")
    assert len(plant.arbitrator.claims("T3")) == 3
    assert plant.arbitration_consistency()["consistent"]


def test_a_read_repeated_after_reflight_resolution_is_idempotent(plant):
    first, second = _two_flights(plant)
    plant.record_reflight("RF", first.id, second.id)
    plant.check_in("RF", first.id)
    one = plant.check_in_adjudicated("RF", second.id)
    again = plant.check_in_adjudicated("RF", second.id)
    assert again.duplicate is True
    assert again.decision_id == one.decision_id
    assert len(plant.arbitrator.claims("RF")) == 2
