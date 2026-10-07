"""Barcode conflict arbitration: rulings, idempotency, queues and replay."""

from __future__ import annotations

import pytest

from bagsort.arbitration import policy
from bagsort.arbitration.correspondence import replay_projection
from bagsort.errors import ConflictError, NotFoundError, OrderingError
from bagsort.model.enums import ConflictState, ReadStatus
from bagsort.store.store import Store
from tests.support import make_plant


def _plant(tmp_path):
    return make_plant(tmp_path)


def _flights(plant):
    first = plant.register_flight("CA1001")
    second = plant.register_flight("CA2002")
    return first, second


def test_first_read_creates_bag_and_mapping(tmp_path):
    plant = _plant(tmp_path)
    flight, _ = _flights(plant)
    result = plant.report_reading("B1", flight.id)
    assert result.outcome == "first"
    state = plant.store.state()
    assert state.mappings[result.bag.id] == flight.id
    assert len(plant.arbiter.reads("B1")) == 1
    assert state.arbitration_policy == policy.POLICY_FREEZE
    assert state.arbitration_policy_rationale


def test_unknown_flight_is_rejected(tmp_path):
    plant = _plant(tmp_path)
    with pytest.raises(NotFoundError):
        plant.report_reading("B1", "T1-NOPE")


def test_conflict_freezes_and_escalates(tmp_path):
    plant = _plant(tmp_path)
    first, second = _flights(plant)
    plant.report_reading("B9", first.id)
    plant.clock.advance(10)
    result = plant.report_reading("B9", second.id)
    assert result.outcome == "inconclusive"
    case = result.case
    assert case.state is ConflictState.OPEN
    assert case.decision.value == "inconclusive"
    assert case.claimed_flights == [first.id, second.id]
    assert "B9" in plant.store.state().frozen_barcodes
    assert [item.id for item in plant.review_queue()] == [case.id]
    assert case.policy == policy.POLICY_FREEZE and case.policy_rationale


def test_frozen_bag_cannot_be_sorted(tmp_path):
    plant = _plant(tmp_path)
    first, second = _flights(plant)
    initial = plant.report_reading("B9", first.id)
    plant.clock.advance(10)
    plant.report_reading("B9", second.id)
    plant.assign_chute(second.id, plant.namespace.chute_id("A"))
    with pytest.raises(ConflictError):
        plant.push_bag(initial.bag.id, plant.namespace.chute_id("A"))
    # the denial itself is auditable
    assert any(entry.kind == "sort_blocked" for entry in plant.audit.entries())
    assert plant.verify_correspondence() == []


def test_rereporting_same_conflict_is_idempotent(tmp_path):
    plant = _plant(tmp_path)
    first, second = _flights(plant)
    plant.report_reading("B9", first.id)
    plant.clock.advance(10)
    opened = plant.report_reading("B9", second.id)
    watermark = plant.store.watermark
    audit_count = plant.audit.count()
    again = plant.report_reading("B9", second.id)
    assert again.outcome == "rereported"
    assert again.duplicate is True
    assert again.case.id == opened.case.id
    assert plant.store.watermark == watermark
    assert plant.audit.count() == audit_count
    assert [c.id for c in plant.review_queue()] == [opened.case.id]


def test_a_new_flight_claim_attaches_to_the_open_case(tmp_path):
    plant = _plant(tmp_path)
    first, second = _flights(plant)
    third = plant.register_flight("CA3003")
    plant.report_reading("B9", first.id)
    plant.clock.advance(10)
    opened = plant.report_reading("B9", second.id)
    plant.clock.advance(10)
    attached = plant.report_reading("B9", third.id)
    assert attached.outcome == "rereported"
    case = plant.arbiter.case(opened.case.id)
    assert case.claimed_flights == [first.id, second.id, third.id]
    # still one adjudication
    assert len(plant.conflict_cases()) == 1


def test_manual_reroute_resolves_and_unfreezes(tmp_path):
    plant = _plant(tmp_path)
    first, second = _flights(plant)
    initial = plant.report_reading("B9", first.id)
    plant.clock.advance(10)
    opened = plant.report_reading("B9", second.id)
    plant.resolve_conflict(
        opened.case.id, "reroute", new_flight_id=second.id,
        operator="alice", note="same bag, passenger rebooked",
    )
    plant.complete_review(opened.case.id)
    case = plant.arbiter.case(opened.case.id)
    assert case.state is ConflictState.RESOLVED
    assert case.decision.value == "reroute"
    assert case.decided_by == "alice"
    assert "B9" not in plant.store.state().frozen_barcodes
    assert plant.store.state().mappings[initial.bag.id] == second.id
    plant.assign_chute(second.id, plant.namespace.chute_id("A"))
    record = plant.push_bag(initial.bag.id, plant.namespace.chute_id("A"))
    assert record.flight_id == second.id


def test_resolve_is_idempotent(tmp_path):
    plant = _plant(tmp_path)
    first, second = _flights(plant)
    plant.report_reading("B9", first.id)
    plant.clock.advance(10)
    opened = plant.report_reading("B9", second.id)
    plant.resolve_conflict(opened.case.id, "reroute", new_flight_id=second.id)
    watermark = plant.store.watermark
    audit_count = plant.audit.count()
    plant.resolve_conflict(opened.case.id, "reroute", new_flight_id=second.id)
    assert plant.store.watermark == watermark
    assert plant.audit.count() == audit_count


def test_duplicate_read_of_settled_bag_is_a_noop(tmp_path):
    plant = _plant(tmp_path)
    first, second = _flights(plant)
    initial = plant.report_reading("B9", first.id)
    plant.clock.advance(10)
    opened = plant.report_reading("B9", second.id)
    plant.resolve_conflict(opened.case.id, "reroute", new_flight_id=second.id)
    plant.complete_review(opened.case.id)
    watermark = plant.store.watermark
    result = plant.report_reading("B9", second.id)
    assert result.outcome == "duplicate"
    assert plant.store.watermark == watermark
    assert initial.bag.id


def test_dcs_evidence_auto_adjudicates_reroute(tmp_path):
    plant = _plant(tmp_path)
    first, second = _flights(plant)
    initial = plant.report_reading("B7", first.id)
    plant.record_evidence(
        "B7", policy.EVIDENCE_DCS_REROUTE,
        from_flight_id=first.id, to_flight_id=second.id, source="dcs",
    )
    plant.clock.advance(5)
    result = plant.report_reading("B7", second.id)
    assert result.outcome == "reroute"
    case = result.case
    assert case.state.value == "auto_adjudicated"
    assert "B7" not in plant.store.state().frozen_barcodes
    assert plant.review_queue() == []
    assert plant.store.state().mappings[initial.bag.id] == second.id
    plant.assign_chute(second.id, plant.namespace.chute_id("B"))
    assert plant.push_bag(initial.bag.id, plant.namespace.chute_id("B")).flight_id == second.id


def test_dual_track_marks_distinct_and_holds_for_relabel(tmp_path):
    plant = _plant(tmp_path)
    first, second = _flights(plant)
    one = plant.report_reading("B5", first.id)
    plant.record_evidence("B5", policy.EVIDENCE_DUAL_TRACK, source="cv-track")
    plant.clock.advance(5)
    result = plant.report_reading("B5", second.id)
    assert result.outcome == "distinct"
    case = result.case
    assert case.state is ConflictState.OPEN
    assert case.decision.value == "distinct"
    assert len(case.bag_ids) == 2
    assert case.bag_ids[0] == one.bag.id
    assert "B5" in plant.store.state().frozen_barcodes
    for bag_id in case.bag_ids:
        assert plant.bag(bag_id).state.value == "rechecked"
    # human relabel: both pieces then sort to their own flights
    plant.resolve_conflict(case.id, "distinct", operator="bob")
    plant.complete_review(case.id)
    settled = plant.arbiter.case(case.id)
    assert settled.state is ConflictState.RESOLVED
    assert "B5" not in plant.store.state().frozen_barcodes
    plant.assign_chute(first.id, plant.namespace.chute_id("A"))
    plant.assign_chute(second.id, plant.namespace.chute_id("C"))
    first_bag, second_bag = (plant.bag(bag_id) for bag_id in settled.bag_ids)
    assert plant.push_bag(first_bag.id, plant.namespace.chute_id("A")).flight_id == first.id
    assert plant.push_bag(second_bag.id, plant.namespace.chute_id("C")).flight_id == second.id


def test_review_queue_is_fifo(tmp_path):
    plant = _plant(tmp_path)
    first, second = _flights(plant)
    a = plant.report_reading("A1", first.id)
    plant.clock.advance(1)
    a_conflict = plant.report_reading("A1", second.id)
    plant.report_reading("Z9", first.id)
    plant.clock.advance(1)
    z_conflict = plant.report_reading("Z9", second.id)
    with pytest.raises(OrderingError):
        plant.resolve_conflict(z_conflict.case.id, "reroute", new_flight_id=second.id)
    assert plant.next_review().id == a_conflict.case.id
    plant.resolve_conflict(a_conflict.case.id, "reroute", new_flight_id=second.id)
    plant.complete_review(a_conflict.case.id)
    assert plant.next_review().id == z_conflict.case.id


def test_history_is_replayable_entry_by_entry(tmp_path):
    plant = _plant(tmp_path)
    first, second = _flights(plant)
    plant.report_reading("B9", first.id)
    plant.clock.advance(10)
    opened = plant.report_reading("B9", second.id)
    plant.clock.advance(30)
    plant.resolve_conflict(opened.case.id, "reroute", new_flight_id=second.id, operator="alice")
    trail = plant.conflict_history(opened.case.id)[0]["trail"]
    events = [row["event"] for row in trail]
    assert events == ["reading_committed", "reading_committed", "case_opened", "resolved"]
    assert all(row["at"] for row in trail)


def test_replay_by_time_reconstructs_state(tmp_path):
    plant = _plant(tmp_path)
    first, second = _flights(plant)
    plant.report_reading("B9", first.id)
    plant.clock.advance(10)
    opened = plant.report_reading("B9", second.id)
    freeze_at = opened.case.created_at
    before = replay_projection(plant.store, instant="2024-01-01T00:00:05Z")
    assert before.frozen == set()
    after = replay_projection(plant.store, instant=freeze_at)
    assert "B9" in after.frozen
    assert opened.case.id in after.cases
    # after the desk rules, the code is released
    plant.clock.advance(30)
    plant.resolve_conflict(opened.case.id, "reroute", new_flight_id=second.id)
    released = replay_projection(plant.store)
    assert "B9" not in released.frozen


def test_correspondence_holds_and_survives_restart(tmp_path):
    plant = _plant(tmp_path)
    first, second = _flights(plant)
    plant.report_reading("B1", first.id)
    plant.report_reading("B2", first.id)
    plant.clock.advance(2)
    plant.report_reading("B2", second.id)
    assert plant.verify_correspondence() == []
    plant.store.persist()
    reopened = Store.open(plant.store.path)
    reopened_plant = type(plant).build(reopened, clock=plant.clock)
    assert reopened_plant.verify_correspondence() == []


def test_every_reading_has_a_mapping_when_attributed(tmp_path):
    plant = _plant(tmp_path)
    first, second = _flights(plant)
    initial = plant.report_reading("B9", first.id)
    plant.clock.advance(1)
    plant.report_reading("B9", second.id)
    state = plant.store.state()
    attributed = [r for r in state.reads if r.status is ReadStatus.FROZEN]
    assert len(attributed) == 2
    # the attributed first bag still maps; the frozen later read is not yet owned
    assert state.mappings.get(initial.bag.id) == first.id
    assert plant.verify_correspondence() == []


def test_audit_chain_records_conflict_and_resolution(tmp_path):
    plant = _plant(tmp_path)
    first, second = _flights(plant)
    plant.report_reading("B9", first.id)
    plant.clock.advance(1)
    opened = plant.report_reading("B9", second.id)
    plant.resolve_conflict(opened.case.id, "reroute", new_flight_id=second.id)
    kinds = {entry.kind for entry in plant.audit.entries()}
    assert "conflict_open" in kinds
    assert "conflict_resolve" in kinds
    assert plant.audit.verify()


# ---------------------------------------------------------------------------
# HTTP console
# ---------------------------------------------------------------------------

import json

from tests.support import make_server


def test_api_full_conflict_cycle(tmp_path):
    server = make_server(tmp_path)
    plant = server.plant
    first = plant.register_flight("CA1001")
    second = plant.register_flight("CA2002")

    resp = server.call("POST", "/api/arbitration/read",
                       {"barcode": "W1", "flight_id": first.id})
    assert resp.status == 201
    assert json.loads(resp.body)["outcome"] == "first"

    plant.clock.advance(10)
    resp = server.call("POST", "/api/arbitration/read",
                       {"barcode": "W1", "flight_id": second.id})
    body = json.loads(resp.body)
    assert body["outcome"] == "inconclusive"
    case_id = body["case"]["id"]

    resp = server.call("GET", "/api/arbitration/review")
    queued = json.loads(resp.body)
    assert [case["id"] for case in queued] == [case_id]

    resp = server.call("POST", f"/api/arbitration/cases/{case_id}/resolve",
                       {"decision": "reroute", "new_flight_id": second.id,
                        "operator": "carol"})
    assert resp.status == 200
    assert json.loads(resp.body)["state"] == "resolved"

    resp = server.call("POST", f"/api/arbitration/cases/{case_id}/complete")
    assert resp.status == 200

    resp = server.call("GET", f"/api/arbitration/cases/{case_id}/history")
    trail = json.loads(resp.body)[0]["trail"]
    assert [row["event"] for row in trail] == [
        "reading_committed", "reading_committed", "case_opened", "resolved"]

    resp = server.call("GET", "/api/arbitration/verify")
    assert json.loads(resp.body) == {"ok": True, "violations": []}


def test_api_evidence_auto_reroute(tmp_path):
    server = make_server(tmp_path)
    plant = server.plant
    first = plant.register_flight("CA1001")
    second = plant.register_flight("CA2002")
    server.call("POST", "/api/arbitration/read", {"barcode": "W2", "flight_id": first.id})
    resp = server.call("POST", "/api/arbitration/evidence", {
        "barcode": "W2", "kind": "dcs.reroute",
        "from_flight_id": first.id, "to_flight_id": second.id, "source": "dcs"})
    assert resp.status == 201
    plant.clock.advance(5)
    resp = server.call("POST", "/api/arbitration/read",
                       {"barcode": "W2", "flight_id": second.id})
    assert json.loads(resp.body)["outcome"] == "reroute"


def test_api_read_ledger_is_listed(tmp_path):
    server = make_server(tmp_path)
    plant = server.plant
    flight = plant.register_flight("CA1001")
    server.call("POST", "/api/arbitration/read", {"barcode": "W3", "flight_id": flight.id})
    resp = server.call("GET", "/api/arbitration/reads", query={"barcode": ["W3"]})
    rows = json.loads(resp.body)
    assert len(rows) == 1 and rows[0]["barcode"] == "W3"
