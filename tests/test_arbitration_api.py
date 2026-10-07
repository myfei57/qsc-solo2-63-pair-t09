"""HTTP surface of barcode conflict arbitration."""

from __future__ import annotations

import json

from tests.support import make_server


def call(server, method, path, payload=None):
    response = server.call(method, path, payload)
    if response.content_type.startswith("application/json"):
        return response.status, json.loads(response.body)
    return response.status, response.body.decode("utf-8")


def flights(server):
    assert call(server, "POST", "/api/flights", {"code": "CA111", "chute": "CHA"})[0] == 201
    assert call(server, "POST", "/api/flights", {"code": "CA222", "chute": "CHB"})[0] == 201
    return "T1-CA111", "T1-CA222"


def test_policy_endpoint_documents_the_trade_off(server):
    status, policy = call(server, "GET", "/api/arbitration/policy")
    assert status == 200
    assert policy["code"] == "freeze_pending_review"
    assert policy["rejected"] == "last_read_wins"
    assert policy["chosen_risk"] == "missed_sortation"
    assert policy["rejected_risk"] == "mis_sortation"
    assert policy["rationale"]


def test_indeterminate_conflict_end_to_end(server):
    first, second = flights(server)
    call(server, "POST", "/api/bags/checkin", {"barcode": "AB1", "flight_id": first})

    status, outcome = call(
        server,
        "POST",
        "/api/bags/checkin-adjudicated",
        {"barcode": "AB1", "flight_id": second, "weight": 10.0},
    )
    assert status == 201
    assert outcome["escalated"] is True
    assert outcome["verdict"] == "indeterminate"
    assert outcome["sortable"] is False
    case_id = outcome["case_id"]

    status, queue = call(server, "GET", "/api/arbitration/recheck")
    assert [case["id"] for case in queue] == [case_id]
    assert call(server, "GET", "/api/arbitration/recheck/next")[1]["id"] == case_id

    assert call(server, "GET", "/api/arbitration/frozen")[1]["frozen"] == ["AB1"]

    status, case = call(server, "GET", f"/api/arbitration/cases/{case_id}")
    assert case["status"] == "open" and case["decision"] is None

    # the earlier bag is blocked from sorting
    prior = "T1-AB1"
    assert call(server, "GET", f"/api/bags/{prior}/sortable")[1] == {
        "bag_id": prior,
        "sortable": False,
        "reason": "barcode_frozen_pending_review",
    }
    status, _ = call(server, "POST", "/api/sort/dispatch", {"bag_id": prior})
    assert status == 409

    # manual resolution unfreezes and clears the queue
    status, decision = call(
        server,
        "POST",
        f"/api/arbitration/cases/{case_id}/resolve",
        {"verdict": "shared_barcode", "decided_by": "clerk-9", "note": "tags differ"},
    )
    assert status == 200
    assert decision["verdict"] == "shared_barcode"
    assert decision["policy"]["chosen"] == "freeze_pending_review"
    assert call(server, "GET", "/api/arbitration/frozen")[1]["frozen"] == []
    assert call(server, "GET", "/api/arbitration/recheck")[1] == []

    status, resolved = call(server, "GET", f"/api/arbitration/cases/{case_id}")
    assert resolved["status"] == "resolved"
    assert resolved["decision"]["id"] == decision["id"]

    # both pieces are now sortable, each to its own flight
    assert call(server, "GET", f"/api/bags/{prior}/sortable")[1]["sortable"] is True
    latter = "T1-AB1#2"
    assert call(server, "GET", f"/api/bags/{latter}/sortable")[1]["sortable"] is True


def test_directive_driven_reflight_is_automatic(server):
    first, second = flights(server)
    status, _ = call(
        server,
        "POST",
        "/api/arbitration/reflight",
        {"barcode": "RX", "from_flight_id": first, "to_flight_id": second},
    )
    assert status == 201
    call(server, "POST", "/api/bags/checkin", {"barcode": "RX", "flight_id": first})
    status, outcome = call(
        server,
        "POST",
        "/api/bags/checkin-adjudicated",
        {"barcode": "RX", "flight_id": second},
    )
    assert outcome["verdict"] == "same_bag_reflight"
    assert outcome["decision_id"] and outcome["escalated"] is False
    status, decisions = call(server, "GET", "/api/arbitration/decisions")
    assert len(decisions) == 1
    assert decisions[0]["basis"] == "reflight_directive"
    status, history = call(server, "GET", "/api/arbitration/barcode/RX/history")
    kinds = [row["kind"] for row in history]
    assert kinds[0] == "read" and "decision" in kinds


def test_conflict_report_endpoint_is_idempotent(server):
    first, second = flights(server)
    call(server, "POST", "/api/bags/checkin", {"barcode": "RP", "flight_id": first})
    outcome = call(
        server,
        "POST",
        "/api/bags/checkin-adjudicated",
        {"barcode": "RP", "flight_id": second},
    )[1]
    _, all_claims = call(server, "GET", "/api/arbitration/claims")
    claims = [claim for claim in all_claims if claim["barcode"] == "RP"]
    body = {"prior_claim_id": claims[0]["id"], "latter_claim_id": claims[1]["id"]}
    status, reported = call(server, "POST", "/api/arbitration/conflicts/report", body)
    assert status == 201 and reported["id"] == outcome["case_id"]
    call(server, "POST", "/api/arbitration/conflicts/report", body)
    assert len(call(server, "GET", "/api/arbitration/cases")[1]) == 1


def test_consistency_endpoint_reports_ok(server):
    first, second = flights(server)
    call(server, "POST", "/api/bags/checkin", {"barcode": "OK", "flight_id": first})
    call(
        server,
        "POST",
        "/api/bags/checkin-adjudicated",
        {"barcode": "OK", "flight_id": second},
    )
    status, report = call(server, "GET", "/api/arbitration/consistency")
    assert status == 200 and report["consistent"] is True and report["problems"] == []
