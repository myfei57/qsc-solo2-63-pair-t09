"""Immutable records of the barcode conflict arbitration subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .enums import CaseStatus, ClaimStatus, PolicyChoice, Verdict

POLICY_CODE = PolicyChoice.FREEZE_PENDING_REVIEW.value


def policy_reason() -> dict[str, str]:
    """The chosen policy and why the alternative was rejected.

    ``last_read_wins`` keeps the belt moving, so it never leaks a bag through
    the *missed sortation* path. But when two physical pieces share one code it
    silently points both at the later flight: one piece is physically loaded
    onto the wrong aircraft, which is effectively irreversible once airborne.
    ``freeze_pending_review`` instead stalls both claimants at the recheck
    station. Its worst case is a *missed* sortation, which a manual review
    station recovers before the flight closes. The freeze policy is therefore
    the conservative side of the trade and the only one consistent with the
    rule that baggage with no settled ownership may not be sorted.
    """

    return {
        "chosen": POLICY_CODE,
        "rejected": PolicyChoice.LAST_READ_WINS.value,
        "chosen_risk": "missed_sortation",
        "rejected_risk": "mis_sortation",
        "rationale": (
            "last_read_wins silently mis-loads a physically shared barcode onto "
            "the later flight, which is irreversible after departure; freezing "
            "both claimants for manual review delays sortation but never routes "
            "a bag whose ownership is unsettled"
        ),
    }


def _clean(raw: Any) -> Any:
    if isinstance(raw, dict):
        return {str(key): _clean(value) for key, value in raw.items()}
    if isinstance(raw, (list, tuple)):
        return [_clean(item) for item in raw]
    return raw


@dataclass(frozen=True)
class ReflightDirective:
    """Evidence that one barcode was officially moved from one flight to another."""

    barcode: str
    from_flight_id: str
    to_flight_id: str
    recorded_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "barcode": self.barcode,
            "from_flight_id": self.from_flight_id,
            "to_flight_id": self.to_flight_id,
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ReflightDirective":
        return cls(
            barcode=str(raw.get("barcode", "")),
            from_flight_id=str(raw.get("from_flight_id", "")),
            to_flight_id=str(raw.get("to_flight_id", "")),
            recorded_at=str(raw.get("recorded_at", "")),
        )


@dataclass(frozen=True)
class ReadClaim:
    """One committed barcode read and its claim to which flight the bag belongs."""

    id: str
    barcode: str
    bag_id: str
    flight_id: str
    sequence: int
    epoch: int
    weight: float
    status: ClaimStatus
    at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "barcode": self.barcode,
            "bag_id": self.bag_id,
            "flight_id": self.flight_id,
            "sequence": self.sequence,
            "epoch": self.epoch,
            "weight": self.weight,
            "status": self.status.value,
            "at": self.at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ReadClaim":
        return cls(
            id=str(raw.get("id", "")),
            barcode=str(raw.get("barcode", "")),
            bag_id=str(raw.get("bag_id", "")),
            flight_id=str(raw.get("flight_id", "")),
            sequence=int(raw.get("sequence", 0)),
            epoch=int(raw.get("epoch", 0)),
            weight=float(raw.get("weight", 0.0)),
            status=ClaimStatus(raw.get("status", ClaimStatus.PENDING.value)),
            at=str(raw.get("at", "")),
        )

    def evolved(self, **changes: Any) -> "ReadClaim":
        return replace(self, **changes)


@dataclass(frozen=True)
class ConflictCase:
    """One conflict between two reads of the same barcode pointing at different flights."""

    id: str
    barcode: str
    prior_claim_id: str
    latter_claim_id: str
    status: CaseStatus
    detected_at: str
    escalated_at: str
    recheck_item_id: str
    evidence: dict[str, Any] = field(default_factory=dict)
    decision_id: str = ""
    resolved_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "barcode": self.barcode,
            "prior_claim_id": self.prior_claim_id,
            "latter_claim_id": self.latter_claim_id,
            "status": self.status.value,
            "detected_at": self.detected_at,
            "escalated_at": self.escalated_at,
            "recheck_item_id": self.recheck_item_id,
            "evidence": _clean(self.evidence),
            "decision_id": self.decision_id,
            "resolved_at": self.resolved_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ConflictCase":
        return cls(
            id=str(raw.get("id", "")),
            barcode=str(raw.get("barcode", "")),
            prior_claim_id=str(raw.get("prior_claim_id", "")),
            latter_claim_id=str(raw.get("latter_claim_id", "")),
            status=CaseStatus(raw.get("status", CaseStatus.OPEN.value)),
            detected_at=str(raw.get("detected_at", "")),
            escalated_at=str(raw.get("escalated_at", "")),
            recheck_item_id=str(raw.get("recheck_item_id", "")),
            evidence=_clean(raw.get("evidence", {})),
            decision_id=str(raw.get("decision_id", "")),
            resolved_at=str(raw.get("resolved_at", "")),
        )

    def evolved(self, **changes: Any) -> "ConflictCase":
        return replace(self, **changes)

    @property
    def is_open(self) -> bool:
        return self.status is CaseStatus.OPEN


@dataclass(frozen=True)
class DecisionRecord:
    """The adjudication of one conflict case, complete enough to be re-checked."""

    id: str
    case_id: str
    barcode: str
    verdict: Verdict
    decided_by: str
    basis: str
    evidence: dict[str, Any]
    policy: dict[str, Any]
    prior_claim_id: str
    latter_claim_id: str
    prior_bag_id: str
    latter_bag_id: str
    prior_flight_id: str
    latter_flight_id: str
    at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "case_id": self.case_id,
            "barcode": self.barcode,
            "verdict": self.verdict.value,
            "decided_by": self.decided_by,
            "basis": self.basis,
            "evidence": _clean(self.evidence),
            "policy": _clean(self.policy),
            "prior_claim_id": self.prior_claim_id,
            "latter_claim_id": self.latter_claim_id,
            "prior_bag_id": self.prior_bag_id,
            "latter_bag_id": self.latter_bag_id,
            "prior_flight_id": self.prior_flight_id,
            "latter_flight_id": self.latter_flight_id,
            "at": self.at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DecisionRecord":
        return cls(
            id=str(raw.get("id", "")),
            case_id=str(raw.get("case_id", "")),
            barcode=str(raw.get("barcode", "")),
            verdict=Verdict(raw.get("verdict", Verdict.INDETERMINATE.value)),
            decided_by=str(raw.get("decided_by", "")),
            basis=str(raw.get("basis", "")),
            evidence=_clean(raw.get("evidence", {})),
            policy=_clean(raw.get("policy", {})),
            prior_claim_id=str(raw.get("prior_claim_id", "")),
            latter_claim_id=str(raw.get("latter_claim_id", "")),
            prior_bag_id=str(raw.get("prior_bag_id", "")),
            latter_bag_id=str(raw.get("latter_bag_id", "")),
            prior_flight_id=str(raw.get("prior_flight_id", "")),
            latter_flight_id=str(raw.get("latter_flight_id", "")),
            at=str(raw.get("at", "")),
        )
