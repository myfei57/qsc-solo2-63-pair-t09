"""Enumerations owned by the barcode conflict arbitration subsystem."""

from __future__ import annotations

from enum import Enum


class ClaimStatus(str, Enum):
    """Lifecycle of one committed barcode read (one claim to ownership)."""

    PENDING = "pending"
    OWNER = "owner"
    SUPERSEDED = "superseded"
    SHARED = "shared"
    REJECTED = "rejected"


class CaseStatus(str, Enum):
    """Lifecycle of one barcode conflict case."""

    OPEN = "open"
    RESOLVED = "resolved"


class Verdict(str, Enum):
    """What a conflict is judged to be."""

    SAME_BAG_REFLIGHT = "same_bag_reflight"
    SHARED_BARCODE = "shared_barcode"
    INDETERMINATE = "indeterminate"


class PolicyChoice(str, Enum):
    """The two ways a barcode conflict can be settled.

    They trade the two sortation errors against each other in opposite
    directions, so every decision names the one that was taken.
    """

    FREEZE_PENDING_REVIEW = "freeze_pending_review"
    LAST_READ_WINS = "last_read_wins"
