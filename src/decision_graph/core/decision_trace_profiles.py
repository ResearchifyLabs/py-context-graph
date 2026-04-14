from __future__ import annotations
from typing import Optional

INTERNAL_TEAM_DECISION_TYPES = [
    "ActionRequest",
    "TaskUpdate",
    "StatusUpdate",
    "Decision",
    "ReviewRequest",
    "Approval",
    "ChangeAnnouncement",
    "BugReport",
    "FeatureRequest",
    "AccessRequest",
    "Other",
]


INDUSTRY_PROFILES: dict[str, list[str]] = {
    "product_support": [
        "BugReport",
        "FeatureRequest",
        "ActionRequest",
        "AccessRequest",
        "StatusUpdate",
        "ReviewRequest",
        "ChangeAnnouncement",
        "Other",
    ],
    "concierge": [
        "TravelBooking",
        "BookingArrangement",
        "PurchaseIntent",
        "VendorFollowUp",
        "PaymentRequest",
        "PaymentConfirmation",
        "StatusUpdate",
        "ActionRequest",
        "PromotionalMessage",
        "ClientPreference",
        "Other",
    ],
    "investment": [
        "ReviewRequest",
        "Approval",
        "PaymentRequest",
        "PaymentConfirmation",
        "StatusUpdate",
        "ActionRequest",
        "Other",
    ],
    "logistics_ops": [
        "LogisticsArrangement",
        "VendorFollowUp",
        "QuotePending",
        "PaymentRequest",
        "PaymentConfirmation",
        "StatusUpdate",
        "ActionRequest",
        "Other",
    ],
    "generic_b2b": [
        "ActionRequest",
        "Decision",
        "StatusUpdate",
        "TaskUpdate",
        "Other",
    ],
    "internal_team": INTERNAL_TEAM_DECISION_TYPES,
}


def allowed_decision_types_for_industry(industry_type: Optional[str]) -> list[str]:
    industry_type = (industry_type or "").strip().lower()
    if not industry_type:
        return INDUSTRY_PROFILES["generic_b2b"]
    return INDUSTRY_PROFILES.get(industry_type, INDUSTRY_PROFILES["generic_b2b"])
