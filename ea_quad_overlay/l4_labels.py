"""Contracts and deterministic calculations for M1 L4 annotations."""

from __future__ import annotations

from collections.abc import Mapping


MODALITIES = ("text", "speech", "macro", "micro")
CONTRADICTION_TYPES = (
    "consistent",
    "masking",
    "sarcasm",
    "hidden_emotion",
    "intensity_mismatch",
)

TYPE_MULTIPLIERS = {
    "consistent": {"text": 1.0, "speech": 1.0, "macro": 1.0, "micro": 1.0},
    "sarcasm": {"text": 0.6, "speech": 1.2, "macro": 1.1, "micro": 1.0},
    "masking": {"text": 0.7, "speech": 1.1, "macro": 0.8, "micro": 1.2},
    "hidden_emotion": {"text": 0.8, "speech": 1.1, "macro": 0.7, "micro": 1.2},
    "intensity_mismatch": {
        "text": 1.0,
        "speech": 1.0,
        "macro": 1.0,
        "micro": 1.0,
    },
}


def calculate_fusion_weights(
    confidences: Mapping[str, float],
    contradiction_type: str,
    micro_review_status: str,
) -> dict[str, float]:
    """Calculate normalized modality weights under the M1 review policy."""
    if contradiction_type not in TYPE_MULTIPLIERS:
        raise ValueError(f"unsupported contradiction_type: {contradiction_type}")

    reliability = {"text": 1.0, "speech": 1.0, "macro": 1.0, "micro": 0.5}
    raw = {
        modality: float(confidences[modality])
        * reliability[modality]
        * TYPE_MULTIPLIERS[contradiction_type][modality]
        for modality in MODALITIES
    }
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("all raw fusion weights are zero")

    weights = {modality: raw[modality] / total for modality in MODALITIES}
    if micro_review_status == "pending_issue_5" and weights["micro"] > 0.1:
        excess = weights["micro"] - 0.1
        non_micro_total = sum(weights[modality] for modality in MODALITIES[:-1])
        if non_micro_total <= 0:
            raise ValueError("no non-micro weight is available")
        weights["micro"] = 0.1
        for modality in MODALITIES[:-1]:
            weights[modality] += excess * weights[modality] / non_micro_total

    rounded = {modality: round(weights[modality], 6) for modality in MODALITIES}
    rounded["macro"] = round(rounded["macro"] + (1.0 - sum(rounded.values())), 6)
    return rounded


def calculate_inter_va(
    modality_va: Mapping[str, Mapping[str, float]],
    weights: Mapping[str, float],
) -> dict[str, float]:
    """Calculate the weighted inter-personal VA and confidence result."""
    return {
        field: round(
            sum(
                float(modality_va[modality][field]) * float(weights[modality])
                for modality in MODALITIES
            ),
            6,
        )
        for field in ("valence", "arousal", "confidence")
    }
