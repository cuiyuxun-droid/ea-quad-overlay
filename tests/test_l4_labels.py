from __future__ import annotations

import pytest

from ea_quad_overlay.l4_labels import calculate_fusion_weights, calculate_inter_va


def test_consistent_weights_follow_confidence_and_sum_to_one() -> None:
    weights = calculate_fusion_weights(
        {"text": 0.9, "speech": 0.8, "macro": 0.8, "micro": 0.0},
        "consistent",
        "pending_issue_5",
    )

    assert weights == {
        "text": 0.36,
        "speech": 0.32,
        "macro": 0.32,
        "micro": 0.0,
    }
    assert sum(weights.values()) == 1.0


def test_pending_micro_weight_is_capped_and_excess_is_redistributed() -> None:
    weights = calculate_fusion_weights(
        {"text": 0.4, "speech": 0.4, "macro": 0.4, "micro": 0.6},
        "hidden_emotion",
        "pending_issue_5",
    )

    assert weights["micro"] == 0.1
    assert abs(sum(weights.values()) - 1.0) <= 1e-6


def test_inter_va_is_weighted_and_rounded_to_six_places() -> None:
    modality_va = {
        "text": {"valence": 0.2, "arousal": 0.1, "confidence": 0.9},
        "speech": {"valence": 0.1, "arousal": 0.2, "confidence": 0.8},
        "macro": {"valence": 0.15, "arousal": 0.1, "confidence": 0.8},
        "micro": {"valence": 0.0, "arousal": 0.0, "confidence": 0.0},
    }

    result = calculate_inter_va(
        modality_va,
        {"text": 0.36, "speech": 0.32, "macro": 0.32, "micro": 0.0},
    )

    assert result == {"valence": 0.152, "arousal": 0.132, "confidence": 0.836}


def test_zero_confidences_are_rejected() -> None:
    with pytest.raises(ValueError, match="all raw fusion weights are zero"):
        calculate_fusion_weights(
            {"text": 0.0, "speech": 0.0, "macro": 0.0, "micro": 0.0},
            "consistent",
            "pending_issue_5",
        )


def test_unknown_contradiction_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported contradiction_type"):
        calculate_fusion_weights(
            {"text": 1.0, "speech": 1.0, "macro": 1.0, "micro": 0.0},
            "unknown",
            "pending_issue_5",
        )
