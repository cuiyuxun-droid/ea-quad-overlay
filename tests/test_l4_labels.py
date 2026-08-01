from __future__ import annotations

import pytest

from ea_quad_overlay.l4_labels import (
    L4ValidationError,
    calculate_fusion_weights,
    calculate_inter_va,
    validate_annotation,
)
from tests.l4_test_data import make_valid_label


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


def test_rounding_never_creates_a_negative_weight() -> None:
    weights = calculate_fusion_weights(
        {"text": 0.0140625, "speech": 0.8859375, "macro": 0.0, "micro": 0.2},
        "consistent",
        "pending_issue_5",
    )

    assert all(weight >= 0.0 for weight in weights.values())
    assert sum(weights.values()) == 1.0


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


def test_valid_annotation_passes() -> None:
    validate_annotation(make_valid_label(), "EAQ000001", "CH-SIMS")


def test_annotation_root_must_be_an_object() -> None:
    with pytest.raises(L4ValidationError, match="annotation must be an object"):
        validate_annotation([], "EAQ000001", "CH-SIMS")  # type: ignore[arg-type]


def test_missing_field_is_reported_without_internal_error() -> None:
    label = make_valid_label()
    del label["inter_va"]

    with pytest.raises(L4ValidationError, match="missing fields: inter_va"):
        validate_annotation(label, "EAQ000001", "CH-SIMS")


def test_non_consistent_label_requires_involved_modalities() -> None:
    label = make_valid_label()
    label["contradiction_type"] = "sarcasm"

    with pytest.raises(L4ValidationError, match="require involved_modalities"):
        validate_annotation(label, "EAQ000001", "CH-SIMS")


def test_consistent_label_rejects_involved_modalities() -> None:
    label = make_valid_label()
    label["involved_modalities"] = ["text", "speech"]

    with pytest.raises(L4ValidationError, match="consistent requires empty"):
        validate_annotation(label, "EAQ000001", "CH-SIMS")


@pytest.mark.parametrize(
    ("modality", "field", "value"),
    [
        ("text", "valence", 1.01),
        ("speech", "arousal", -1.01),
        ("macro", "confidence", 1.01),
        ("micro", "confidence", True),
    ],
)
def test_modality_ranges_and_boolean_values_are_rejected(
    modality: str,
    field: str,
    value: float | bool,
) -> None:
    label = make_valid_label()
    label["modality_va"][modality][field] = value

    with pytest.raises(L4ValidationError, match=rf"{modality}\.{field}"):
        validate_annotation(label, "EAQ000001", "CH-SIMS")


def test_duplicate_and_unknown_involved_modalities_are_rejected() -> None:
    label = make_valid_label()
    label["contradiction_type"] = "masking"
    label["involved_modalities"] = ["text", "text", "body"]

    with pytest.raises(L4ValidationError, match="unique.*known modalities"):
        validate_annotation(label, "EAQ000001", "CH-SIMS")


def test_unhashable_involved_modality_is_reported_as_validation_error() -> None:
    label = make_valid_label()
    label["contradiction_type"] = "masking"
    label["involved_modalities"] = [{}]

    with pytest.raises(L4ValidationError, match="unique list of known modalities"):
        validate_annotation(label, "EAQ000001", "CH-SIMS")


def test_weight_sum_is_enforced() -> None:
    label = make_valid_label()
    label["fusion_weights"]["text"] = 0.35

    with pytest.raises(L4ValidationError, match="sum to 1"):
        validate_annotation(label, "EAQ000001", "CH-SIMS")


def test_weights_must_match_the_deterministic_policy() -> None:
    label = make_valid_label()
    label["fusion_weights"] = {
        "text": 0.34,
        "speech": 0.34,
        "macro": 0.32,
        "micro": 0.0,
    }
    label["inter_va"] = {"valence": 0.15, "arousal": 0.134, "confidence": 0.834}

    with pytest.raises(L4ValidationError, match="deterministic policy"):
        validate_annotation(label, "EAQ000001", "CH-SIMS")


def test_inter_va_must_match_weighted_modalities() -> None:
    label = make_valid_label()
    label["inter_va"]["valence"] = 0.0

    with pytest.raises(L4ValidationError, match=r"inter_va\.valence"):
        validate_annotation(label, "EAQ000001", "CH-SIMS")


def test_pending_micro_caps_are_enforced() -> None:
    label = make_valid_label()
    label["modality_va"]["micro"]["confidence"] = 0.61

    with pytest.raises(L4ValidationError, match="micro confidence"):
        validate_annotation(label, "EAQ000001", "CH-SIMS")


def test_reason_must_be_non_empty() -> None:
    label = make_valid_label()
    label["reason"] = "  "

    with pytest.raises(L4ValidationError, match="reason must be non-empty"):
        validate_annotation(label, "EAQ000001", "CH-SIMS")


def test_identity_dataset_and_contradiction_type_are_enforced() -> None:
    label = make_valid_label()
    label["ea_id"] = "EAQ000099"
    label["source_dataset"] = "OTHER"
    label["contradiction_type"] = "unknown"

    with pytest.raises(
        L4ValidationError,
        match="ea_id must be EAQ000001.*source_dataset must be CH-SIMS.*invalid contradiction_type",
    ):
        validate_annotation(label, "EAQ000001", "CH-SIMS")


def test_dataset_enum_is_enforced_even_when_it_matches_the_index() -> None:
    label = make_valid_label()
    label["source_dataset"] = "OTHER"

    with pytest.raises(L4ValidationError, match="source_dataset must be CH-SIMS or MELD"):
        validate_annotation(label, "EAQ000001", "OTHER")


def _refresh_inter_va(label: dict) -> None:
    label["inter_va"] = calculate_inter_va(
        label["modality_va"],
        label["fusion_weights"],
    )


def test_consistent_rejects_opposing_confident_valences() -> None:
    label = make_valid_label()
    label["modality_va"]["text"]["valence"] = 0.6
    label["modality_va"]["speech"]["valence"] = -0.6
    _refresh_inter_va(label)

    with pytest.raises(L4ValidationError, match="opposing confident valences"):
        validate_annotation(label, "EAQ000001", "CH-SIMS")


def test_consistent_rejects_pairwise_va_distance_above_threshold() -> None:
    label = make_valid_label()
    label["modality_va"]["text"].update(valence=0.1, arousal=0.1)
    label["modality_va"]["speech"].update(valence=0.5, arousal=0.1)
    _refresh_inter_va(label)

    with pytest.raises(L4ValidationError, match="pairwise VA distance exceeds 0.35"):
        validate_annotation(label, "EAQ000001", "CH-SIMS")


def test_consistent_accepts_pairwise_va_distance_at_threshold() -> None:
    label = make_valid_label()
    label["modality_va"]["text"].update(valence=0.1, arousal=0.1)
    label["modality_va"]["speech"].update(valence=0.45, arousal=0.1)
    label["modality_va"]["macro"].update(valence=0.2, arousal=0.1)
    _refresh_inter_va(label)

    validate_annotation(label, "EAQ000001", "CH-SIMS")


def test_annotation_metadata_requires_known_evidence_tokens() -> None:
    label = make_valid_label()
    label["annotation_meta"]["evidence"] = ["source_annotation", "invented_signal"]

    with pytest.raises(L4ValidationError, match="invalid evidence token"):
        validate_annotation(label, "EAQ000001", "CH-SIMS")
