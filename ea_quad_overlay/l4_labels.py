"""Contracts and deterministic calculations for M1 L4 annotations."""

from __future__ import annotations

import math
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any


MODALITIES = ("text", "speech", "macro", "micro")
SOURCE_DATASETS = ("CH-SIMS", "MELD")
CONSISTENT_DISTANCE_THRESHOLD = 0.35
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

REQUIRED_FIELDS = {
    "schema_version",
    "ea_id",
    "segment_id",
    "source_dataset",
    "modality_va",
    "inter_va",
    "contradiction_type",
    "involved_modalities",
    "fusion_weights",
    "reason",
    "annotation_meta",
}
VA_FIELDS = ("valence", "arousal", "confidence")
ALLOWED_EVIDENCE = {
    "source_annotation",
    "raw_audio",
    "raw_video",
    "issue_4_quality_metadata",
}
EA_ID_RE = re.compile(r"^EAQ\d{6}$")


class L4ValidationError(ValueError):
    """Raised when an L4 annotation violates the M1 contract."""


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _quantize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Round weights to six places while preserving bounds and an exact unit sum."""
    scale = 1_000_000
    total = sum(float(weights[modality]) for modality in MODALITIES)
    scaled = {
        modality: float(weights[modality]) / total * scale for modality in MODALITIES
    }
    units = {modality: math.floor(scaled[modality]) for modality in MODALITIES}
    remaining = scale - sum(units.values())
    ranked = sorted(
        MODALITIES,
        key=lambda modality: (
            scaled[modality] - units[modality],
            scaled[modality],
        ),
        reverse=True,
    )
    for index in range(remaining):
        units[ranked[index]] += 1
    return {modality: units[modality] / scale for modality in MODALITIES}


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

    return _quantize_weights(weights)


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


def validate_annotation(
    label: Mapping[str, Any],
    expected_ea_id: str,
    expected_dataset: str,
) -> None:
    """Validate one annotation against its source-index identity and policy."""
    if not isinstance(label, Mapping):
        raise L4ValidationError("annotation must be an object")

    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - set(label))
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")

    if label.get("schema_version") != "m1-l4-gold-v1":
        errors.append("schema_version must be m1-l4-gold-v1")

    ea_id = label.get("ea_id")
    if not isinstance(ea_id, str) or not EA_ID_RE.fullmatch(ea_id):
        errors.append("ea_id must match ^EAQ[0-9]{6}$")
    if ea_id != expected_ea_id:
        errors.append(f"ea_id must be {expected_ea_id}")
    if label.get("segment_id") != f"{expected_ea_id}_seg001":
        errors.append(f"segment_id must be {expected_ea_id}_seg001")
    source_dataset = label.get("source_dataset")
    if source_dataset not in SOURCE_DATASETS:
        errors.append("source_dataset must be CH-SIMS or MELD")
    if source_dataset != expected_dataset:
        errors.append(f"source_dataset must be {expected_dataset}")

    modality_va = label.get("modality_va")
    valid_va_shape = isinstance(modality_va, Mapping) and set(modality_va) == set(
        MODALITIES
    )
    if not valid_va_shape:
        errors.append("modality_va must contain exactly text, speech, macro, and micro")
    else:
        for modality in MODALITIES:
            values = modality_va[modality]
            if not isinstance(values, Mapping) or set(values) != set(VA_FIELDS):
                errors.append(
                    f"modality_va.{modality} must contain valence, arousal, and confidence"
                )
                valid_va_shape = False
                continue
            for field in VA_FIELDS:
                value = values[field]
                low, high = (-1.0, 1.0) if field != "confidence" else (0.0, 1.0)
                if not _is_number(value) or not low <= value <= high:
                    errors.append(
                        f"modality_va.{modality}.{field} must be numeric in [{low}, {high}]"
                    )
                    valid_va_shape = False

    contradiction_type = label.get("contradiction_type")
    if contradiction_type not in CONTRADICTION_TYPES:
        errors.append("invalid contradiction_type")

    involved = label.get("involved_modalities")
    if (
        not isinstance(involved, list)
        or any(not isinstance(modality, str) for modality in involved)
        or any(modality not in MODALITIES for modality in involved)
        or len(involved) != len(set(involved))
    ):
        errors.append("involved_modalities must be a unique list of known modalities")
    elif contradiction_type == "consistent" and involved:
        errors.append("consistent requires empty involved_modalities")
    elif contradiction_type in CONTRADICTION_TYPES[1:] and not involved:
        errors.append("non-consistent labels require involved_modalities")

    if valid_va_shape and contradiction_type == "consistent":
        confident_modalities = [
            modality
            for modality in MODALITIES
            if float(modality_va[modality]["confidence"]) >= 0.50
        ]
        confident_valences = [
            float(modality_va[modality]["valence"])
            for modality in confident_modalities
        ]
        if confident_valences and min(confident_valences) < 0 < max(
            confident_valences
        ):
            errors.append("consistent has opposing confident valences")

        exceeds_distance = any(
            math.dist(
                (
                    float(modality_va[left]["valence"]),
                    float(modality_va[left]["arousal"]),
                ),
                (
                    float(modality_va[right]["valence"]),
                    float(modality_va[right]["arousal"]),
                ),
            )
            > CONSISTENT_DISTANCE_THRESHOLD + 1e-12
            for left_index, left in enumerate(confident_modalities)
            for right in confident_modalities[left_index + 1 :]
        )
        if exceeds_distance:
            errors.append("consistent pairwise VA distance exceeds 0.35")

    weights = label.get("fusion_weights")
    valid_weights = isinstance(weights, Mapping) and set(weights) == set(MODALITIES)
    if not valid_weights:
        errors.append("fusion_weights must contain exactly text, speech, macro, and micro")
    else:
        for modality in MODALITIES:
            value = weights[modality]
            if not _is_number(value) or not 0.0 <= value <= 1.0:
                errors.append(
                    f"fusion_weights.{modality} must be numeric in [0.0, 1.0]"
                )
                valid_weights = False
        if valid_weights and not math.isclose(
            sum(float(weights[modality]) for modality in MODALITIES),
            1.0,
            abs_tol=1e-6,
        ):
            errors.append("fusion_weights must sum to 1")
            valid_weights = False

    meta = label.get("annotation_meta")
    valid_meta = isinstance(meta, Mapping)
    if not valid_meta:
        errors.append("annotation_meta must be an object")
        meta = {}
    else:
        if meta.get("method") != "evidence_triangulation_single_pass":
            errors.append("annotation_meta.method is invalid")
        if meta.get("review_status") != "single_pass_pending_second_review":
            errors.append("annotation_meta.review_status is invalid")
        if meta.get("micro_review_status") != "pending_issue_5":
            errors.append("annotation_meta.micro_review_status is invalid")
        evidence = meta.get("evidence")
        if (
            not isinstance(evidence, list)
            or not evidence
            or any(not isinstance(token, str) for token in evidence)
            or any(token not in ALLOWED_EVIDENCE for token in evidence)
        ):
            errors.append("annotation_meta contains an invalid evidence token")

    if valid_va_shape and valid_weights and contradiction_type in CONTRADICTION_TYPES:
        confidences = {
            modality: float(modality_va[modality]["confidence"])
            for modality in MODALITIES
        }
        try:
            expected_weights = calculate_fusion_weights(
                confidences,
                contradiction_type,
                str(meta.get("micro_review_status", "")),
            )
        except ValueError as exc:
            errors.append(f"fusion weight policy error: {exc}")
        else:
            if any(
                not math.isclose(
                    float(weights[modality]),
                    expected_weights[modality],
                    abs_tol=1e-6,
                )
                for modality in MODALITIES
            ):
                errors.append("fusion_weights do not match deterministic policy")

        expected_inter = calculate_inter_va(modality_va, weights)
        inter_va = label.get("inter_va")
        if not isinstance(inter_va, Mapping) or set(inter_va) != set(VA_FIELDS):
            errors.append("inter_va must contain valence, arousal, and confidence")
        else:
            for field in VA_FIELDS:
                value = inter_va[field]
                low, high = (-1.0, 1.0) if field != "confidence" else (0.0, 1.0)
                if not _is_number(value) or not low <= value <= high:
                    errors.append(f"inter_va.{field} must be numeric in [{low}, {high}]")
                elif not math.isclose(
                    float(value), expected_inter[field], abs_tol=1e-6
                ):
                    errors.append(
                        f"inter_va.{field} does not match weighted modalities"
                    )

        if meta.get("micro_review_status") == "pending_issue_5":
            if float(modality_va["micro"]["confidence"]) > 0.60:
                errors.append("pending micro confidence exceeds 0.60")
            if float(weights["micro"]) > 0.10:
                errors.append("pending micro fusion weight exceeds 0.10")

    reason = label.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        errors.append("reason must be non-empty")

    if errors:
        raise L4ValidationError("; ".join(errors))


def validate_dataset(
    index_rows: Sequence[Mapping[str, str]],
    annotations_dir: Path,
) -> list[dict[str, Any]]:
    """Validate exact source-index coverage and return labels in index order."""
    source_ids = [row["ea_id"] for row in index_rows]
    duplicates = sorted(
        ea_id for ea_id, count in Counter(source_ids).items() if count > 1
    )
    if duplicates:
        raise L4ValidationError(f"duplicate source-index ea_id: {', '.join(duplicates)}")

    expected = {
        f"{row['ea_id']}_seg001_l4_gold.json": row for row in index_rows
    }
    actual = {
        path.name: path for path in annotations_dir.glob("*_seg001_l4_gold.json")
    }
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing annotation: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected annotation: {', '.join(unexpected)}")
        raise L4ValidationError("; ".join(details))

    labels: list[dict[str, Any]] = []
    for filename, row in expected.items():
        path = actual[filename]
        try:
            label = json.loads(path.read_text(encoding="utf-8"))
            validate_annotation(label, row["ea_id"], row["source_dataset"])
        except (json.JSONDecodeError, L4ValidationError) as exc:
            raise L4ValidationError(f"{filename}: {exc}") from exc
        labels.append(label)
    return labels


def summarize_annotations(labels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return stable aggregate statistics for validated annotations."""
    datasets = Counter(str(label["source_dataset"]) for label in labels)
    contradiction_types = Counter(str(label["contradiction_type"]) for label in labels)

    def mean_for(section: str, modality: str, field: str | None = None) -> float:
        if not labels:
            return 0.0
        if field is None:
            values = [float(label[section][modality]) for label in labels]
        else:
            values = [float(label[section][modality][field]) for label in labels]
        return round(fmean(values), 6)

    return {
        "total": len(labels),
        "datasets": dict(datasets),
        "contradiction_types": dict(contradiction_types),
        "mean_confidence": {
            modality: mean_for("modality_va", modality, "confidence")
            for modality in MODALITIES
        },
        "mean_weight": {
            modality: mean_for("fusion_weights", modality) for modality in MODALITIES
        },
        "low_confidence_ids": [
            str(label["ea_id"])
            for label in labels
            if float(label["inter_va"]["confidence"]) < 0.60
        ],
        "pending_micro_review": sum(
            label["annotation_meta"]["micro_review_status"] == "pending_issue_5"
            for label in labels
        ),
    }
