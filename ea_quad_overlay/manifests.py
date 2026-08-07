"""M1 manifest contracts, generation, and repository read/validation logic."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ea_quad_overlay.l4_labels import CONTRADICTION_TYPES, MODALITIES, VA_FIELDS

L2_SCHEMA_VERSION = "m1-l2-manifest-v1"
FUSION_SCHEMA_VERSION = "m1-fusion-manifest-v1"
EA_ID_RE = re.compile(r"^EAQ\d{6}$")
SEG_ID_RE = re.compile(r"^EAQ\d{6}_seg\d{3}$")

L2_REQUIRED_FIELDS = {
    "schema_version",
    "ea_id",
    "segment_id",
    "modality",
    "source_dataset",
    "source_id",
    "source_split",
    "feature_path",
    "meta_path",
    "shape",
    "dtype",
    "status",
    "model",
}

FUSION_REQUIRED_FIELDS = {
    "schema_version",
    "ea_id",
    "segment_id",
    "source_dataset",
    "source_id",
    "source_split",
    "feature_paths",
    "meta_paths",
    "l4_gold_path",
    "micro_review_path",
    "modality_va",
    "inter_va",
    "contradiction_type",
    "involved_modalities",
    "fusion_weights",
    "reason",
    "micro_review_status",
    "has_micro_expression",
}

L2_MANIFEST_NAMES = {
    "text": "l2_text_m1.jsonl",
    "speech": "l2_speech_m1.jsonl",
    "macro": "l2_macro_m1.jsonl",
    "micro": "l2_micro_m1.jsonl",
}
FUSION_MANIFEST_NAME = "fusion_segments_m1.jsonl"


class ManifestError(ValueError):
    """Raised when a manifest record or file violates the M1 contract."""


def segment_id_for(ea_id: str, seg: int = 1) -> str:
    return f"{ea_id}_seg{seg:03d}"


def read_source_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"ea_id", "source_dataset", "source_id", "source_split"}
        if not required <= fields:
            missing = ", ".join(sorted(required - fields))
            raise ManifestError(f"source index missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ManifestError(f"source index is empty: {path}")
    return rows


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestError(f"expected object JSON in {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Repository read path for L2 / fusion manifests."""
    if not path.is_file():
        raise ManifestError(f"manifest file missing: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(
                    f"invalid JSONL at {path}:{line_no}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ManifestError(f"expected object at {path}:{line_no}")
            records.append(payload)
    return records


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _require_string(record: Mapping[str, Any], field: str, *, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{context}: missing/invalid string field `{field}`")
    return value


def _require_path_exists(root: Path, rel: str, *, context: str) -> Path:
    path = root / rel
    if not path.is_file() or path.stat().st_size <= 0:
        raise ManifestError(f"{context}: path missing or empty: {rel}")
    return path


def _validate_ids(ea_id: str, segment_id: str, *, context: str) -> None:
    if not EA_ID_RE.fullmatch(ea_id):
        raise ManifestError(f"{context}: invalid ea_id `{ea_id}`")
    if not SEG_ID_RE.fullmatch(segment_id):
        raise ManifestError(f"{context}: invalid segment_id `{segment_id}`")
    if not segment_id.startswith(f"{ea_id}_"):
        raise ManifestError(
            f"{context}: segment_id `{segment_id}` does not match ea_id `{ea_id}`"
        )


def _validate_va_block(block: object, *, context: str) -> None:
    if not isinstance(block, Mapping):
        raise ManifestError(f"{context}: expected object")
    for field in VA_FIELDS:
        value = block.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ManifestError(f"{context}: `{field}` must be a number")
        if field == "confidence":
            if not 0.0 <= float(value) <= 1.0:
                raise ManifestError(f"{context}: confidence out of range")
        elif not -1.0 <= float(value) <= 1.0:
            raise ManifestError(f"{context}: {field} out of range")


def build_l2_record(
    index_row: Mapping[str, str],
    modality: str,
    feature_meta: Mapping[str, Any],
) -> dict[str, Any]:
    if modality not in MODALITIES:
        raise ManifestError(f"unsupported modality: {modality}")
    ea_id = index_row["ea_id"]
    segment_id = str(feature_meta.get("segment_id") or segment_id_for(ea_id))
    feature_path = str(feature_meta.get("feature_path") or "")
    if not feature_path:
        raise ManifestError(f"{ea_id}/{modality}: feature_path missing in meta")
    return {
        "schema_version": L2_SCHEMA_VERSION,
        "ea_id": ea_id,
        "segment_id": segment_id,
        "modality": modality,
        "source_dataset": index_row["source_dataset"],
        "source_id": index_row["source_id"],
        "source_split": index_row["source_split"],
        "feature_path": feature_path.replace("\\", "/"),
        "meta_path": f"features/{modality}/{segment_id}_{modality}.json",
        "shape": list(feature_meta.get("shape") or []),
        "dtype": str(feature_meta.get("dtype") or "float32"),
        "status": str(feature_meta.get("status") or "unknown"),
        "model": str(feature_meta.get("model") or ""),
    }


def build_fusion_record(
    index_row: Mapping[str, str],
    feature_metas: Mapping[str, Mapping[str, Any]],
    l4_label: Mapping[str, Any],
    micro_review: Mapping[str, Any],
) -> dict[str, Any]:
    ea_id = index_row["ea_id"]
    segment_id = str(l4_label.get("segment_id") or segment_id_for(ea_id))
    feature_paths = {
        modality: str(feature_metas[modality]["feature_path"]).replace("\\", "/")
        for modality in MODALITIES
    }
    meta_paths = {
        modality: f"features/{modality}/{segment_id}_{modality}.json"
        for modality in MODALITIES
    }
    return {
        "schema_version": FUSION_SCHEMA_VERSION,
        "ea_id": ea_id,
        "segment_id": segment_id,
        "source_dataset": index_row["source_dataset"],
        "source_id": index_row["source_id"],
        "source_split": index_row["source_split"],
        "feature_paths": feature_paths,
        "meta_paths": meta_paths,
        "l4_gold_path": f"annotations/l4_gold/{segment_id}_l4_gold.json",
        "micro_review_path": f"annotations/micro_review/{segment_id}_micro_review.json",
        "modality_va": l4_label["modality_va"],
        "inter_va": l4_label["inter_va"],
        "contradiction_type": l4_label["contradiction_type"],
        "involved_modalities": list(l4_label.get("involved_modalities") or []),
        "fusion_weights": l4_label["fusion_weights"],
        "reason": l4_label["reason"],
        "micro_review_status": micro_review.get("review_status")
        or l4_label.get("annotation_meta", {}).get("micro_review_status"),
        "has_micro_expression": bool(micro_review.get("has_micro_expression", False)),
    }


def validate_l2_record(
    record: Mapping[str, Any],
    *,
    expected_modality: str,
    root: Path,
    index_by_id: Mapping[str, Mapping[str, str]] | None = None,
) -> None:
    context = f"l2/{expected_modality}"
    missing = L2_REQUIRED_FIELDS - set(record)
    if missing:
        raise ManifestError(f"{context}: missing fields {sorted(missing)}")
    if record["schema_version"] != L2_SCHEMA_VERSION:
        raise ManifestError(f"{context}: unexpected schema_version")
    ea_id = _require_string(record, "ea_id", context=context)
    segment_id = _require_string(record, "segment_id", context=context)
    _validate_ids(ea_id, segment_id, context=context)
    modality = _require_string(record, "modality", context=context)
    if modality != expected_modality:
        raise ManifestError(
            f"{context}: modality `{modality}` does not match file modality"
        )
    feature_path = _require_string(record, "feature_path", context=context)
    meta_path = _require_string(record, "meta_path", context=context)
    _require_path_exists(root, feature_path, context=f"{context}/{ea_id}")
    _require_path_exists(root, meta_path, context=f"{context}/{ea_id}")
    if not isinstance(record["shape"], list) or not record["shape"]:
        raise ManifestError(f"{context}/{ea_id}: shape must be a non-empty list")
    if index_by_id is not None:
        if ea_id not in index_by_id:
            raise ManifestError(f"{context}/{ea_id}: not present in source index")
        index_row = index_by_id[ea_id]
        for field in ("source_dataset", "source_id", "source_split"):
            if record[field] != index_row[field]:
                raise ManifestError(
                    f"{context}/{ea_id}: `{field}` mismatch with source index"
                )


def validate_fusion_record(
    record: Mapping[str, Any],
    *,
    root: Path,
    index_by_id: Mapping[str, Mapping[str, str]] | None = None,
) -> None:
    context = "fusion"
    missing = FUSION_REQUIRED_FIELDS - set(record)
    if missing:
        raise ManifestError(f"{context}: missing fields {sorted(missing)}")
    if record["schema_version"] != FUSION_SCHEMA_VERSION:
        raise ManifestError(f"{context}: unexpected schema_version")
    ea_id = _require_string(record, "ea_id", context=context)
    segment_id = _require_string(record, "segment_id", context=context)
    _validate_ids(ea_id, segment_id, context=context)

    feature_paths = record["feature_paths"]
    meta_paths = record["meta_paths"]
    if not isinstance(feature_paths, Mapping) or not isinstance(meta_paths, Mapping):
        raise ManifestError(f"{context}/{ea_id}: feature_paths/meta_paths must be objects")
    for modality in MODALITIES:
        if modality not in feature_paths or modality not in meta_paths:
            raise ManifestError(f"{context}/{ea_id}: missing modality path `{modality}`")
        _require_path_exists(
            root, str(feature_paths[modality]), context=f"{context}/{ea_id}/{modality}"
        )
        _require_path_exists(
            root, str(meta_paths[modality]), context=f"{context}/{ea_id}/{modality}"
        )

    for rel_field in ("l4_gold_path", "micro_review_path"):
        rel = _require_string(record, rel_field, context=f"{context}/{ea_id}")
        _require_path_exists(root, rel, context=f"{context}/{ea_id}")

    modality_va = record["modality_va"]
    if not isinstance(modality_va, Mapping):
        raise ManifestError(f"{context}/{ea_id}: modality_va must be an object")
    for modality in MODALITIES:
        if modality not in modality_va:
            raise ManifestError(f"{context}/{ea_id}: modality_va missing `{modality}`")
        _validate_va_block(modality_va[modality], context=f"{context}/{ea_id}/{modality}")
    _validate_va_block(record["inter_va"], context=f"{context}/{ea_id}/inter_va")

    contradiction = record["contradiction_type"]
    if contradiction not in CONTRADICTION_TYPES:
        raise ManifestError(f"{context}/{ea_id}: illegal contradiction_type")
    involved = record["involved_modalities"]
    if not isinstance(involved, list) or any(item not in MODALITIES for item in involved):
        raise ManifestError(f"{context}/{ea_id}: invalid involved_modalities")

    weights = record["fusion_weights"]
    if not isinstance(weights, Mapping):
        raise ManifestError(f"{context}/{ea_id}: fusion_weights must be an object")
    for modality in MODALITIES:
        value = weights.get(modality)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ManifestError(f"{context}/{ea_id}: weight `{modality}` must be a number")
        if float(value) < 0.0:
            raise ManifestError(f"{context}/{ea_id}: negative fusion weight")
    if abs(sum(float(weights[m]) for m in MODALITIES) - 1.0) > 1e-6:
        raise ManifestError(f"{context}/{ea_id}: fusion_weights must sum to 1")

    reason = record["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ManifestError(f"{context}/{ea_id}: reason must be non-empty")
    if not isinstance(record["micro_review_status"], str):
        raise ManifestError(f"{context}/{ea_id}: micro_review_status must be a string")
    if not isinstance(record["has_micro_expression"], bool):
        raise ManifestError(f"{context}/{ea_id}: has_micro_expression must be bool")

    if index_by_id is not None:
        if ea_id not in index_by_id:
            raise ManifestError(f"{context}/{ea_id}: not present in source index")
        index_row = index_by_id[ea_id]
        for field in ("source_dataset", "source_id", "source_split"):
            if record[field] != index_row[field]:
                raise ManifestError(
                    f"{context}/{ea_id}: `{field}` mismatch with source index"
                )


def generate_m1_manifests(
    *,
    root: Path,
    index_path: Path,
    features_root: Path | None = None,
    l4_dir: Path | None = None,
    micro_dir: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build L2 + fusion manifests from Issue #4/#5/#6 artifacts."""
    features_root = features_root or (root / "features")
    l4_dir = l4_dir or (root / "annotations" / "l4_gold")
    micro_dir = micro_dir or (root / "annotations" / "micro_review")

    rows = read_source_index(index_path)
    l2: dict[str, list[dict[str, Any]]] = {modality: [] for modality in MODALITIES}
    fusion: list[dict[str, Any]] = []

    for row in rows:
        ea_id = row["ea_id"]
        segment_id = segment_id_for(ea_id)
        feature_metas: dict[str, dict[str, Any]] = {}
        for modality in MODALITIES:
            meta_path = features_root / modality / f"{segment_id}_{modality}.json"
            if not meta_path.is_file():
                raise ManifestError(f"missing Issue #4 meta: {meta_path}")
            meta = load_json(meta_path)
            feature_metas[modality] = meta
            l2[modality].append(build_l2_record(row, modality, meta))

        l4_path = l4_dir / f"{segment_id}_l4_gold.json"
        micro_path = micro_dir / f"{segment_id}_micro_review.json"
        if not l4_path.is_file():
            raise ManifestError(f"missing Issue #6 L4 label: {l4_path}")
        if not micro_path.is_file():
            raise ManifestError(f"missing Issue #5 micro review: {micro_path}")
        fusion.append(
            build_fusion_record(row, feature_metas, load_json(l4_path), load_json(micro_path))
        )

    return {
        "l2_text": l2["text"],
        "l2_speech": l2["speech"],
        "l2_macro": l2["macro"],
        "l2_micro": l2["micro"],
        "fusion": fusion,
    }


def write_m1_manifests(manifests: Mapping[str, Sequence[Mapping[str, Any]]], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    mapping = {
        "l2_text": L2_MANIFEST_NAMES["text"],
        "l2_speech": L2_MANIFEST_NAMES["speech"],
        "l2_macro": L2_MANIFEST_NAMES["macro"],
        "l2_micro": L2_MANIFEST_NAMES["micro"],
        "fusion": FUSION_MANIFEST_NAME,
    }
    for key, filename in mapping.items():
        path = out_dir / filename
        write_jsonl(path, manifests[key])
        written[key] = path
    return written


def validate_m1_manifests(
    *,
    root: Path,
    manifest_dir: Path,
    index_path: Path,
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Read and validate all M1 manifests with repository path checks."""
    rows = read_source_index(index_path)
    index_by_id = {row["ea_id"]: row for row in rows}
    if expected_count is None:
        expected_count = len(rows)

    counts: dict[str, int] = {}
    for modality, filename in L2_MANIFEST_NAMES.items():
        records = read_jsonl(manifest_dir / filename)
        if len(records) != expected_count:
            raise ManifestError(
                f"{filename}: expected {expected_count} records, got {len(records)}"
            )
        seen: set[str] = set()
        for record in records:
            validate_l2_record(
                record,
                expected_modality=modality,
                root=root,
                index_by_id=index_by_id,
            )
            ea_id = str(record["ea_id"])
            if ea_id in seen:
                raise ManifestError(f"{filename}: duplicate ea_id `{ea_id}`")
            seen.add(ea_id)
        if seen != set(index_by_id):
            raise ManifestError(f"{filename}: ea_id set does not match source index")
        counts[modality] = len(records)

    fusion_records = read_jsonl(manifest_dir / FUSION_MANIFEST_NAME)
    if len(fusion_records) != expected_count:
        raise ManifestError(
            f"{FUSION_MANIFEST_NAME}: expected {expected_count} records, "
            f"got {len(fusion_records)}"
        )
    seen_fusion: set[str] = set()
    contradiction_counts: Counter[str] = Counter()
    micro_status_counts: Counter[str] = Counter()
    for record in fusion_records:
        validate_fusion_record(record, root=root, index_by_id=index_by_id)
        ea_id = str(record["ea_id"])
        if ea_id in seen_fusion:
            raise ManifestError(f"fusion: duplicate ea_id `{ea_id}`")
        seen_fusion.add(ea_id)
        contradiction_counts[str(record["contradiction_type"])] += 1
        micro_status_counts[str(record["micro_review_status"])] += 1
    if seen_fusion != set(index_by_id):
        raise ManifestError("fusion: ea_id set does not match source index")

    return {
        "total": expected_count,
        "l2_counts": counts,
        "fusion_count": len(fusion_records),
        "contradiction_counts": dict(contradiction_counts),
        "micro_status_counts": dict(micro_status_counts),
    }


def render_manifest_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# M1 Manifest Check",
        "",
        "GitHub issue: <https://github.com/cuiyuxun-droid/ea-quad-overlay/issues/7>",
        "",
        "## Scope and result",
        "",
        "- Generated four L2 manifests and one L4 fusion manifest for the 20 M1 samples.",
        "- Records are built from Issue #4 feature meta, Issue #5 micro reviews, and Issue #6 L4 gold labels.",
        "- Repository read/validation confirms JSONL parseability, complete fusion fields, and existing `feature_path` files.",
        "",
        "## Manifest files",
        "",
        "| File | Records |",
        "| --- | ---: |",
    ]
    for modality in MODALITIES:
        lines.append(
            f"| `manifests/{L2_MANIFEST_NAMES[modality]}` | {summary['l2_counts'][modality]} |"
        )
    lines.append(f"| `manifests/{FUSION_MANIFEST_NAME}` | {summary['fusion_count']} |")
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Samples validated: **{summary['total']}**",
            "- All L2 `feature_path` and `meta_path` files exist.",
            "- All fusion `feature_paths`, `l4_gold_path`, and `micro_review_path` files exist.",
            "- Fusion required fields are complete; `fusion_weights` sum to 1.",
            "",
            "## Fusion contradiction distribution",
            "",
            "| Type | Count |",
            "| --- | ---: |",
        ]
    )
    for key in CONTRADICTION_TYPES:
        lines.append(f"| `{key}` | {summary['contradiction_counts'].get(key, 0)} |")
    lines.extend(
        [
            "",
            "## Micro-review status in fusion manifest",
            "",
            "| Status | Count |",
            "| --- | ---: |",
        ]
    )
    for key, count in sorted(summary["micro_status_counts"].items()):
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## Errors", "", "- none", ""])
    return "\n".join(lines)
