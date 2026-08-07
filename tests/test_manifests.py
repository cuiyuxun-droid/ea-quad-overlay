from __future__ import annotations

import json
from pathlib import Path

import pytest

from ea_quad_overlay.manifests import (
    FUSION_SCHEMA_VERSION,
    L2_SCHEMA_VERSION,
    ManifestError,
    build_fusion_record,
    build_l2_record,
    generate_m1_manifests,
    read_jsonl,
    validate_fusion_record,
    validate_l2_record,
    validate_m1_manifests,
    write_jsonl,
    write_m1_manifests,
)


def _write_feature(
    root: Path,
    ea_id: str,
    modality: str,
    *,
    shape: list[int],
) -> dict:
    segment_id = f"{ea_id}_seg001"
    feature_rel = f"features/{modality}/{segment_id}_{modality}.npy"
    meta_rel = f"features/{modality}/{segment_id}_{modality}.json"
    (root / feature_rel).parent.mkdir(parents=True, exist_ok=True)
    (root / feature_rel).write_bytes(b"\x00\x01")
    meta = {
        "ea_id": ea_id,
        "segment_id": segment_id,
        "modality": modality,
        "shape": shape,
        "dtype": "float32",
        "feature_path": feature_rel,
        "model": f"dummy-{modality}",
        "status": "ok",
    }
    (root / meta_rel).write_text(json.dumps(meta), encoding="utf-8")
    return meta


def _write_annotations(root: Path, ea_id: str) -> tuple[dict, dict]:
    segment_id = f"{ea_id}_seg001"
    l4 = {
        "schema_version": "m1-l4-gold-v1",
        "ea_id": ea_id,
        "segment_id": segment_id,
        "source_dataset": "CH-SIMS",
        "modality_va": {
            "text": {"valence": 0.1, "arousal": 0.2, "confidence": 0.9},
            "speech": {"valence": 0.0, "arousal": 0.1, "confidence": 0.8},
            "macro": {"valence": 0.05, "arousal": 0.15, "confidence": 0.7},
            "micro": {"valence": 0.0, "arousal": 0.0, "confidence": 0.0},
        },
        "inter_va": {"valence": 0.05, "arousal": 0.15, "confidence": 0.8},
        "contradiction_type": "consistent",
        "involved_modalities": [],
        "fusion_weights": {
            "text": 0.4,
            "speech": 0.3,
            "macro": 0.3,
            "micro": 0.0,
        },
        "reason": "fixture reason",
        "annotation_meta": {"micro_review_status": "negative"},
    }
    micro = {
        "ea_id": ea_id,
        "segment_id": segment_id,
        "review_status": "negative",
        "has_micro_expression": False,
    }
    l4_path = root / "annotations" / "l4_gold" / f"{segment_id}_l4_gold.json"
    micro_path = (
        root / "annotations" / "micro_review" / f"{segment_id}_micro_review.json"
    )
    l4_path.parent.mkdir(parents=True, exist_ok=True)
    micro_path.parent.mkdir(parents=True, exist_ok=True)
    l4_path.write_text(json.dumps(l4), encoding="utf-8")
    micro_path.write_text(json.dumps(micro), encoding="utf-8")
    return l4, micro


def test_build_l2_and_fusion_records_include_required_fields(tmp_path: Path) -> None:
    ea_id = "EAQ000001"
    index_row = {
        "ea_id": ea_id,
        "source_dataset": "CH-SIMS",
        "source_id": "CH-SIMS/x",
        "source_split": "train",
    }
    metas = {
        modality: _write_feature(tmp_path, ea_id, modality, shape=[4])
        for modality in ("text", "speech", "macro", "micro")
    }
    l4, micro = _write_annotations(tmp_path, ea_id)

    l2 = build_l2_record(index_row, "text", metas["text"])
    fusion = build_fusion_record(index_row, metas, l4, micro)

    assert l2["schema_version"] == L2_SCHEMA_VERSION
    assert l2["feature_path"].endswith("_text.npy")
    assert fusion["schema_version"] == FUSION_SCHEMA_VERSION
    assert set(fusion["feature_paths"]) == {"text", "speech", "macro", "micro"}
    validate_l2_record(l2, expected_modality="text", root=tmp_path)
    validate_fusion_record(fusion, root=tmp_path)


def test_missing_feature_path_is_rejected(tmp_path: Path) -> None:
    record = {
        "schema_version": L2_SCHEMA_VERSION,
        "ea_id": "EAQ000001",
        "segment_id": "EAQ000001_seg001",
        "modality": "text",
        "source_dataset": "CH-SIMS",
        "source_id": "x",
        "source_split": "train",
        "feature_path": "features/text/missing.npy",
        "meta_path": "features/text/missing.json",
        "shape": [1],
        "dtype": "float32",
        "status": "ok",
        "model": "dummy",
    }
    with pytest.raises(ManifestError, match="path missing"):
        validate_l2_record(record, expected_modality="text", root=tmp_path)


def test_generate_and_validate_roundtrip(tmp_path: Path) -> None:
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "ea_id,source_dataset,source_split,source_id\n"
        "EAQ000001,CH-SIMS,train,CH-SIMS/a\n"
        "EAQ000002,MELD,test,MELD/b\n",
        encoding="utf-8",
    )
    for ea_id in ("EAQ000001", "EAQ000002"):
        for modality, shape in (
            ("text", [2]),
            ("speech", [3]),
            ("macro", [4]),
            ("micro", [5]),
        ):
            _write_feature(tmp_path, ea_id, modality, shape=shape)
        _write_annotations(tmp_path, ea_id)

    manifests = generate_m1_manifests(root=tmp_path, index_path=index_path)
    out_dir = tmp_path / "manifests"
    write_m1_manifests(manifests, out_dir)
    summary = validate_m1_manifests(
        root=tmp_path,
        manifest_dir=out_dir,
        index_path=index_path,
    )

    assert summary["total"] == 2
    assert summary["l2_counts"] == {
        "text": 2,
        "speech": 2,
        "macro": 2,
        "micro": 2,
    }
    assert len(read_jsonl(out_dir / "fusion_segments_m1.jsonl")) == 2


def test_incomplete_fusion_fields_are_rejected(tmp_path: Path) -> None:
    write_jsonl(
        tmp_path / "bad.jsonl",
        [{"schema_version": FUSION_SCHEMA_VERSION, "ea_id": "EAQ000001"}],
    )
    records = read_jsonl(tmp_path / "bad.jsonl")
    with pytest.raises(ManifestError, match="missing fields"):
        validate_fusion_record(records[0], root=tmp_path)
