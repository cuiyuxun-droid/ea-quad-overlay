from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ea_quad_overlay.l4_labels import (
    L4ValidationError,
    summarize_annotations,
    validate_dataset,
)
from tests.l4_test_data import make_valid_label


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "validate_m1_l4_labels.py"


def write_complete_temporary_dataset(tmp_path: Path) -> tuple[Path, Path]:
    index_path = tmp_path / "index.csv"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ea_id", "source_dataset"])
        writer.writeheader()
        writer.writerow({"ea_id": "EAQ000001", "source_dataset": "CH-SIMS"})
    (labels_dir / "EAQ000001_seg001_l4_gold.json").write_text(
        json.dumps(make_valid_label()),
        encoding="utf-8",
    )
    return index_path, labels_dir


def test_dataset_rejects_missing_annotation(tmp_path: Path) -> None:
    rows = [{"ea_id": "EAQ000001", "source_dataset": "CH-SIMS"}]

    with pytest.raises(L4ValidationError, match="missing annotation"):
        validate_dataset(rows, tmp_path)


def test_dataset_rejects_unexpected_annotation(tmp_path: Path) -> None:
    (tmp_path / "EAQ999999_seg001_l4_gold.json").write_text("{}", encoding="utf-8")

    with pytest.raises(L4ValidationError, match="unexpected annotation"):
        validate_dataset([], tmp_path)


def test_dataset_prefixes_invalid_file_errors(tmp_path: Path) -> None:
    rows = [{"ea_id": "EAQ000001", "source_dataset": "CH-SIMS"}]
    filename = "EAQ000001_seg001_l4_gold.json"
    (tmp_path / filename).write_text("{}", encoding="utf-8")

    with pytest.raises(L4ValidationError, match=rf"{filename}: missing fields"):
        validate_dataset(rows, tmp_path)


def test_cli_accepts_one_complete_temporary_dataset(tmp_path: Path) -> None:
    index_path, labels_dir = write_complete_temporary_dataset(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--index",
            str(index_path),
            "--annotations",
            str(labels_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )

    assert result.returncode == 0
    assert "OK: validated 1 L4 labels" in result.stdout
    assert result.stderr == ""


def test_summary_reports_counts_means_and_low_confidence_ids() -> None:
    label = make_valid_label()

    summary = summarize_annotations([label])

    assert summary["total"] == 1
    assert summary["datasets"] == {"CH-SIMS": 1}
    assert summary["contradiction_types"] == {"consistent": 1}
    assert summary["mean_confidence"] == {
        "text": 0.9,
        "speech": 0.8,
        "macro": 0.8,
        "micro": 0.0,
    }
    assert summary["mean_weight"] == {
        "text": 0.36,
        "speech": 0.32,
        "macro": 0.32,
        "micro": 0.0,
    }
    assert summary["low_confidence_ids"] == []
    assert summary["pending_micro_review"] == 1
