# Issue #6 M1 L4 Labels Implementation Plan

> 2026-08-03 follow-up: PR #30 merged Issue #5 with 14 `negative`, 6
> `uncertain`, and 0 `positive` micro reviews. The implementation now
> cross-validates those source statuses, replaces `pending_issue_5` in the 20
> repository labels, and keeps micro VA/confidence/weight at zero because no
> positive event was confirmed.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce validated, evidence-traceable L4 labels for all 20 M1 samples and an accurate statistics report.

**Architecture:** A focused package module owns deterministic fusion calculations, per-annotation validation, dataset coverage checks, and summary statistics. A thin CLI joins the M1 source index to per-sample JSON files. Raw media and source annotations are reviewed outside Git; only final labels and aggregate findings are committed.

**Tech Stack:** Python 3.10+, Python standard library, pytest 7+, Git, SSH and ffmpeg for temporary evidence review.

## Global Constraints

- Work only on `feature/issue-6-l4-labels`; the branch name contains no `codex`.
- Do not merge or modify the Issue #4 feature-extraction branch.
- Do not create or claim completion of Issue #5 micro-review annotations.
- Do not claim two-reviewer adjudication.
- Do not commit raw datasets, downloaded media, credentials, or temporary review artifacts.
- Server dataset paths are read-only evidence sources; temporary review artifacts remain local and outside the repository.
- Generic Issue #4 embeddings are availability and quality evidence, not VA predictions.
- Every sample produces exactly one `annotations/l4_gold/EAQxxxxxx_seg001_l4_gold.json`.
- VA values remain in `[-1, 1]`; confidence and weights remain in `[0, 1]`.
- Pending Issue #5 micro confidence is at most `0.60` and micro fusion weight is at most `0.10`.
- `fusion_weights` sum to `1.0` within `1e-6`; `inter_va` is the deterministic weighted result.
- Use only `consistent`, `masking`, `sarcasm`, `hidden_emotion`, and `intensity_mismatch`.

---

## File Structure

- Create `ea_quad_overlay/l4_labels.py`: domain constants, fusion calculations, validation, and aggregate statistics.
- Create `scripts/validate_m1_l4_labels.py`: repository-aware validation CLI.
- Create `tests/test_l4_labels.py`: calculation and per-annotation behavior tests.
- Create `tests/__init__.py`: make shared test helpers importable deterministically.
- Create `tests/l4_test_data.py`: shared complete annotation fixture for validator tests.
- Create `tests/test_validate_m1_l4_labels.py`: temporary-dataset, CLI, and real-repository coverage tests.
- Create 20 `annotations/l4_gold/EAQxxxxxx_seg001_l4_gold.json` files: reviewed labels.
- Create `reports/m1_l4_label_stats.md`: distributions, confidence/weight statistics, flagged cases, and limitations.

### Task 1: Deterministic Fusion and Inter-VA Calculations

**Files:**
- Create: `ea_quad_overlay/l4_labels.py`
- Create: `tests/test_l4_labels.py`

**Interfaces:**
- Consumes: confidence mappings keyed by `text`, `speech`, `macro`, and `micro`.
- Produces: `calculate_fusion_weights(confidences: Mapping[str, float], contradiction_type: str, micro_review_status: str) -> dict[str, float]`.
- Produces: `calculate_inter_va(modality_va: Mapping[str, Mapping[str, float]], weights: Mapping[str, float]) -> dict[str, float]`.

- [ ] **Step 1: Write failing calculation tests**

```python
import pytest

from ea_quad_overlay.l4_labels import calculate_fusion_weights, calculate_inter_va


def test_consistent_weights_follow_confidence_and_sum_to_one():
    weights = calculate_fusion_weights(
        {"text": 0.9, "speech": 0.8, "macro": 0.8, "micro": 0.0},
        "consistent",
        "pending_issue_5",
    )
    assert weights == {"text": 0.36, "speech": 0.32, "macro": 0.32, "micro": 0.0}
    assert sum(weights.values()) == 1.0


def test_pending_micro_weight_is_capped_and_excess_is_redistributed():
    weights = calculate_fusion_weights(
        {"text": 0.4, "speech": 0.4, "macro": 0.4, "micro": 0.6},
        "hidden_emotion",
        "pending_issue_5",
    )
    assert weights["micro"] == 0.1
    assert abs(sum(weights.values()) - 1.0) <= 1e-6


def test_inter_va_is_weighted_and_rounded_to_six_places():
    va = {
        "text": {"valence": 0.2, "arousal": 0.1, "confidence": 0.9},
        "speech": {"valence": 0.1, "arousal": 0.2, "confidence": 0.8},
        "macro": {"valence": 0.15, "arousal": 0.1, "confidence": 0.8},
        "micro": {"valence": 0.0, "arousal": 0.0, "confidence": 0.0},
    }
    result = calculate_inter_va(
        va, {"text": 0.36, "speech": 0.32, "macro": 0.32, "micro": 0.0}
    )
    assert result == {"valence": 0.152, "arousal": 0.132, "confidence": 0.836}


def test_zero_confidences_are_rejected():
    with pytest.raises(ValueError, match="all raw fusion weights are zero"):
        calculate_fusion_weights(
            {"text": 0.0, "speech": 0.0, "macro": 0.0, "micro": 0.0},
            "consistent",
            "pending_issue_5",
        )


def test_unknown_contradiction_type_is_rejected():
    with pytest.raises(ValueError, match="unsupported contradiction_type"):
        calculate_fusion_weights(
            {"text": 1.0, "speech": 1.0, "macro": 1.0, "micro": 0.0},
            "unknown",
            "pending_issue_5",
        )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/test_l4_labels.py -q`

Expected: collection fails because `ea_quad_overlay.l4_labels` does not exist.

- [ ] **Step 3: Implement the minimum calculations**

```python
from collections.abc import Mapping

MODALITIES = ("text", "speech", "macro", "micro")
CONTRADICTION_TYPES = (
    "consistent", "masking", "sarcasm", "hidden_emotion", "intensity_mismatch"
)
TYPE_MULTIPLIERS = {
    "consistent": {"text": 1.0, "speech": 1.0, "macro": 1.0, "micro": 1.0},
    "sarcasm": {"text": 0.6, "speech": 1.2, "macro": 1.1, "micro": 1.0},
    "masking": {"text": 0.7, "speech": 1.1, "macro": 0.8, "micro": 1.2},
    "hidden_emotion": {"text": 0.8, "speech": 1.1, "macro": 0.7, "micro": 1.2},
    "intensity_mismatch": {"text": 1.0, "speech": 1.0, "macro": 1.0, "micro": 1.0},
}


def calculate_fusion_weights(confidences, contradiction_type, micro_review_status):
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
        non_micro_total = sum(weights[m] for m in MODALITIES if m != "micro")
        if non_micro_total <= 0:
            raise ValueError("no non-micro weight is available")
        weights["micro"] = 0.1
        for modality in MODALITIES:
            if modality != "micro":
                weights[modality] += excess * weights[modality] / non_micro_total
    rounded = {modality: round(weights[modality], 6) for modality in MODALITIES}
    rounded["macro"] = round(rounded["macro"] + (1.0 - sum(rounded.values())), 6)
    return rounded


def calculate_inter_va(modality_va, weights):
    return {
        field: round(
            sum(float(modality_va[m][field]) * float(weights[m]) for m in MODALITIES),
            6,
        )
        for field in ("valence", "arousal", "confidence")
    }
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_l4_labels.py -q`

Expected: five tests pass.

- [ ] **Step 5: Commit the calculation unit**

```powershell
git add ea_quad_overlay/l4_labels.py tests/test_l4_labels.py
git commit -m "feat: add deterministic L4 fusion calculations"
```

### Task 2: Annotation Contract and Dataset Validator

**Files:**
- Modify: `ea_quad_overlay/l4_labels.py`
- Modify: `tests/test_l4_labels.py`
- Create: `tests/__init__.py`
- Create: `tests/l4_test_data.py`
- Create: `scripts/validate_m1_l4_labels.py`
- Create: `tests/test_validate_m1_l4_labels.py`

**Interfaces:**
- Produces: `L4ValidationError(ValueError)`.
- Produces: `validate_annotation(label: Mapping[str, Any], expected_ea_id: str, expected_dataset: str) -> None`.
- Produces: `validate_dataset(index_rows: Sequence[Mapping[str, str]], annotations_dir: Path) -> list[dict[str, Any]]`.
- Produces: `summarize_annotations(labels: Sequence[Mapping[str, Any]]) -> dict[str, Any]`.
- CLI: `python scripts/validate_m1_l4_labels.py [--index PATH] [--annotations PATH]`.

- [ ] **Step 1: Add failing per-annotation tests**

Create `make_valid_label()` in `tests/l4_test_data.py` with the exact JSON example
from the design, import it in both validator test files, and add these behaviors:

```python
from ea_quad_overlay.l4_labels import L4ValidationError, validate_annotation
from tests.l4_test_data import make_valid_label


def test_valid_annotation_passes():
    validate_annotation(make_valid_label(), "EAQ000001", "CH-SIMS")


def test_non_consistent_label_requires_involved_modalities():
    label = make_valid_label()
    label["contradiction_type"] = "sarcasm"
    with pytest.raises(L4ValidationError, match="involved_modalities"):
        validate_annotation(label, "EAQ000001", "CH-SIMS")


def test_out_of_range_va_is_rejected():
    label = make_valid_label()
    label["modality_va"]["text"]["valence"] = 1.1
    with pytest.raises(L4ValidationError, match="valence"):
        validate_annotation(label, "EAQ000001", "CH-SIMS")


def test_pending_micro_caps_are_enforced():
    label = make_valid_label()
    label["modality_va"]["micro"]["confidence"] = 0.61
    with pytest.raises(L4ValidationError, match="micro confidence"):
        validate_annotation(label, "EAQ000001", "CH-SIMS")
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_l4_labels.py -q`

Expected: import fails because `L4ValidationError` and `validate_annotation` do not exist.

- [ ] **Step 3: Implement complete per-annotation validation**

Add `L4ValidationError` and `validate_annotation`. The function must collect and
raise file/field-specific errors for:

```python
REQUIRED_FIELDS = {
    "schema_version", "ea_id", "segment_id", "source_dataset", "modality_va",
    "inter_va", "contradiction_type", "involved_modalities", "fusion_weights",
    "reason", "annotation_meta",
}
```

It must reject missing/extra modality keys, booleans where numbers are expected,
invalid ID/dataset/type values, ranges, duplicate involved modalities, empty
reasons, weights outside `[0, 1]`, weight sums outside `1e-6`, inter-VA values
inconsistent with `calculate_inter_va`, micro confidence above `0.60`, and micro
weight above `0.10` while `micro_review_status == "pending_issue_5"`.

Implement the checks with guarded dictionary access so malformed input produces
one `L4ValidationError` rather than `KeyError` or `TypeError`:

```python
class L4ValidationError(ValueError):
    pass


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_annotation(label, expected_ea_id, expected_dataset):
    errors = []
    missing = sorted(REQUIRED_FIELDS - set(label))
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
    if label.get("schema_version") != "m1-l4-gold-v1":
        errors.append("schema_version must be m1-l4-gold-v1")
    if label.get("ea_id") != expected_ea_id:
        errors.append(f"ea_id must be {expected_ea_id}")
    if label.get("segment_id") != f"{expected_ea_id}_seg001":
        errors.append(f"segment_id must be {expected_ea_id}_seg001")
    if label.get("source_dataset") != expected_dataset:
        errors.append(f"source_dataset must be {expected_dataset}")

    modality_va = label.get("modality_va", {})
    weights = label.get("fusion_weights", {})
    if set(modality_va) != set(MODALITIES):
        errors.append("modality_va must contain exactly four modalities")
    if set(weights) != set(MODALITIES):
        errors.append("fusion_weights must contain exactly four modalities")
    for modality in MODALITIES:
        values = modality_va.get(modality, {})
        for field, low, high in (
            ("valence", -1.0, 1.0),
            ("arousal", -1.0, 1.0),
            ("confidence", 0.0, 1.0),
        ):
            value = values.get(field)
            if not _is_number(value) or not low <= value <= high:
                errors.append(f"modality_va.{modality}.{field} outside [{low}, {high}]")

    contradiction_type = label.get("contradiction_type")
    if contradiction_type not in CONTRADICTION_TYPES:
        errors.append("invalid contradiction_type")
    involved = label.get("involved_modalities", [])
    if not isinstance(involved, list) or len(involved) != len(set(involved)):
        errors.append("involved_modalities must be a unique list")
    elif any(modality not in MODALITIES for modality in involved):
        errors.append("involved_modalities contains an invalid modality")
    elif contradiction_type == "consistent" and involved:
        errors.append("consistent requires empty involved_modalities")
    elif contradiction_type in CONTRADICTION_TYPES[1:] and not involved:
        errors.append("non-consistent labels require involved_modalities")

    valid_va_shape = set(modality_va) == set(MODALITIES) and all(
        set(modality_va[m]) == {"valence", "arousal", "confidence"}
        and all(_is_number(modality_va[m][field]) for field in ("valence", "arousal", "confidence"))
        for m in MODALITIES
    )
    if (
        valid_va_shape
        and set(weights) == set(MODALITIES)
        and all(_is_number(weights[m]) for m in MODALITIES)
    ):
        if any(not 0.0 <= weights[m] <= 1.0 for m in MODALITIES):
            errors.append("fusion_weights values must be in [0, 1]")
        if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6):
            errors.append("fusion_weights must sum to 1")
        expected_inter = calculate_inter_va(modality_va, weights)
        for field, expected in expected_inter.items():
            actual = label.get("inter_va", {}).get(field)
            if not _is_number(actual) or not math.isclose(actual, expected, abs_tol=1e-6):
                errors.append(f"inter_va.{field} does not match weighted modalities")

    meta = label.get("annotation_meta", {})
    if meta.get("micro_review_status") == "pending_issue_5":
        if modality_va.get("micro", {}).get("confidence", 1.0) > 0.60:
            errors.append("pending micro confidence exceeds 0.60")
        if weights.get("micro", 1.0) > 0.10:
            errors.append("pending micro fusion weight exceeds 0.10")
    if not isinstance(label.get("reason"), str) or not label["reason"].strip():
        errors.append("reason must be non-empty")
    if errors:
        raise L4ValidationError("; ".join(errors))
```

- [ ] **Step 4: Add failing dataset and CLI tests**

```python
def test_dataset_rejects_missing_annotation(tmp_path):
    rows = [{"ea_id": "EAQ000001", "source_dataset": "CH-SIMS"}]
    with pytest.raises(L4ValidationError, match="missing annotation"):
        validate_dataset(rows, tmp_path)


def test_dataset_rejects_unexpected_annotation(tmp_path):
    rows = []
    (tmp_path / "EAQ999999_seg001_l4_gold.json").write_text("{}", encoding="utf-8")
    with pytest.raises(L4ValidationError, match="unexpected annotation"):
        validate_dataset(rows, tmp_path)


def test_cli_accepts_one_complete_temporary_dataset(tmp_path):
    index_path, labels_dir = write_complete_temporary_dataset(tmp_path)
    result = subprocess.run(
        [sys.executable, "scripts/validate_m1_l4_labels.py",
         "--index", str(index_path), "--annotations", str(labels_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "OK: validated 1 L4 labels" in result.stdout


def test_summary_reports_counts_means_and_low_confidence_ids():
    label = make_valid_label()
    summary = summarize_annotations([label])
    assert summary["total"] == 1
    assert summary["datasets"] == {"CH-SIMS": 1}
    assert summary["contradiction_types"] == {"consistent": 1}
    assert summary["mean_confidence"]["text"] == 0.9
    assert summary["mean_weight"]["micro"] == 0.0
    assert summary["low_confidence_ids"] == []
    assert summary["pending_micro_review"] == 1
```

`write_complete_temporary_dataset` writes a two-column CSV with
`ea_id,source_dataset` and one JSON from `make_valid_label()`.

- [ ] **Step 5: Run dataset/CLI tests and verify RED**

Run: `python -m pytest tests/test_validate_m1_l4_labels.py -q`

Expected: import or subprocess failure because dataset validation and the CLI do not exist.

- [ ] **Step 6: Implement dataset validation, summary, and CLI**

`validate_dataset` must require exact one-to-one coverage between source-index
IDs and `*_seg001_l4_gold.json`, reject unexpected JSON files, parse UTF-8, and
prefix annotation errors with the filename. `summarize_annotations` must return:

```python
{
    "total": int,
    "datasets": dict[str, int],
    "contradiction_types": dict[str, int],
    "mean_confidence": dict[str, float],
    "mean_weight": dict[str, float],
    "low_confidence_ids": list[str],
    "pending_micro_review": int,
}
```

`low_confidence_ids` contains labels whose `inter_va.confidence` is below
`0.60`. Means are calculated with `statistics.fmean` and rounded to six decimal
places.

Implement dataset coverage with exact expected filenames:

```python
def validate_dataset(index_rows, annotations_dir):
    expected = {
        f"{row['ea_id']}_seg001_l4_gold.json": row for row in index_rows
    }
    actual = {path.name: path for path in annotations_dir.glob("*_l4_gold.json")}
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing annotation: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected annotation: {', '.join(unexpected)}")
        raise L4ValidationError("; ".join(details))
    labels = []
    for filename, row in expected.items():
        path = actual[filename]
        try:
            label = json.loads(path.read_text(encoding="utf-8"))
            validate_annotation(label, row["ea_id"], row["source_dataset"])
        except (json.JSONDecodeError, L4ValidationError) as exc:
            raise L4ValidationError(f"{filename}: {exc}") from exc
        labels.append(label)
    return labels
```

The CLI defaults are:

```python
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "source_index" / "m1_sample_20.csv"
DEFAULT_ANNOTATIONS = ROOT / "annotations" / "l4_gold"
```

It prints `OK: validated <n> L4 labels` on success and `ERROR: <details>` to
stderr with exit code `1` on read, parse, or validation failure.

- [ ] **Step 7: Run validator tests and verify GREEN**

Run: `python -m pytest tests/test_l4_labels.py tests/test_validate_m1_l4_labels.py -q`

Expected: all calculation, contract, temporary-dataset, and CLI tests pass.

- [ ] **Step 8: Commit the validator unit**

```powershell
git add ea_quad_overlay/l4_labels.py scripts/validate_m1_l4_labels.py tests
git commit -m "feat: validate M1 L4 annotations"
```

### Task 3: Evidence Review and Twenty L4 Annotation Files

**Files:**
- Create: `annotations/l4_gold/EAQ000001_seg001_l4_gold.json` through `annotations/l4_gold/EAQ000020_seg001_l4_gold.json`
- Remove: `annotations/l4_gold/.gitkeep`
- Modify: `tests/test_validate_m1_l4_labels.py`

**Interfaces:**
- Consumes: M1 source index, server source annotations/raw clips, Issue #4 quality metadata, and calculation functions.
- Produces: exactly 20 JSON documents satisfying `validate_dataset`.

- [ ] **Step 1: Add the real-repository coverage test**

```python
def test_repository_m1_labels_are_complete_and_summarizable():
    root = Path(__file__).resolve().parents[1]
    with (root / "source_index" / "m1_sample_20.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    labels = validate_dataset(rows, root / "annotations" / "l4_gold")
    summary = summarize_annotations(labels)
    assert summary["total"] == 20
    assert summary["datasets"] == {"CH-SIMS": 11, "MELD": 9}
    assert sum(summary["contradiction_types"].values()) == 20
```

- [ ] **Step 2: Run the real-repository test and verify RED**

Run: `python -m pytest tests/test_validate_m1_l4_labels.py::test_repository_m1_labels_are_complete_and_summarizable -q`

Expected: failure naming the 20 missing annotations.

- [ ] **Step 3: Collect evidence outside Git**

Create `C:\tmp\ea-quad-overlay-issue-6-review`. Through the supplied SSH
account, retrieve the 11 referenced CH-SIMS rows, 9 referenced MELD rows, and
the corresponding media or streamed review frames. Inspect Issue #4 metadata
with `git show origin/feature/issue-4-m1-feature-extract:<feature-json-path>`.

Record externally for every ID: source transcript/emotion, text VA, speech
intensity and VA, macro cue and VA, any transient micro cue near the Issue #4
peak, confidences, contradiction type, involved modalities, and one evidence
sentence. Never save the password in the worksheet or repository.

- [ ] **Step 4: Apply the fixed review rubric**

- Confidence `0.90`: direct, clear source or media evidence.
- Confidence `0.75`: clear direction with moderate intensity uncertainty.
- Confidence `0.60`: weak but observable direction.
- Confidence below `0.60`: ambiguous and listed in the report.
- No confirmed micro cue: micro VA `(0.0, 0.0)`, confidence `0.0`, weight `0.0`.
- Observed but unreviewed micro cue: confidence no greater than `0.60`.
- Contradiction priority: `sarcasm > masking > hidden_emotion > intensity_mismatch > consistent`.

- [ ] **Step 5: Create the 20 annotation JSON files**

For each reviewed sample, compute rather than hand-edit weights and inter-VA:

```python
weights = calculate_fusion_weights(
    {m: modality_va[m]["confidence"] for m in MODALITIES},
    contradiction_type,
    "pending_issue_5",
)
inter_va = calculate_inter_va(modality_va, weights)
```

Write UTF-8 JSON with two-space indentation and a trailing newline. Every file
uses `schema_version: m1-l4-gold-v1`, `segment_id: <ea_id>_seg001`, a non-empty
reason, and:

```json
"annotation_meta": {
  "method": "evidence_triangulation_single_pass",
  "review_status": "single_pass_pending_second_review",
  "micro_review_status": "pending_issue_5",
  "evidence": ["source_annotation", "raw_audio", "raw_video", "issue_4_quality_metadata"]
}
```

Omit unavailable evidence tokens, lower the affected confidence, and state the
limitation in `reason`.

- [ ] **Step 6: Run the coverage test and validator and verify GREEN**

```powershell
python -m pytest tests/test_validate_m1_l4_labels.py::test_repository_m1_labels_are_complete_and_summarizable -q
python scripts/validate_m1_l4_labels.py
```

Expected: the test passes and the CLI prints `OK: validated 20 L4 labels`.

- [ ] **Step 7: Commit the annotation data**

```powershell
git add annotations/l4_gold tests/test_validate_m1_l4_labels.py
git commit -m "data: add M1 L4 labels for 20 samples"
```

### Task 4: Statistics Report and Final Verification

**Files:**
- Create: `reports/m1_l4_label_stats.md`

**Interfaces:**
- Consumes: `summarize_annotations` output and the 20 validated labels.
- Produces: a report whose counts and means match programmatic statistics.

- [ ] **Step 1: Compute the final summary from production code**

Run a read-only Python command importing `validate_dataset` and
`summarize_annotations`; print the returned mapping as indented JSON. Use those
exact values in the report rather than recomputing them manually.

```powershell
@'
import csv
import json
from pathlib import Path

from ea_quad_overlay.l4_labels import summarize_annotations, validate_dataset

root = Path.cwd()
with (root / "source_index" / "m1_sample_20.csv").open(
    newline="", encoding="utf-8"
) as handle:
    rows = list(csv.DictReader(handle))
labels = validate_dataset(rows, root / "annotations" / "l4_gold")
print(json.dumps(summarize_annotations(labels), ensure_ascii=False, indent=2))
'@ | python -
```

- [ ] **Step 2: Write the statistics report**

Use these sections with actual values:

```markdown
# M1 L4 Label Statistics

## Scope and result
## Dataset distribution
## Contradiction-type distribution
## Mean confidence and fusion weights
## Non-consistent and low-confidence samples
## Evidence and review limitations
## Validation
```

State that all labels are single-pass and pending second review. State the exact
count pending Issue #5 micro review and do not describe these records as human
micro-expression confirmation.

- [ ] **Step 3: Run complete fresh verification**

```powershell
python -m pytest tests/test_l4_labels.py tests/test_validate_m1_l4_labels.py -q
python scripts/validate_m1_sample_20.py
python scripts/validate_m1_l4_labels.py
python -m pytest -q
git diff --check
git status --short --branch
```

Expected: all tests pass, both validators exit `0`, `git diff --check` emits no
errors, and status contains only the intended report before commit.

- [ ] **Step 4: Review the Issue #6 acceptance checklist**

Confirm from output and files: exactly 20 JSON files, all required fields,
valid ranges, unit weight sums, legal contradiction types, involved modalities
for every non-consistent label, non-empty reasons, and no credential/raw media/
unrelated file in the diff.

- [ ] **Step 5: Commit the report**

```powershell
git add reports/m1_l4_label_stats.md
git commit -m "docs: report M1 L4 label statistics"
```

- [ ] **Step 6: Re-run verification after the commit**

```powershell
python -m pytest -q
python scripts/validate_m1_sample_20.py
python scripts/validate_m1_l4_labels.py
git status --short --branch
git log -5 --oneline --decorate
```

Expected: all commands exit `0`, the working tree is clean, and the branch shows
the design, implementation, annotation data, and report commits.
