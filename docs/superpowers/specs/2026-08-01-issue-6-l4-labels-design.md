# Issue #6 M1 L4 Labeling Design

## Goal

Produce one traceable L4 annotation JSON for each of the 20 M1 samples, plus a
validation script, behavior-focused tests, and a statistics report. The result
must satisfy GitHub Issue #6 without presenting unreviewed micro-expression
signals as completed Issue #5 human review.

## Scope

This change will:

- create 20 files under `annotations/l4_gold/`;
- validate their schema and cross-field consistency;
- summarize label distributions and evidence limitations in
  `reports/m1_l4_label_stats.md`;
- use raw media and source annotations as the primary evidence;
- use Issue #4 artifacts only as modality availability and quality evidence.

This change will not:

- merge or modify the Issue #4 feature-extraction branch;
- create Issue #5 micro-review annotations;
- claim two-reviewer adjudication;
- commit raw datasets, downloaded media, credentials, or temporary review
  artifacts.

## Working Branch

All changes are made on `feature/issue-6-l4-labels`, branched from `main`.

## Evidence Sources

Evidence is considered in this order:

1. `source_index/m1_sample_20.csv` for identity, dataset, split, and source
   pointers.
2. Original CH-SIMS and MELD annotations on the supplied server for transcript,
   sentiment, and emotion context.
3. Original audio and video clips on the supplied server for speech, macro, and
   micro visual review.
4. Issue #4 feature metadata for extraction status, duration, face-detection
   coverage, and micro candidate timing. Generic embeddings are not interpreted
   as VA predictions.

The server is accessed read-only. Credentials are supplied at runtime and never
written into repository files.

## Annotation File Contract

Each M1 sample produces exactly one file named
`EAQxxxxxx_seg001_l4_gold.json`. The JSON shape is:

```json
{
  "schema_version": "m1-l4-gold-v1",
  "ea_id": "EAQ000001",
  "segment_id": "EAQ000001_seg001",
  "source_dataset": "CH-SIMS",
  "modality_va": {
    "text": {"valence": 0.2, "arousal": 0.1, "confidence": 0.9},
    "speech": {"valence": 0.1, "arousal": 0.2, "confidence": 0.8},
    "macro": {"valence": 0.15, "arousal": 0.1, "confidence": 0.8},
    "micro": {"valence": 0.0, "arousal": 0.0, "confidence": 0.0}
  },
  "inter_va": {"valence": 0.152, "arousal": 0.132, "confidence": 0.836},
  "contradiction_type": "consistent",
  "involved_modalities": [],
  "fusion_weights": {
    "text": 0.36,
    "speech": 0.32,
    "macro": 0.32,
    "micro": 0.0
  },
  "reason": "Concise evidence-based explanation.",
  "annotation_meta": {
    "method": "evidence_triangulation_single_pass",
    "review_status": "single_pass_pending_second_review",
    "micro_review_status": "pending_issue_5",
    "evidence": ["source_annotation", "raw_audio", "raw_video"]
  }
}
```

Contract rules:

- `ea_id` must match `^EAQ[0-9]{6}$` and the M1 source index.
- `segment_id` must equal `<ea_id>_seg001`.
- `source_dataset` must be `CH-SIMS` or `MELD` and match the source index.
- Each modality contains numeric `valence`, `arousal`, and `confidence`.
- Valence and arousal are in `[-1, 1]`; confidence is in `[0, 1]`.
- `contradiction_type` is one of `consistent`, `masking`, `sarcasm`,
  `hidden_emotion`, or `intensity_mismatch`.
- `involved_modalities` contains only `text`, `speech`, `macro`, and `micro`,
  with no duplicates. It is empty for `consistent` and non-empty otherwise.
- `fusion_weights` contains exactly the four modalities, each weight is in
  `[0, 1]`, and the sum is `1.0` within an absolute tolerance of `1e-6`.
- `reason` is a non-empty evidence explanation, not a restatement of the label.

## VA and Contradiction Decisions

VA values are assigned from direct evidence on a continuous scale. Source labels
anchor text interpretation but do not overwrite audible or visible evidence.

The contradiction decision uses the following order when multiple descriptions
could apply:

1. `sarcasm`: literal text polarity conflicts with speech or visual signals and
   contextual irony is present.
2. `masking`: overt text or macro display is positive/neutral while speech or
   micro evidence indicates a concealed negative state.
3. `hidden_emotion`: overt text and macro evidence are weak/neutral while speech
   or micro evidence carries a stronger coherent emotional direction.
4. `intensity_mismatch`: modalities agree in polarity but differ materially in
   magnitude; a VA norm gap of at least `0.45` is the default quantitative cue.
5. `consistent`: no higher-priority type is supported, no confident modalities
   oppose in valence, and the maximum pairwise VA distance among modalities with
   confidence of at least `0.50` is at most `0.35`.

Thresholds are decision aids, not substitutes for evidence. Every non-consistent
label must identify the conflicting modalities and explain the observed cue.

## Confidence, Weights, and Inter-VA

The four modality confidences reflect evidence quality. While Issue #5 is
incomplete:

- micro confidence is capped at `0.60`;
- micro fusion weight is capped at `0.10`;
- no confirmed micro signal is represented as VA `(0.0, 0.0)` with confidence
  `0.0` and weight `0.0`, not as a negative micro-expression decision.

Initial raw weights are calculated as:

```text
raw_weight[modality] = confidence[modality] * reliability[modality]
```

Reliability is `1.0` for text, speech, and macro and `0.5` for pending-review
micro evidence. Contradiction-specific multipliers are then applied:

| Type | text | speech | macro | micro |
| --- | ---: | ---: | ---: | ---: |
| consistent | 1.0 | 1.0 | 1.0 | 1.0 |
| sarcasm | 0.6 | 1.2 | 1.1 | 1.0 |
| masking | 0.7 | 1.1 | 0.8 | 1.2 |
| hidden_emotion | 0.8 | 1.1 | 0.7 | 1.2 |
| intensity_mismatch | 1.0 | 1.0 | 1.0 | 1.0 |

Weights are normalized, the micro cap is applied, and any excess micro weight is
redistributed proportionally across non-micro modalities. If every raw weight is
zero, or if no non-micro raw weight is available to enforce the micro cap,
validation fails instead of inventing weights.

`inter_va.valence` and `inter_va.arousal` are the weighted sums of modality VA
values. `inter_va.confidence` is the weighted sum of modality confidences. Values
are rounded to six decimal places only after calculation.

## Components

### Annotation files

Twenty JSON files carry the reviewed evidence and final M1 labels. They contain
no raw embeddings or media.

### Validator

`scripts/validate_m1_l4_labels.py` reads the M1 source index and annotation
directory. It validates file count, naming, identity, schema, ranges, enum
values, involved-modality rules, weight normalization, inter-VA calculations,
reason presence, and pending-micro caps. It exits non-zero with a specific file
and field error on failure.

### Tests

`scripts/tests/test_validate_m1_l4_labels.py` uses temporary directories and
hand-written fixtures. Tests cover a valid label and independent failures for a
missing sample, illegal contradiction type, out-of-range VA/confidence,
duplicate/invalid involved modality, non-unit weights, inconsistent inter-VA,
empty reason, and a pending-micro cap violation.

### Statistics report

`reports/m1_l4_label_stats.md` records:

- total sample count and CH-SIMS/MELD split;
- contradiction-type distribution;
- mean modality confidence and fusion weight;
- low-confidence and non-consistent samples;
- the number of samples pending Issue #5 micro review;
- the exact validation command and result;
- the single-pass review limitation.

## Data Flow

```text
M1 source index
  -> source annotation lookup
  -> raw text/audio/video evidence review
  -> Issue #4 quality metadata check
  -> modality VA and confidence assignment
  -> contradiction classification
  -> deterministic weight and inter-VA calculation
  -> per-sample JSON
  -> validator
  -> statistics report
```

## Error Handling

- A missing source-index row, transcript, audio clip, or video clip blocks that
  sample; no annotation is fabricated.
- Missing Issue #4 metadata lowers available evidence but does not block raw
  media review. The limitation is recorded in `annotation_meta.evidence` and the
  report.
- Missing Issue #5 output invokes the explicit pending-micro caps above.
- Ambiguous cues are evaluated through the stated priority and thresholds. A
  sample is labeled `consistent` only when its numeric conditions hold; every
  low-confidence decision is listed in the report and explains the ambiguity in
  `reason`.
- Any validation failure blocks the final commit that contains annotation data.

## Verification and Acceptance

Implementation follows red-green-refactor for the validator. Completion requires:

```powershell
python -m pytest scripts/tests/test_validate_m1_l4_labels.py -q
python scripts/validate_m1_sample_20.py
python scripts/validate_m1_l4_labels.py
python -m pytest -q
```

The final result is acceptable only when:

- all 20 source-index IDs have exactly one L4 JSON;
- all required Issue #6 fields satisfy the contract;
- all deterministic inter-VA and weight checks pass;
- the report accurately describes the evidence and review limitations;
- the Git diff contains no credentials, raw media, or unrelated changes.
