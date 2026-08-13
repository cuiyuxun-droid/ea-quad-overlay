# EA-Quad-Overlay File Structure

This document defines the repository layout, directory purpose, and naming
rules for the EA-Quad-Overlay L4 dataset work.

## Directory Layout

```text
EA-Quad-Overlay/
  source_index/
  features/
    text/
    speech/
    macro/
    micro/
  annotations/
    micro_review/
    l4_gold/
  manifests/
  reports/
  scripts/
```

## Directory Purpose

| Directory | Purpose |
| --- | --- |
| `source_index/` | Unified index tables that map EA sample IDs back to the original datasets and source records. |
| `features/text/` | Text modality features, embeddings, token-level signals, or processed transcript artifacts. |
| `features/speech/` | Speech/audio modality features such as acoustic, prosody, voice, and timing features. |
| `features/macro/` | Macro-expression and face-level features extracted from frames or clips. |
| `features/micro/` | Micro-expression event features, candidate detections, and derived event attributes. |
| `annotations/micro_review/` | Human review files for micro-expression positive/negative confirmation. |
| `annotations/l4_gold/` | Final or candidate L4 gold annotation files. |
| `manifests/` | Dataset manifests that enumerate selected samples, splits, feature files, and annotation versions. |
| `reports/` | QA reports, experiment summaries, error analyses, and release notes. |
| `scripts/` | Reusable scripts for indexing, feature extraction, validation, and manifest generation. |

## Identifier Rules

All downstream files should be named from the unified EA identifiers so that
samples remain traceable across modalities and tasks.

### Sample ID

Format:

```text
EAQ000001
```

Rules:

- Prefix is always `EAQ`.
- Numeric suffix is zero-padded to six digits.
- IDs are assigned once and never reused.
- Example range: `EAQ000001`, `EAQ000002`, `EAQ000003`.

### Segment ID

Format:

```text
EAQ000001_seg001
```

Rules:

- Segment IDs append `_segNNN` to the sample ID.
- Segment number is zero-padded to three digits.
- A sample with multiple clips or utterance windows increments the segment
  suffix: `EAQ000001_seg001`, `EAQ000001_seg002`.

### Micro-Expression Event ID

Format:

```text
EAQ000001_seg001_micro001
```

Rules:

- Micro-event IDs append `_microNNN` to the segment ID.
- Event number is zero-padded to three digits.
- Multiple reviewed events in the same segment increment the event suffix.

## File Naming Examples

| Artifact | Example |
| --- | --- |
| Text feature | `features/text/EAQ000001_seg001_text.json` |
| Speech feature | `features/speech/EAQ000001_seg001_speech.json` |
| Macro feature | `features/macro/EAQ000001_seg001_macro.json` |
| Micro feature | `features/micro/EAQ000001_seg001_micro001.json` |
| Micro review annotation | `annotations/micro_review/EAQ000001_seg001_micro_review.csv` |
| L4 gold annotation | `annotations/l4_gold/EAQ000001_seg001_l4_gold.json` |
| Manifest | `manifests/ea_l4_gold_v0.1_manifest.csv` |

## Source Traceability

Each EA sample must have one row in `source_index/source_index_template.csv`
or a derived source index file using the same schema. The `source_id` field
must preserve the original dataset identifier so the sample can be traced back
to CH-SIMS, MELD, IEMOCAP, MOSEI, or MUStARD.

For global identity, index roles, path pointer formats, and cross-index merge
rules, see [`source_index_contract.md`](source_index_contract.md). PR authors
should also use [`pr_acceptance_checklist.md`](pr_acceptance_checklist.md).
