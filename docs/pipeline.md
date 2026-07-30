# EA-Quad-Overlay Pipeline

## Overview

```
Raw Datasets                     EA Workspace                 L4 Gold
┌──────────┐    ┌──────────────────────────────────┐    ┌──────────────┐
│ CH-SIMS  │───▶│                                  │───▶│              │
│ MELD     │───▶│  1. Ingest → 2. Align → 3.       │    │   L4 Gold    │
│ IEMOCAP  │───▶│     Extract → 4. Annotate →      │───▶│   Dataset    │
│ MOSEI    │───▶│     5. Review → 6. Package       │    │              │
│ MUStARD  │───▶│                                  │    └──────────────┘
└──────────┘    └──────────────────────────────────┘
```

## Stages

### Stage 1 — Ingest (scripts/ingest/)
- Read raw data from each source dataset.
- Assign EA sample/segment IDs.
- Write source index record (`source_index/`).
- Copy / symlink raw files into `features/<modality>/` as `EAQ*_<modality>.*`.

### Stage 2 — Align (scripts/align/)
- For multi-modal datasets, ensure text/audio/video windows match.
- Output alignment manifests to `manifests/`.

### Stage 3 — Feature Extraction (scripts/extract/)
One script per modality:

| Modality | Script | Output |
|----------|--------|--------|
| Text | `extract_text.py` | Embeddings, token-level signals |
| Speech | `extract_speech.py` | Acoustic/prosody features |
| Macro | `extract_macro.py` | Face-level features (expression, AU) |
| Micro | `extract_micro.py` | Micro-expression candidates |

### Stage 4 — Annotation (scripts/annotate/)
- Generate micro-expression candidate lists for human review.
- Output to `annotations/micro_review/`.

### Stage 5 — Review & Curation (scripts/review/)
- Aggregate human reviews.
- Compute inter-rater agreement.
- Flag disagreements for adjudication.
- Output reviewed positives to `annotations/l4_gold/`.

### Stage 6 — Packaging (scripts/package/)
- Merge L4 gold annotations with feature files.
- Split into train/val/test.
- Write final manifests to `manifests/`.
- Generate release report to `reports/`.

## Running a Pipeline

```bash
# Single stage
python scripts/ingest/ingest_ch_sims.py --config configs/run_v1.yaml

# Full pipeline (example)
python scripts/pipeline.py --config configs/run_v1.yaml --stages ingest,extract_text
```

## ID Lifecycle

```
Raw record     →  source_index/  →  EAQ000001
EAQ000001      →  split clip     →  EAQ000001_seg001
EAQ000001_seg001 → micro-event   →  EAQ000001_seg001_micro001
```
