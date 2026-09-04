# IEMOCAP / MOSEI Source Index Report

GitHub issue: https://github.com/cuiyuxun-droid/ea-quad-overlay/issues/10

## Scope

- Target outputs: `source_index/iemocap_index.csv`, `source_index/mosei_index.csv`.
- Required source metadata: audio/video paths, text paths, raw emotion labels, raw sentiment labels, VA labels, and usable sample flags.
- Acceptance criteria: usable sample paths are complete; raw labels can be mapped into later weak labels.

## Issue Context

- Issue #10 is open and has no comments as of the GitHub API check in this
  worktree.
- No `issue-10` remote branch was found. Existing remote branches cover earlier
  source indexing, feature extraction, micro review, L4 labels, and M1
  manifests.
- Related implementation check: `origin/main` keeps the shared source index
  template but does not contain dedicated IEMOCAP/MOSEI index outputs.

## Implemented Repository Support

- Added `scripts/build_iemocap_mosei_index.py`.
- Added `scripts/validate_iemocap_mosei_index.py`.
- Added schema-ready output files for both requested indexes.
- Validation now requires rows marked `usable_for_l4=true` to have non-empty
  video, audio, and text paths plus a mapped weak label.
- MOSEI rows use segmented media paths and transcript pointers of the form
  `Transcript/Segmented/Combined/<video_id>.txt#clip=<clip_id>`.
- Allocation map source: `docs/source_index_contract.md`.
- IEMOCAP new-row range: `EAQ300000` - `EAQ399999`.
- MOSEI new-row range: `EAQ400000` - `EAQ499999`.
- MOSI new-row range: `EAQ500000` - `EAQ599999`.

The generated index schema starts with the shared source index fields and appends:

| Field | Purpose |
| --- | --- |
| `raw_emotion` | Original categorical emotion label, used by IEMOCAP and by MOSEI/MOSI when available. |
| `raw_sentiment` | Original sentiment label or score, used by MOSEI/MOSI. |
| `raw_valence` | Original valence score, used by IEMOCAP when available. |
| `raw_arousal` | Original activation/arousal score, used by IEMOCAP when available. |
| `weak_label_hint` | Deterministic positive/neutral/negative mapping for later weak labels. |
| `label_source` | Exact source annotation file pointer for traceability. |

## Data Access Status

The provided server command was tested:

```bash
ssh -p 11482 root@connect.westd.seetacloud.com
```

SSH key access was configured and the server validation run completed against
the dataset directories:

| Dataset | Server root |
| --- | --- |
| IEMOCAP | `/root/autodl-tmp/data/datasets/alipan/iemocap/IEMOCAP_full_release` |
| MOSEI | `/root/autodl-tmp/data/datasets/baidupcs/CMU-MOSEI` |
| MOSI | `/root/autodl-tmp/data/datasets/mosi` (not present in this server snapshot) |

## Generation Command

Once server access is available, run on a machine with the datasets mounted:

```bash
python scripts/build_iemocap_mosei_index.py \
  --iemocap-root /root/autodl-tmp/data/datasets/alipan/iemocap/IEMOCAP_full_release \
  --mosei-root /root/autodl-tmp/data/datasets/baidupcs/CMU-MOSEI \
  --mosi-root /root/autodl-tmp/data/datasets/mosi \
  --id-registry source_index/iemocap_index.csv source_index/mosei_index.csv
```

If the actual server directory names differ, pass the discovered roots with the same arguments.

## Validation

Strict validation run on the dataset server:

```bash
/root/miniconda3/bin/python -B scripts/validate_iemocap_mosei_index.py --check-paths
```

Result:

```text
OK: validated 14672 rows (IEMOCAP=5766, MOSEI=8906, MOSI=0)
```

The validated CSV outputs were copied back into this worktree.

## Allocation Summary

```text
dataset: IEMOCAP
first_ea_id: EAQ300000
last_ea_id: EAQ305765
seed_rows_inherited: 0
new_rows_allocated: 5766

dataset: MOSEI
first_ea_id: EAQ400000
last_ea_id: EAQ408905
seed_rows_inherited: 0
new_rows_allocated: 8906
```

## Quality Heuristics

- `face_quality=high` when the referenced video path exists; otherwise `low`.
- `audio_quality=high` when the referenced audio path exists; otherwise `low`.
- `text_quality=high` when the transcript pointer resolves; otherwise `low`.
- `usable_for_micro=true` requires video, audio, and text pointers to resolve.
- `usable_for_l4=true` additionally requires a positive/neutral/negative weak
  label mapping from the original IEMOCAP emotion or aggregated MOSEI sentiment.
