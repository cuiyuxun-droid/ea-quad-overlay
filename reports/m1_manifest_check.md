# M1 Manifest Check

GitHub issue: <https://github.com/cuiyuxun-droid/ea-quad-overlay/issues/7>

## Scope and result

- Generated four L2 manifests and one L4 fusion manifest for the 20 M1 samples.
- Records are built from Issue #4 feature meta, Issue #5 micro reviews, and Issue #6 L4 gold labels.
- Repository read/validation confirms JSONL parseability, complete fusion fields, and existing `feature_path` files.

## Manifest files

| File | Records |
| --- | ---: |
| `manifests/l2_text_m1.jsonl` | 20 |
| `manifests/l2_speech_m1.jsonl` | 20 |
| `manifests/l2_macro_m1.jsonl` | 20 |
| `manifests/l2_micro_m1.jsonl` | 20 |
| `manifests/fusion_segments_m1.jsonl` | 20 |

## Validation

- Samples validated: **20**
- All L2 `feature_path` and `meta_path` files exist.
- All fusion `feature_paths`, `l4_gold_path`, and `micro_review_path` files exist.
- Fusion required fields are complete; `fusion_weights` sum to 1.

## Fusion contradiction distribution

| Type | Count |
| --- | ---: |
| `consistent` | 15 |
| `masking` | 1 |
| `sarcasm` | 0 |
| `hidden_emotion` | 1 |
| `intensity_mismatch` | 3 |

## Micro-review status in fusion manifest

| Status | Count |
| --- | ---: |
| `negative` | 14 |
| `uncertain` | 6 |

## Errors

- none
