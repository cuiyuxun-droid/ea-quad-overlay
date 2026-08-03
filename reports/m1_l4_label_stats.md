# M1 L4 Label Statistics

GitHub issue: <https://github.com/cuiyuxun-droid/ea-quad-overlay/issues/6>

## Scope and result

- Created one `m1-l4-gold-v1` JSON annotation for each of the 20 M1 samples.
- Every annotation contains four modality VA records, deterministic fusion
  weights, weighted `inter_va`, a legal contradiction type, involved modalities,
  an evidence reason, and review metadata.
- All annotations are single-pass labels pending a second reviewer. They should
  not be described as adjudicated gold labels yet.

## Dataset distribution

| Dataset | Samples | Share |
| --- | ---: | ---: |
| CH-SIMS | 11 | 55% |
| MELD | 9 | 45% |
| **Total** | **20** | **100%** |

## Contradiction-type distribution

| Type | Count | Share |
| --- | ---: | ---: |
| `consistent` | 15 | 75% |
| `intensity_mismatch` | 3 | 15% |
| `hidden_emotion` | 1 | 5% |
| `masking` | 1 | 5% |
| `sarcasm` | 0 | 0% |

## Mean confidence and fusion weights

The values below are calculated by
`ea_quad_overlay.l4_labels.summarize_annotations`, after every label passes
schema and deterministic cross-field validation.

| Modality | Mean confidence | Mean fusion weight |
| --- | ---: | ---: |
| text | 0.920000 | 0.360805 |
| speech | 0.845000 | 0.343923 |
| macro | 0.760000 | 0.295273 |
| micro | 0.000000 | 0.000000 |

## Issue #5 micro-review status

| Status | Count | L4 treatment |
| --- | ---: | --- |
| `negative` | 14 | micro VA, confidence, and fusion weight remain zero |
| `uncertain` | 6 | micro VA, confidence, and fusion weight remain zero pending second review |
| `positive` | 0 | no confirmed positive micro-expression is available in M1 |

Every L4 annotation is cross-checked against its corresponding
`annotations/micro_review/*_micro_review.json` file. The six uncertain samples
are `EAQ000004`, `EAQ000009`, `EAQ000013`, `EAQ000014`, `EAQ000015`, and
`EAQ000020`.

## Non-consistent and low-confidence samples

| EA ID | Type | Involved modalities |
| --- | --- | --- |
| `EAQ000008` | `intensity_mismatch` | text, macro |
| `EAQ000009` | `intensity_mismatch` | text, macro |
| `EAQ000012` | `hidden_emotion` | text, speech, macro |
| `EAQ000013` | `masking` | text, speech, macro |
| `EAQ000018` | `intensity_mismatch` | text, macro |

No sample has weighted `inter_va.confidence` below the report threshold of
`0.60`. Modality-level uncertainty is still retained in each JSON; in
particular, low Issue #4 face-detection coverage reduces the affected macro
confidence rather than being hidden by the aggregate value.

## Evidence and review limitations

- Source annotations and transcripts were checked against the 20 rows in
  `source_index/m1_sample_20.csv`.
- All 20 source videos were decoded and visually reviewed. All 20 audio streams
  were decoded with server-side ffmpeg, and their duration and level statistics
  were checked as speech-quality evidence.
- Issue #4 feature metadata was inspected for extraction status, face-detection
  coverage, and micro-candidate timing. Its generic embeddings were not treated
  as VA predictions.
- Issue #5 / PR #30 supplied 20 micro-review records: 14 `negative`, 6
  `uncertain`, and 0 `positive`. The L4 metadata now mirrors those statuses
  exactly. Because no positive event was confirmed, all micro VA values,
  confidences, and fusion weights remain zero; the six uncertain records are
  not treated as negative decisions.
- All 20 records use `review_status: single_pass_pending_second_review`; a
  second reviewer should resolve disagreements before a future adjudicated-gold
  release.

## Validation

Run:

```bash
python scripts/validate_m1_l4_labels.py
```

Result:

```text
OK: validated 20 L4 labels
```

The validator checks exact source-index coverage, filenames and identity,
required fields, legal enums, numeric ranges, unique involved modalities,
Issue #5 status agreement and zero/nonzero micro-signal semantics, unit-sum
weights, deterministic weight policy, and weighted `inter_va` equality.
