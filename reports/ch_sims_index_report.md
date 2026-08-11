# CH-SIMS Index Report

GitHub issue: <https://github.com/cuiyuxun-droid/ea-quad-overlay/issues/8>

## Scope and result

- Built a unified CH-SIMS source index from public `label.csv` metadata.
- Preserved M1 CH-SIMS `ea_id` reservations from `source_index/m1_sample_20.csv`.
- Generated modality paths, quality flags, and split labels for FeatureBank ingestion.

## Inputs

- Label source: `.cache\ch_sims\label.csv`
- Dataset root: `/root/autodl-tmp/data/datasets/ch_sims`
- Output index: `source_index\ch_sims_index.csv`

## Label coverage

| Metric | Count |
| --- | ---: |
| Total label rows | 2281 |
| Missing text | 0 |

### Split distribution (label.csv)

| Split | Count |
| --- | ---: |
| `test` | 457 |
| `train` | 1368 |
| `valid` | 456 |

## Index validation

| Metric | Count |
| --- | ---: |
| Indexed rows | 2281 |
| M1 reservations preserved | 11 |
| `usable_for_micro=true` | 2281 |
| `usable_for_l4=true` | 2281 |

### Split distribution (index)

| Split | Count |
| --- | ---: |
| `test` | 457 |
| `train` | 1368 |
| `validation` | 456 |

### Text quality distribution

| Quality | Count |
| --- | ---: |
| `high` | 2263 |
| `medium` | 18 |

## Notes

- `start/end` are metadata estimates when media probing is unavailable.
- Re-run generation on the dataset host with media access to refine durations and face quality.

## Errors

- none
