# CH-SIMS Index Report

GitHub issue: <https://github.com/cuiyuxun-droid/ea-quad-overlay/issues/8>

## Scope and result

- Built a unified CH-SIMS source index from public `label.csv` metadata.
- Persisted original labels into a versioned companion CSV linked by `ea_id` / `source_id`.
- Preserved M1 CH-SIMS `ea_id` reservations from `source_index/m1_sample_20.csv`.
- New CH-SIMS rows allocate from `EAQ000021+` per `docs/source_index_contract.md`.

## Evidence classes

| Class | Meaning |
| --- | --- |
| measured | Taken from ffprobe or M1 seed measured durations |
| atomic_empty | Whole-file clip; `start/end` left empty intentionally |
| heuristic | Text-quality only; documented below |
| pending_media_probe | Face/audio usability awaits media probing |

## Inputs

- Label source: `.cache\ch_sims\label.csv`
- Dataset root: `/root/autodl-tmp/data/datasets/ch_sims`
- Output index: `source_index\ch_sims_index.csv`
- Output labels: `source_index\ch_sims_labels.csv`

## Allocation

```text
dataset: CH-SIMS
first_ea_id: EAQ000001
last_ea_id: EAQ002290
seed_rows_inherited: 11
new_rows_allocated: 2270
allocation_map_source: source_index/m1_sample_20.csv + docs/source_index_contract.md
```

## Label coverage

| Metric | Count |
| --- | ---: |
| Total label rows | 2281 |
| Missing text | 0 |
| Missing original label fields | 0 |
| Persisted label rows | 2281 |
| Empty persisted label fields | 0 |

### Split distribution (label.csv)

| Split | Count |
| --- | ---: |
| `test` | 457 |
| `train` | 1368 |
| `valid` | 456 |

## Duration provenance

| Source | Count |
| --- | ---: |
| `atomic_empty` | 2270 |
| `m1_seed_measured` | 11 |

## Media probe

| Metric | Count |
| --- | ---: |
| Probe rows available | 0 |
| Probe ok | 0 |
| Missing file | 0 |
| Unreadable/error | 0 |

## Index validation

| Metric | Count |
| --- | ---: |
| Indexed rows | 2281 |
| M1 reservations preserved | 11 |
| Timed rows (`start/end` filled) | 11 |
| Atomic empty bounds | 2270 |
| `usable_for_micro=true` | 11 |
| `usable_for_l4=true` | 11 |

### Face quality distribution

| Quality | Count | Class |
| --- | ---: | --- |
| `high` | 11 | measured_or_rule |
| `missing` | 2270 | pending_media_probe |

### Text quality distribution

| Quality | Count | Class |
| --- | ---: | --- |
| `high` | 2263 | heuristic |
| `medium` | 18 | heuristic |

## Heuristics and pending work

- `text_quality`: `high` if text length >= 4, `medium` if non-empty shorter text, else `missing`.
- `face_quality` / `audio_quality`: `missing` until ffprobe confirms streams; not claimed usable.
- `usable_for_micro` / `usable_for_l4`: false until media probe succeeds.
- Exception: 11 M1 CH-SIMS seed rows inherit measured duration and accepted quality from `m1_sample_20.csv`.
- Re-run with `--probe-media --dataset-root <server_ch_sims>` on the dataset host to fill measured durations and usability for the remaining rows.

## Errors

- none
