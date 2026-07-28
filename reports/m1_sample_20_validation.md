# M1 Sample 20 Validation

GitHub issue: <https://github.com/cuiyuxun-droid/ea-quad-overlay/issues/3>

## Scope

- Selected 20 seed samples for the M1 flow check.
- Dataset mix: 11 CH-SIMS samples and 9 MELD samples.
- Output index: `source_index/m1_sample_20.csv`.

## Server Sources

| Dataset | Source location |
| --- | --- |
| CH-SIMS labels | `/root/autodl-tmp/data/datasets/ch_sims/label.csv` |
| CH-SIMS media archive | `/root/autodl-tmp/data/datasets/ch_sims/Raw.zip` |
| MELD annotations | `/root/autodl-tmp/data/datasets/meld/annotations/` |
| MELD extracted media | `/root/autodl-tmp/data/datasets/meld/extracted/MELD.Raw/` |

## Checks Performed

- CH-SIMS `Raw.zip` completed at 8,819,485,007 bytes and was readable with `unzip -l`.
- Selected CH-SIMS videos were extracted temporarily from the archive and checked with `ffprobe`.
- Selected MELD videos were checked in place with `ffprobe`.
- All 20 selected samples had video and audio streams.
- Durations were recorded in the source index and are short utterance-level clips.
- A middle-frame contact sheet was reviewed for visible face quality.

## Validation

Run:

```bash
python scripts/validate_m1_sample_20.py
```

Expected result:

```text
OK: validated 20 rows (11 CH-SIMS, 9 MELD)
```
