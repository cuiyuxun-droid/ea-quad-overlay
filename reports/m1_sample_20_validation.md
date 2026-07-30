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

## Selection Rationale

The source annotation pointer below is the `text_path` entry in
`source_index/m1_sample_20.csv`. It should be used to re-check the original
emotion or sentiment label before promoting these rows into a gold annotation
set.

| EA ID | Source ID | Source annotation pointer | Duration | Rationale |
| --- | --- | --- | --- | --- |
| `EAQ000001` | `CH-SIMS/video_0001/0001` | `label.csv#video_0001/0001` | 1.32s | Clear face/audio/text signals in a compact CH-SIMS utterance; useful as a short Chinese flow-check sample. |
| `EAQ000002` | `CH-SIMS/video_0001/0002` | `label.csv#video_0001/0002` | 2.15s | Clear multimodal signals and short duration; source label can be rechecked for emotional salience. |
| `EAQ000003` | `CH-SIMS/video_0001/0003` | `label.csv#video_0001/0003` | 3.33s | Test-split CH-SIMS sample with usable face/audio/text; preserves early split coverage. |
| `EAQ000004` | `CH-SIMS/video_0001/0004` | `label.csv#video_0001/0004` | 3.93s | Medium-short CH-SIMS segment with clear modality quality; suitable for first-pass feature extraction. |
| `EAQ000005` | `CH-SIMS/video_0001/0005` | `label.csv#video_0001/0005` | 2.37s | Compact CH-SIMS utterance with complete source traceability and high recorded modality quality. |
| `EAQ000006` | `CH-SIMS/video_0001/0007` | `label.csv#video_0001/0007` | 5.97s | Longer CH-SIMS utterance while still within M1 range; gives enough temporal context for micro review. |
| `EAQ000007` | `CH-SIMS/video_0001/0008` | `label.csv#video_0001/0008` | 2.73s | Clear Chinese utterance-level clip; useful for testing text/audio/video alignment. |
| `EAQ000008` | `CH-SIMS/video_0001/0010` | `label.csv#video_0001/0010` | 3.33s | High-quality CH-SIMS clip with full source lookup path; candidate for modality-agreement checks. |
| `EAQ000009` | `CH-SIMS/video_0001/0011` | `label.csv#video_0001/0011` | 2.15s | Short CH-SIMS segment with complete modalities; keeps early ID sequence dense for pipeline smoke tests. |
| `EAQ000010` | `CH-SIMS/video_0001/0017` | `label.csv#video_0001/0017` | 4.69s | Test-split CH-SIMS sample with visible face quality; broadens split coverage in the seed set. |
| `EAQ000011` | `CH-SIMS/video_0001/0018` | `label.csv#video_0001/0018` | 3.09s | Clear CH-SIMS utterance with high-quality modalities and direct label lookup. |
| `EAQ000012` | `MELD/train/dia0/utt4` | `train_sent_emo.csv#Dialogue_ID=0&Utterance_ID=4` | 6.49s | Longer MELD train utterance with full media paths; good for testing English dialogue context handling. |
| `EAQ000013` | `MELD/train/dia0/utt10` | `train_sent_emo.csv#Dialogue_ID=0&Utterance_ID=10` | 2.02s | Short MELD train utterance with complete annotation pointer; useful for baseline flow checks. |
| `EAQ000014` | `MELD/train/dia0/utt12` | `train_sent_emo.csv#Dialogue_ID=0&Utterance_ID=12` | 3.09s | Medium-short MELD train utterance; keeps dialogue-local coverage within `dia0`. |
| `EAQ000015` | `MELD/train/dia1/utt0` | `train_sent_emo.csv#Dialogue_ID=1&Utterance_ID=0` | 2.48s | MELD train sample from a second dialogue; checks source indexing beyond `dia0`. |
| `EAQ000016` | `MELD/train/dia1/utt1` | `train_sent_emo.csv#Dialogue_ID=1&Utterance_ID=1` | 2.23s | Adjacent MELD utterance with clear modalities; useful for dialogue-continuity sanity checks. |
| `EAQ000017` | `MELD/test/dia0/utt0` | `test_sent_emo.csv#Dialogue_ID=0&Utterance_ID=0` | 2.27s | MELD test-split sample with direct source traceability; adds split diversity. |
| `EAQ000018` | `MELD/test/dia0/utt1` | `test_sent_emo.csv#Dialogue_ID=0&Utterance_ID=1` | 6.78s | Longer MELD test utterance; provides enough temporal context for face/audio review. |
| `EAQ000019` | `MELD/test/dia1/utt2` | `test_sent_emo.csv#Dialogue_ID=1&Utterance_ID=2` | 3.19s | MELD test sample from another dialogue; supports cross-dialogue indexing checks. |
| `EAQ000020` | `MELD/test/dia2/utt4` | `test_sent_emo.csv#Dialogue_ID=2&Utterance_ID=4` | 3.44s | MELD test sample with distinct dialogue and utterance IDs; rounds out the 9-row MELD seed subset. |

## Validation

Run:

```bash
python scripts/validate_m1_sample_20.py
```

Expected result:

```text
OK: validated 20 rows (11 CH-SIMS, 9 MELD)
```
