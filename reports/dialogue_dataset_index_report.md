# Dialogue Dataset Index Report (MELD / MUStARD)

GitHub issue: <https://github.com/cuiyuxun-droid/ea-quad-overlay/issues/9>

## Scope

- Build utterance-level `source_index` for MELD and MUStARD.
- Mark face/audio/text usability for micro and L4 workflows.
- Explicitly flag sarcasm candidates via extension columns.

## Outputs

- `source_index/meld_index.csv`
- `source_index/mustard_index.csv`

## Path Roots

- MELD path root (written into CSV): `/root/autodl-tmp/data/datasets/meld`
- MUStARD path root (written into CSV): `/root/autodl-tmp/data/datasets/mustard`
- MELD annotations used for generation: `data/m1/meld`
- MUStARD JSON used for generation: `.cache/mustard/sarcasm_data.json`
- `--check-media`: `false`

## EA ID Ranges

- MELD: `EAQ100000` … `EAQ107400` (start `EAQ100000`)
- MUStARD: `EAQ200000` … `EAQ200689` (start `EAQ200000`)
- Reserved separately from M1 seed IDs `EAQ000001`–`EAQ000020`.

## Schema Extension

Canonical template columns plus:

- `is_sarcasm_candidate`: `true` / `false`
- `candidate_reason`: `meld_emotion_sentiment_mismatch` or `mustard_label`

## MELD Summary

- Total utterances: **7401**
- Split counts: `{'dev': 1109, 'test': 2610, 'train': 3682}`
- Face quality: `{'medium': 7401}`
- Sarcasm candidates (emotion/sentiment polarity mismatch): **469**
- `usable_for_l4=true`: **7401**
- `usable_for_micro=true`: **6883**

### Traceability example

- `source_id`: `MELD/train/dia0/utt0`
- `text_path`: `{annotations}/train_sent_emo.csv#Dialogue_ID=0&Utterance_ID=0`
- `video_path`: `{extracted}/MELD.Raw/train_splits/dia0_utt0.mp4`

## MUStARD Summary

- Total clips: **690**
- Split counts: `{'all': 690}`
- Face quality: `{'medium': 690}`
- Sarcasm candidates (official label): **345**
- `usable_for_l4=true`: **690**
- `usable_for_micro=true`: **690**

### Traceability example

- `source_id`: original MUStARD utterance key (e.g. `1_60`)
- `text_path`: `sarcasm_data.json#utterance_id=1_60`
- `video_path`: `{mustard_root}/utterances_final/{id}.mp4` (canonical guess)

## Cross-check with M1 seed index

- M1 MELD source_ids: **9**
- Found in `meld_index.csv`: **9**
- Missing: **0**

## Selection / quality rules

1. Keep all annotated rows; do not drop low-quality samples.
2. MELD sarcasm candidate when Emotion polarity conflicts with Sentiment polarity (neutral either side is not a conflict).
3. MUStARD sarcasm candidate when official `sarcasm=true`.
4. Without local face detection, existing media is marked `face_quality=medium` (not `high`).
5. `usable_for_l4` requires usable text and (sarcasm candidate or non-missing face).

## Known limitations

- Full-corpus face detection was not run; face quality is heuristic.
- MUStARD video filenames vary by unpack layout; path is a best-effort canonical guess.
- MELD `end` uses subtitle StartTime/EndTime delta when available; not ffprobe duration.
- Server media existence was not verified unless `--check-media` was enabled on a machine that can see those files.
