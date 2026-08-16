# Dialogue Dataset Index Report (MELD / MUStARD)

GitHub issue: <https://github.com/cuiyuxun-droid/ea-quad-overlay/issues/9>

## Scope

- Build utterance-level `source_index` for MELD and MUStARD.
- Mark face/audio/text usability for micro and L4 workflows.
- Explicitly flag sarcasm candidates via extension columns.
- Preserve M1 MELD seed `ea_id` values per `docs/source_index_contract.md`.

## Outputs

- `source_index/meld_index.csv`
- `source_index/mustard_index.csv`
- allocation map: `source_index/meld_mustard_ea_id_map.csv`

## Path Roots

- MELD path root (written into CSV): `/root/autodl-tmp/data/datasets/meld`
- MUStARD path root (written into CSV): `/root/autodl-tmp/data/datasets/mustard`
- MELD annotations used for generation: `data/m1/meld`
- MUStARD JSON used for generation: `.cache/mustard/sarcasm_data.json`
- `--check-media`: `false`
- generation command: `python scripts/build_meld_mustard_index.py --fetch-mustard`

## Allocation

```text
dataset: MELD
first_ea_id: EAQ000012
last_ea_id: EAQ107400
seed_rows_inherited: 9
new_rows_allocated: 7392
new_ea_id_first: EAQ100000
new_ea_id_last: EAQ107400
allocation_map_source: source_index/m1_sample_20.csv + source_index/meld_mustard_ea_id_map.csv + docs/source_index_contract.md
```

```text
dataset: MUStARD
first_ea_id: EAQ200000
last_ea_id: EAQ200689
seed_rows_inherited: 0
new_rows_allocated: 690
allocation_map_source: source_index/meld_mustard_ea_id_map.csv + docs/source_index_contract.md
```

## Schema Extension

Canonical template columns plus:

- `is_sarcasm_candidate`: `true` / `false`
- `candidate_reason`: `meld_emotion_sentiment_mismatch` or `mustard_label`

## MELD Summary

- Total utterances: **7401**
- Split counts: `{'dev': 1109, 'test': 2610, 'train': 3682}`
- Face quality: `{'high': 9, 'missing': 7392}`
- Media unverified or missing (`face_quality=missing`): **7392**
- Sarcasm candidates (emotion/sentiment polarity mismatch): **469**
- `usable_for_l4=true`: **477**
- `usable_for_micro=true`: **9**

### Traceability example

- `source_id`: `MELD/train/dia0/utt0`
- `text_path`: `{annotations}/train_sent_emo.csv#Dialogue_ID=0&Utterance_ID=0`
- `video_path`: `{extracted}/MELD.Raw/train_splits/dia0_utt0.mp4`
- `video_path` (dev): `{extracted}/MELD.Raw/dev_splits_complete/dia*_utt*.mp4`

## MUStARD Summary

- Total clips: **690**
- Split counts: `{'all': 690}`
- Face quality: `{'missing': 690}`
- Media unverified or missing (`face_quality=missing`): **690**
- Sarcasm candidates (official label): **345**
- `usable_for_l4=true`: **345**
- `usable_for_micro=true`: **0**

### Traceability example

- `source_id`: original MUStARD utterance key (e.g. `1_60`)
- `text_path`: `{mustard_root}/data/sarcasm_data.json#utterance_id=1_60`
- `video_path`: `{mustard_root}/raw/clips/utterances_final/{id}.mp4`

## Cross-check with M1 seed index

- M1 MELD source_ids: **9**
- Found in `meld_index.csv`: **9**
- Seed ea_id preserved: **9**
- Missing: **0**
- Remapped away from seed range: **0**

## Selection / quality rules

1. Keep all annotated rows; do not drop low-quality samples.
2. MELD sarcasm candidate when Emotion polarity conflicts with Sentiment polarity (neutral either side is not a conflict).
3. MUStARD sarcasm candidate when official `sarcasm=true`.
4. Unchecked media (`--check-media` off) => `face_quality=missing`, `usable_for_micro=false` (contract).
5. M1 MELD seed rows inherit measured duration and accepted quality from `m1_sample_20.csv`.
6. `usable_for_l4` requires usable text and (sarcasm candidate or non-missing face).
7. EA IDs persist via allocation map; inserting new source rows does not renumber existing assignments.

## Known limitations

- Full-corpus face detection was not run.
- Default generation does not verify server media existence; re-run on the dataset host with `--check-media` after mounting the AutoDL paths to flip verified rows.
- MUStARD `start/end` are empty (atomic clip files).
