# Source Index

`source_index_template.csv` is the canonical schema for mapping every EA sample
back to its original dataset record.

## Columns

| Column | Description |
| --- | --- |
| `ea_id` | Unified EA sample ID, for example `EAQ000001`. |
| `source_dataset` | Original dataset name: `CH-SIMS`, `MELD`, `IEMOCAP`, `MOSEI`, or `MUStARD`. |
| `source_split` | Original or derived split, such as `train`, `dev`, `validation`, `test`, or `all`. |
| `source_id` | Stable source-side identifier used to trace back to the raw dataset record. |
| `video_path` | Relative or external path to the source video, if available. |
| `audio_path` | Relative or external path to the source audio, if available. |
| `text_path` | Relative or external path to the transcript or text source, if available. |
| `start` | Segment start time in seconds, or empty if the source record is already atomic. |
| `end` | Segment end time in seconds, or empty if the source record is already atomic. |
| `language` | Language code or label, for example `zh`, `en`, or `mixed`. |
| `face_quality` | Face signal quality label, such as `high`, `medium`, `low`, or `missing`. |
| `audio_quality` | Audio signal quality label, such as `high`, `medium`, `low`, or `missing`. |
| `text_quality` | Text signal quality label, such as `high`, `medium`, `low`, or `missing`. |
| `usable_for_micro` | Whether the row can be used for micro-expression review: `true` or `false`. |
| `usable_for_l4` | Whether the row can be used for L4 labeling or evaluation: `true` or `false`. |

## Source ID Guidance

Use the original dataset's most stable identifier in `source_id`. If a dataset
requires a composite key, join the parts with `/` so it remains readable.

| Dataset | Suggested `source_id` pattern |
| --- | --- |
| `CH-SIMS` | Original clip or utterance ID. |
| `MELD` | `season/dialogue_id/utterance_id` or the dataset-provided utterance key. |
| `IEMOCAP` | Session/dialogue/utterance identifier, such as an original utterance filename stem. |
| `MOSEI` | Original video ID plus clip or segment identifier. |
| `MUStARD` | Original clip ID or statement ID. |
