# Source Index Contract

This document is the repository-level contract for every file under
`source_index/`. Dataset-specific builders may add columns, but they must not
change the identity, traceability, or path rules below.

## Scope

The contract applies to seed indexes such as `m1_sample_20.csv`, dataset indexes
such as `ch_sims_index.csv`, `meld_index.csv`, `iemocap_index.csv`,
`mosei_index.csv`, and `mustard_index.csv`, the future merged
`master_source_index.csv`, and derived candidate or split indexes.

The canonical column definitions remain in `source_index/README.md`. This
document adds the cross-file rules that cannot be validated from one CSV alone.

## Index Roles

| Role | Example | Identity rule | Feature input |
| --- | --- | --- | --- |
| `seed` | `m1_sample_20.csv` | Immutable historical IDs and source mappings. | Only when explicitly requested. |
| `dataset` | `meld_index.csv` | One row per source record; matching seed IDs are inherited. | Yes, explicitly or through the master index. |
| `master` | `master_source_index.csv` | One canonical row per source record across all datasets. | Default input for full-corpus extraction. |
| `derived` | `l4_candidate_pool.csv` | A subset or projection that retains the original `ea_id`. | Only when explicitly requested. |

The repository currently contains seed and dataset indexes but no committed
master index. Until a master index exists, full-corpus tools must receive the
dataset index paths explicitly and must not treat every CSV in the directory as
independent samples.

## Global Identity

### `ea_id`

`ea_id` is a global primary key, not a per-dataset row number.

- Format: `EAQ` followed by six decimal digits.
- An ID is assigned once and is never reused or reassigned.
- All source indexes combined must have unique `ea_id` values.
- A dataset builder must read the existing ID registry or accept an explicit
  allocation map. It must never start at `EAQ000001` without checking existing
  assignments.
- A seed row keeps its original ID when copied into a dataset or master index.
- A derived index copies the original ID; it must not mint a new ID for the same
  source record.

### Source-record identity

The stable source identity is the pair `(source_dataset, source_id)`. This pair
must be unique across the merged corpus. The same pair must always map to the
same `ea_id`. A duplicate pair with a different `ea_id` is an identity error
even when all `ea_id` values are technically unique.

### Allocation records

New rows use the following dataset ranges. Historical seed IDs are the only
exception: when a seed row belongs to a dataset outside its normal range, the
dataset index must preserve that historical ID.

| Dataset | New-row range | Notes |
| --- | --- | --- |
| Historical M1 seed | `EAQ000001` - `EAQ000020` | Immutable; never allocate these IDs again. |
| CH-SIMS | `EAQ000021` - `EAQ099999` | Preserve CH-SIMS seed IDs `EAQ000001` - `EAQ000011`. |
| MELD | `EAQ100000` - `EAQ199999` | Preserve MELD seed IDs `EAQ000012` - `EAQ000020`. |
| MUStARD | `EAQ200000` - `EAQ299999` | No current seed exception. |
| IEMOCAP | `EAQ300000` - `EAQ399999` | No current seed exception. |
| MOSEI | `EAQ400000` - `EAQ499999` | No current seed exception. |
| MOSI | `EAQ500000` - `EAQ599999` | Kept separate from MOSEI. |
| Reserved | `EAQ600000` - `EAQ999999` | Do not allocate without updating this contract. |

Within a range, assign new IDs deterministically from stable source-record
ordering. A builder must preserve existing assignments when the source snapshot
grows; inserting a new source record must not renumber previously indexed rows.

Every new dataset index must document its allocation in its report:

```text
dataset: IEMOCAP
first_ea_id: EAQ...
last_ea_id: EAQ...
seed_rows_inherited: <count>
new_rows_allocated: <count>
```

The report must also state the source of the allocation map and the command used
to generate the file.

## Cross-index Validation

Before merging a dataset index, run these checks against `main` plus all other
dataset indexes in the change set:

1. Every file has the canonical required columns.
2. Every `ea_id` matches `^EAQ[0-9]{6}$`.
3. `ea_id` is globally unique.
4. `(source_dataset, source_id)` is globally unique.
5. Repeated source records have identical `ea_id`, paths, and time bounds.
6. Every seed row is present in the owning dataset index with the same `ea_id`.
7. `usable_for_micro=true` has the required media and text paths.
8. `usable_for_l4=true` has the required paths and label fields.
9. Generated reports agree with CSV row counts and ID ranges.

A validator that checks only one CSV is necessary but not sufficient for merge
approval. A PR must include repository-level validator output or an equivalent
recorded command.

## Path Contract

Paths may be absolute server paths, repository-relative paths, or pointers into
an archive or text table. Builders must preserve the path as a traceable pointer;
resolvers must implement the pointer forms used by the indexes.

### Media paths

```text
/data/video.mp4
/data/Raw.zip::video_0001/0001.mp4
```

`zip::member` means the member must be extracted or read from that exact archive.
The resolver must reject missing archives and missing members with a recorded
failure reason.

`start` and `end` are seconds on the referenced media timeline. When `end` is
greater than `start`, extraction must use exactly that interval, including when
`start` is `0`. Empty bounds mean the referenced file is already atomic. A
builder must not invent duration estimates and mark them as measured values.

### Text paths

| Dataset | Format | Meaning |
| --- | --- | --- |
| CH-SIMS | `label.csv#video_0001/0001` | Lookup the clip key in the label table. |
| MELD | `train_sent_emo.csv#Dialogue_ID=0&Utterance_ID=0` | Lookup the dialogue utterance. |
| MUStARD | `sarcasm_data.json#utterance_id=1_10` | Lookup the JSON utterance ID. |
| IEMOCAP | `transcript.txt#Ses01F_impro01_F000` | Lookup the transcript line key. |

Each supported pointer form must have a resolver unit test and at least one
fixture row. Returning the pointer string itself is not valid text resolution.

## Quality and Usability

Quality values describe observed modality quality, not merely whether metadata
or a path string exists.

- `missing` means the modality cannot be resolved.
- `low`, `medium`, and `high` require a documented measurement or inspection
  rule.
- Heuristic values are allowed only when the dataset report names the heuristic
  and reports the affected row count.
- `usable_for_micro=true` requires resolvable video plus a documented face
  quality rule; transcript presence alone is insufficient.
- `usable_for_l4=true` requires the modalities and original labels needed by the
  downstream L4 task. Dataset-specific requirements belong in the report.

## Compatibility and Migration

Until `master_source_index.csv` is introduced:

- keep `m1_sample_20.csv` classified as `seed`;
- do not include a seed and its owning full dataset in the same extraction command;
- pass dataset index paths explicitly to batch extraction;
- preserve M1 IDs in CH-SIMS and MELD dataset indexes;
- record intentionally excluded or unavailable source rows in the report.

When the master index is added, it becomes the only default full-corpus input.
Seed and dataset files remain provenance and generation inputs, but are not
independently scanned for feature extraction.
