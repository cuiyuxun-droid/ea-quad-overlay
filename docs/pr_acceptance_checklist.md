# PR Acceptance Checklist

This checklist applies to source-index and feature-extraction PRs. GitHub's
`MERGEABLE` state only means that Git can apply the change without a textual
conflict; it does not establish data or pipeline correctness.

## Required PR Description

Every PR must state:

- the Issue and target output files;
- the index role (`seed`, `dataset`, `master`, or `derived`);
- the source data snapshot and generation command;
- the `ea_id` allocation range and inherited seed row count;
- confirmation that the allocation follows `docs/source_index_contract.md`;
- path roots used in generated files;
- unavailable or heuristic fields and their counts;
- exact local test and validator commands with results.

## Source-index PRs

### Single-PR checks

- [ ] CSV has the canonical required columns.
- [ ] `ea_id` format and per-file uniqueness pass.
- [ ] `(source_dataset, source_id)` is unique within the file.
- [ ] All rows are traceable to source annotations or metadata.
- [ ] `usable_for_micro` and `usable_for_l4` agree with required paths.
- [ ] Dataset-specific labels and extension columns are documented.
- [ ] Generated report counts match the CSV.
- [ ] Builder is reproducible from a documented input snapshot.
- [ ] Unit tests cover normal rows, missing media, duplicate rows, and seed rows.

### Merge checks

- [ ] Combined `ea_id` values are globally unique against `main` and related PRs.
- [ ] Repeated `(source_dataset, source_id)` pairs keep the same `ea_id`.
- [ ] Seed IDs and source mappings are inherited exactly.
- [ ] Allocation ranges do not overlap existing or concurrent allocations.
- [ ] Deterministic regeneration does not renumber existing source records.
- [ ] A combined validator or equivalent output is attached to the PR.

## Feature-extraction PRs

- [ ] Input index selection distinguishes seed from full indexes.
- [ ] Duplicate `ea_id` and duplicate source-record identities fail before extraction.
- [ ] Every `text_path` pointer form present in the indexes resolves to actual text.
- [ ] Archive media, segmented media, and missing media have tested outcomes.
- [ ] A segment beginning at `start=0` is clipped when `end>start`.
- [ ] `text`, `speech`, `macro`, and optional `micro` modes are independently tested.
- [ ] Existing complete features are skipped without loading their extractor.
- [ ] Failed and face-filtered samples appear in `feature_quality.csv` with reasons.
- [ ] Manifest paths are relative to the configured output root and unique by `ea_id`.
- [ ] A smoke test processes at least one real or fixture row from one dataset.
- [ ] The full test suite is run with the repository's test layout.

## Minimum Commands

Use commands appropriate to the PR, but include equivalent evidence for each:

```bash
python -m pytest tests -q
python -m pytest scripts/tests -q
python scripts/<dataset_validator>.py
git diff --check
```

For a multi-index change, also run a combined validation command that checks
global `ea_id` and source-record uniqueness. A per-file validator alone is not
merge evidence.

## Approval Outcomes

| Outcome | Meaning |
| --- | --- |
| Merge | All single-PR and merge checks pass; no unresolved P0/P1 findings. |
| Merge after revision | Implementation is close, but a documented P1 or missing test must be fixed. |
| Do not merge | Global identity, traceability, or extraction correctness is broken. |
