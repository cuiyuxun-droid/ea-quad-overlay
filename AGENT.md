# Agent Guide

## Repository Purpose

EA-Quad-Overlay is a Python 3.10+ workspace for building an L4 multimodal
emotion-analysis dataset. The pipeline covers ingestion, alignment, feature
extraction, annotation, review, and packaging for datasets such as CH-SIMS,
MELD, IEMOCAP, MOSEI, and MUStARD.

## Setup

Create an isolated Python environment, then install the project and the
dependencies needed for the task:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[text]"   # optional text models
python -m pip install -e ".[audio]"  # optional audio models
python -m pip install -e ".[video]"  # optional video models
python -m pip install -e ".[ml]"     # all modality dependencies
```

Do not install heavyweight modality dependencies unless the change requires
them.

## Repository Map

- `ea_quad_overlay/`: reusable package code.
- `scripts/`: pipeline entry points and extraction/review utilities.
- `tests/`: package and repository-level tests.
- `scripts/tests/`: script and feature-extraction tests.
- `configs/`: dataset defaults and per-run overrides.
- `source_index/`: source-to-EA identity mappings.
- `features/`, `annotations/`, `manifests/`, `reports/`: generated or curated
  pipeline artifacts.
- `docs/`: pipeline, file layout, source-index, and acceptance contracts.

Read the closest README and relevant contract before changing a subsystem.

## Engineering Conventions

- Keep code compatible with Python 3.10 or newer.
- Follow the existing module layout and prefer small, focused changes.
- Use `snake_case.py` for scripts and `test_*.py` for tests.
- Keep lines at or below 100 characters; Ruff enforces `E`, `F`, `I`, and `W`.
- Use `logging` in scripts instead of `print` for operational output.
- Accept configuration through YAML and existing CLI conventions. Do not
  hardcode machine-specific dataset paths.
- Make pipeline stages deterministic and idempotent where practical.
- Preserve existing user data and generated artifacts unless the task
  explicitly requires regeneration or removal.
- Add or update focused tests whenever behavior changes.

## Data and Identity Rules

Treat `docs/source_index_contract.md` as authoritative for source-index work.
In particular:

- `ea_id` is a global primary key with format `EAQ` plus six digits.
- Never allocate IDs without checking existing indexes and reserved ranges.
- Preserve historical seed IDs and stable `(source_dataset, source_id)`
  mappings.
- Do not scan seed and owning dataset indexes as independent samples in one
  extraction run.
- Preserve traceable archive, media, transcript, and time-bound pointers.
- Never commit raw licensed datasets, credentials, local absolute paths, model
  caches, or large generated files unless they are intentional repository
  artifacts.

For source-index or extraction changes, also follow
`docs/pr_acceptance_checklist.md`, even when work is pushed without a PR.

## Validation

Run the narrowest relevant tests while developing, then the applicable full
checks before handing off:

```bash
python -m pytest tests -q
python -m pytest scripts/tests -q
ruff check .
git diff --check
```

Run dataset-specific validators for changed indexes, labels, manifests, or
features. Multi-index changes also require repository-level checks for global
`ea_id` and source-record uniqueness.

If an optional dependency or external dataset prevents a check from running,
report the exact skipped command and reason. Do not present an unrun check as
passing.

## Change Hygiene

- Inspect `git status` before editing and do not overwrite unrelated work.
- Keep commits scoped to one coherent purpose.
- Document regeneration commands and source snapshots for generated data.
- Review diffs for accidental path leaks, binary files, encoding damage, and
  unrelated artifact churn before committing.
