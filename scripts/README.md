# Scripts

Reusable processing scripts for the EA-Quad-Overlay dataset pipeline.

## Structure

```
scripts/
├── README.md
├── ingest/          # Stage 1 — read raw datasets, assign EA IDs
├── align/           # Stage 2 — multi-modal alignment
├── extract/         # Stage 3 — feature extraction per modality
├── annotate/        # Stage 4 — micro-expression candidate generation
├── review/          # Stage 5 — review aggregation and adjudication
├── package/         # Stage 6 — L4 gold packaging
└── utils/           # Shared utilities (paths, io, logging, config)
```

## Conventions

1. **Configuration** — Every script accepts `--config` pointing to a YAML file.
   See `configs/` for defaults and `configs/README.md` for usage.

2. **Path handling** — Use the utilities in `scripts/utils/paths.py` (to be created)
   to resolve EA IDs to file paths. Never hardcode paths.

3. **Logging** — Use `logging` (NOT `print`). Log level defaults to INFO;
   pass `--verbose` for DEBUG.

4. **Output** — Write outputs to the paths dictated by `docs/file_structure.md`.
   Each script should be idempotent where possible.

5. **Testing** — Unit tests go in `tests/`, mirroring the `scripts/` structure.
   Run with: `pytest tests/`

6. **Naming** — Script names are `snake_case.py`. A script that processes a
   single dataset is named `{action}_{dataset}.py` (e.g. `ingest_ch_sims.py`).
   A multi-dataset script is named `{action}.py` (e.g. `ingest.py`).

## Unified FeatureBank extraction

`extract_features.py` reads one or more source-index CSV files, lazily loads
the requested extractors, and writes portable feature paths into
`manifests/feature_bank.jsonl`. It is incremental by default and records
per-sample quality, filter, and failure details in
`reports/feature_quality.csv`.

```bash
# Discover all source_index/*.csv files and extract all four modalities.
python scripts/extract_features.py --device cpu

# Process one dataset without forcing micro extraction.
python scripts/extract_features.py \
  --index source_index/iemocap_index.csv \
  --modalities text,speech,macro \
  --device cpu

# Isolated smoke run that does not write generated data into the repository.
python scripts/extract_features.py \
  --index source_index/m1_sample_20.csv \
  --output-root .work/feature-smoke \
  --modalities text \
  --limit 1 \
  --device cpu
```

Use `--overwrite` to recompute existing outputs and `--fail-on-error` when a
non-zero exit code is required if any sample fails. Micro extraction is
skipped when `usable_for_micro=false` or when the configurable face detection
rate / face-size thresholds are not met.
