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
