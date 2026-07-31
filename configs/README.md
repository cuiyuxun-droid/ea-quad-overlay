# Configuration

Dataset-specific paths and parameters are defined in `dataset_defaults.yaml`.

## Usage Convention

1. **Default config** — `configs/dataset_defaults.yaml` contains all known datasets and default extraction parameters.
2. **Per-run overrides** — Create a separate YAML (e.g. `configs/run_v1_ie.yaml`) that only overrides the fields you need.
3. **CLI override** — Scripts should accept `--config` pointing to either a YAML file or a comma-separated `key=value` string.

## Adding a New Dataset

Add an entry under `datasets:` in `dataset_defaults.yaml` with:
- `base_path` — where the raw data lives
- `language` — `zh` or `en` (affects tokenizer selection)
- `modalities` — which modalities are available
- Any dataset-specific fields
