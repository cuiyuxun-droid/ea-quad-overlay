#!/usr/bin/env python
"""Extract M1 four-modality features for the 20 seed samples."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Avoid broken CUDA contexts on some AutoDL containers.
if "--device" in sys.argv:
    try:
        idx = sys.argv.index("--device")
        if idx + 1 < len(sys.argv) and sys.argv[idx + 1] == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
    except ValueError:
        pass
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

from ea_features.io_utils import FEATURE_ROOT, feature_exists, save_feature
from ea_features.media import MediaResolver, read_source_index


DEFAULT_INDEX = ROOT / "source_index" / "m1_sample_20.csv"
FAILURES_PATH = ROOT / "reports" / "m1_feature_failures.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument(
        "--modalities",
        default="text,speech,macro,micro",
        help="Comma-separated modalities to extract",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for smoke tests")
    parser.add_argument("--device", default=None, help="cpu/cuda override for torch models")
    return parser.parse_args()


def pick_torch_device(preferred: str | None) -> str:
    import torch

    if preferred:
        return preferred
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return "cuda"
        except Exception:  # noqa: BLE001
            return "cpu"
    return "cpu"


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        import huggingface_hub.constants as hf_constants

        hf_constants.HF_HUB_ENABLE_HF_TRANSFER = False
        if "hf-mirror.com" in os.environ.get("HF_ENDPOINT", ""):
            hf_constants.ENDPOINT = os.environ["HF_ENDPOINT"].rstrip("/")
    except Exception:  # noqa: BLE001
        pass

    torch_device = pick_torch_device(args.device)
    print(f"Using torch device: {torch_device}", flush=True)
    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    rows = read_source_index(args.index)
    if args.limit > 0:
        rows = rows[: args.limit]

    for modality in modalities:
        (FEATURE_ROOT / modality).mkdir(parents=True, exist_ok=True)
    FAILURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    resolver = MediaResolver()
    text_ex = None
    speech_ex = None
    macro_ex = None
    micro_ex = None

    def ensure_extractor(modality: str) -> None:
        nonlocal text_ex, speech_ex, macro_ex, micro_ex
        if modality == "text" and text_ex is None:
            from ea_features.text_extract import TextExtractor

            print("  loading text extractor...", flush=True)
            text_ex = TextExtractor(device=torch_device)
        elif modality == "speech" and speech_ex is None:
            from ea_features.speech_extract import SpeechExtractor

            print("  loading speech extractor...", flush=True)
            speech_ex = SpeechExtractor(device=torch_device)
        elif modality == "macro" and macro_ex is None:
            from ea_features.macro_extract import MacroExtractor

            print("  loading macro extractor...", flush=True)
            macro_ex = MacroExtractor(device=torch_device if torch_device in {"cpu", "cuda"} else "cpu")
        elif modality == "micro" and micro_ex is None:
            from ea_features.micro_extract import MicroExtractor

            print("  loading micro extractor...", flush=True)
            micro_ex = MicroExtractor()

    failures: list[dict[str, str]] = []
    success = 0

    try:
        for row in rows:
            ea_id = row["ea_id"]
            print(f"[extract] {ea_id}", flush=True)
            try:
                media = resolver.resolve_row(row)
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "ea_id": ea_id,
                        "modality": "media",
                        "error": str(exc),
                        "trace": traceback.format_exc(limit=3),
                    }
                )
                print(f"  FAIL media: {exc}", flush=True)
                continue

            row_ok = True
            for modality in modalities:
                if args.skip_existing and feature_exists(ea_id, modality):
                    print(f"  skip {modality}", flush=True)
                    continue
                try:
                    ensure_extractor(modality)
                    if modality == "text":
                        assert text_ex is not None
                        vec, meta = text_ex.extract(media.text)
                    elif modality == "speech":
                        assert speech_ex is not None
                        vec, meta = speech_ex.extract(media.audio_path)
                    elif modality == "macro":
                        assert macro_ex is not None
                        vec, meta = macro_ex.extract(media.video_path)
                    elif modality == "micro":
                        assert micro_ex is not None
                        if row.get("usable_for_micro", "true").lower() != "true":
                            import numpy as np

                            vec = np.zeros((16,), dtype=np.float32)
                            meta = {
                                "model": "face_roi_framediff_opticalflow_v1",
                                "status": "not_usable_for_micro",
                                "skipped": True,
                                "candidate_score": 0.0,
                            }
                        else:
                            vec, meta = micro_ex.extract(media.video_path)
                    else:
                        raise ValueError(f"unknown modality: {modality}")

                    save_feature(ea_id, modality, vec, meta)
                    print(
                        f"  ok {modality} shape={tuple(vec.shape)} status={meta.get('status')}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    row_ok = False
                    failures.append(
                        {
                            "ea_id": ea_id,
                            "modality": modality,
                            "error": str(exc),
                            "trace": traceback.format_exc(limit=3),
                        }
                    )
                    print(f"  FAIL {modality}: {exc}", flush=True)
            if row_ok:
                success += 1
    finally:
        if macro_ex is not None:
            macro_ex.close()
        if micro_ex is not None:
            micro_ex.close()

    with FAILURES_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ea_id", "modality", "error", "trace"])
        writer.writeheader()
        writer.writerows(failures)

    print(f"Done. success_rows={success}/{len(rows)} failures={len(failures)}", flush=True)
    print(f"Failures written to {FAILURES_PATH}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
