"""Reusable batch orchestration for FeatureBank extraction."""

from __future__ import annotations

import csv
import json
import logging
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ea_features.io_utils import (
    MODALITIES,
    feature_exists,
    feature_paths,
    save_feature,
    segment_id,
)
from ea_features.media import MediaResolver

LOGGER = logging.getLogger(__name__)
MANIFEST_SCHEMA_VERSION = "feature-bank-v1"
REQUIRED_INDEX_COLUMNS = {"ea_id", "source_dataset", "source_id", "source_split"}
QUALITY_FIELDS = (
    "source_index",
    "ea_id",
    "source_dataset",
    "requested_modalities",
    "written_modalities",
    "existing_modalities",
    "filtered_modalities",
    "failed_modalities",
    "face_frames_sampled",
    "face_frames_detected",
    "face_detect_rate",
    "mean_face_ratio",
    "micro_eligible",
    "micro_filter_reason",
    "status",
    "failure_reasons",
)


class BatchExtractionError(ValueError):
    """Raised when batch inputs violate the extraction contract."""


@dataclass(frozen=True)
class BatchConfig:
    root: Path
    output_root: Path
    index_paths: tuple[Path, ...]
    modalities: tuple[str, ...]
    skip_existing: bool = True
    limit: int = 0
    device: str = "cpu"
    face_filter: bool = True
    face_sample_frames: int = 12
    min_face_detect_rate: float = 0.5
    min_face_ratio: float = 0.01
    report_path: Path | None = None
    manifest_path: Path | None = None
    cache_root: Path | None = None

    def __post_init__(self) -> None:
        if not self.index_paths:
            raise BatchExtractionError("at least one source index is required")
        if not self.modalities:
            raise BatchExtractionError("at least one modality is required")
        if self.limit < 0:
            raise BatchExtractionError("limit must be non-negative")
        if self.face_sample_frames <= 0:
            raise BatchExtractionError("face_sample_frames must be positive")
        if not 0.0 <= self.min_face_detect_rate <= 1.0:
            raise BatchExtractionError("min_face_detect_rate must be between 0 and 1")
        if not 0.0 <= self.min_face_ratio <= 1.0:
            raise BatchExtractionError("min_face_ratio must be between 0 and 1")

    @property
    def features_root(self) -> Path:
        return self.output_root / "features"

    @property
    def resolved_report_path(self) -> Path:
        return self.report_path or self.output_root / "reports" / "feature_quality.csv"

    @property
    def resolved_manifest_path(self) -> Path:
        return self.manifest_path or self.output_root / "manifests" / "feature_bank.jsonl"


def parse_modalities(raw: str | Sequence[str]) -> tuple[str, ...]:
    values = raw.split(",") if isinstance(raw, str) else list(raw)
    parsed: list[str] = []
    for value in values:
        modality = str(value).strip().lower()
        if not modality:
            continue
        if modality not in MODALITIES:
            raise BatchExtractionError(
                f"unsupported modality {modality!r}; choose from {', '.join(MODALITIES)}"
            )
        if modality not in parsed:
            parsed.append(modality)
    if not parsed:
        raise BatchExtractionError("at least one modality is required")
    return tuple(parsed)


def discover_index_paths(index_dir: Path) -> tuple[Path, ...]:
    paths = tuple(
        path for path in sorted(index_dir.glob("*.csv")) if "template" not in path.stem.lower()
    )
    if not paths:
        raise BatchExtractionError(f"no source index CSV files found in {index_dir}")
    return paths


def load_index_rows(
    index_paths: Sequence[Path],
    *,
    limit: int = 0,
) -> list[tuple[Path, dict[str, str]]]:
    loaded: list[tuple[Path, dict[str, str]]] = []
    seen: dict[str, Path] = {}
    for index_path in index_paths:
        if not index_path.is_file():
            raise BatchExtractionError(f"source index does not exist: {index_path}")
        with index_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            missing = REQUIRED_INDEX_COLUMNS - fields
            if missing:
                raise BatchExtractionError(
                    f"{index_path}: missing columns {', '.join(sorted(missing))}"
                )
            for row in reader:
                ea_id = (row.get("ea_id") or "").strip()
                if not ea_id:
                    raise BatchExtractionError(f"{index_path}: row has empty ea_id")
                if ea_id in seen:
                    raise BatchExtractionError(
                        f"duplicate ea_id {ea_id!r} in {seen[ea_id]} and {index_path}"
                    )
                seen[ea_id] = index_path
                loaded.append((index_path, {key: value or "" for key, value in row.items()}))
                if limit > 0 and len(loaded) >= limit:
                    return loaded
    if not loaded:
        raise BatchExtractionError("source indexes contain no samples")
    return loaded


def default_extractor_factories(device: str) -> dict[str, Callable[[], Any]]:
    def text_factory() -> Any:
        from ea_features.text_extract import TextExtractor

        return TextExtractor(device=device)

    def speech_factory() -> Any:
        from ea_features.speech_extract import SpeechExtractor

        return SpeechExtractor(device=device)

    def macro_factory() -> Any:
        from ea_features.macro_extract import MacroExtractor

        return MacroExtractor(device=device)

    def micro_factory() -> Any:
        from ea_features.micro_extract import MicroExtractor

        return MicroExtractor()

    return {
        "text": text_factory,
        "speech": speech_factory,
        "macro": macro_factory,
        "micro": micro_factory,
    }


def default_face_quality_checker(video_path: Path | None, **kwargs: Any) -> Mapping[str, Any]:
    from ea_features.face_utils import assess_face_quality

    return assess_face_quality(video_path, **kwargs).as_dict()


def _bool_value(value: str, *, default: bool = True) -> bool:
    clean = (value or "").strip().lower()
    if not clean:
        return default
    return clean in {"1", "true", "yes", "y"}


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _read_meta(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BatchExtractionError(f"invalid manifest JSON at {path}:{line_no}") from exc
            if not isinstance(record, dict) or not record.get("ea_id"):
                raise BatchExtractionError(f"invalid manifest record at {path}:{line_no}")
            records[str(record["ea_id"])] = record
    return records


def _write_manifest(path: Path, records: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for ea_id in sorted(records):
            handle.write(json.dumps(records[ea_id], ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    tmp.replace(path)


def _write_quality_report(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUALITY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


class BatchFeatureRunner:
    """Run modality extractors lazily and keep processing after row failures."""

    def __init__(
        self,
        config: BatchConfig,
        *,
        resolver: MediaResolver | None = None,
        extractor_factories: Mapping[str, Callable[[], Any]] | None = None,
        face_quality_checker: Callable[..., Mapping[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.resolver = resolver or MediaResolver(
            cache_root=config.cache_root or config.output_root / ".cache" / "media"
        )
        self.extractor_factories = dict(
            extractor_factories or default_extractor_factories(config.device)
        )
        self.face_quality_checker = face_quality_checker or default_face_quality_checker
        self.extractors: dict[str, Any] = {}
        self.extractor_load_errors: dict[str, str] = {}

    def _extractor(self, modality: str) -> Any:
        if modality in self.extractor_load_errors:
            raise RuntimeError(self.extractor_load_errors[modality])
        if modality not in self.extractors:
            try:
                LOGGER.info("Loading %s extractor", modality)
                self.extractors[modality] = self.extractor_factories[modality]()
            except Exception as exc:  # noqa: BLE001
                reason = f"extractor load failed: {exc}"
                self.extractor_load_errors[modality] = reason
                raise RuntimeError(reason) from exc
        return self.extractors[modality]

    def _resolve_media(
        self,
        row: Mapping[str, str],
        pending: Sequence[str],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        ea_id = row["ea_id"]
        sample_cache = self.resolver.cache_root / ea_id
        sample_cache.mkdir(parents=True, exist_ok=True)
        media: dict[str, Any] = {"text": "", "video": None, "audio": None}
        errors: dict[str, str] = {}

        if "text" in pending:
            try:
                media["text"] = self.resolver.resolve_text(
                    row.get("text_path", ""), row.get("source_dataset", "")
                )
            except Exception as exc:  # noqa: BLE001
                errors["text"] = f"text resolution failed: {exc}"

        video_needed = bool({"macro", "micro"} & set(pending))
        if video_needed:
            try:
                media["video"] = self.resolver.resolve_video(
                    row.get("video_path", ""),
                    sample_cache,
                    start=float(row.get("start") or 0.0),
                    end=float(row.get("end") or 0.0),
                )
            except Exception as exc:  # noqa: BLE001
                errors["video"] = f"video resolution failed: {exc}"

        if "speech" in pending:
            try:
                fallback_video = media["video"]
                if fallback_video is None and not row.get("audio_path", "").strip():
                    fallback_video = self.resolver.resolve_video(
                        row.get("video_path", ""),
                        sample_cache,
                        start=float(row.get("start") or 0.0),
                        end=float(row.get("end") or 0.0),
                    )
                media["audio"] = self.resolver.resolve_audio(
                    row.get("audio_path", ""),
                    fallback_video,
                    sample_cache,
                    start=float(row.get("start") or 0.0),
                    end=float(row.get("end") or 0.0),
                )
            except Exception as exc:  # noqa: BLE001
                errors["speech"] = f"audio resolution failed: {exc}"
        return media, errors

    def _existing_meta(self, ea_id: str, modality: str) -> dict[str, Any]:
        _, meta_path = feature_paths(
            ea_id,
            modality,
            feature_root=self.config.features_root,
        )
        return _read_meta(meta_path)

    def _manifest_record(
        self,
        index_path: Path,
        row: Mapping[str, str],
        requested: Sequence[str],
    ) -> dict[str, Any]:
        feature_map: dict[str, str] = {}
        meta_map: dict[str, str] = {}
        statuses: dict[str, str] = {}
        for modality in MODALITIES:
            npy_path, meta_path = feature_paths(
                row["ea_id"],
                modality,
                feature_root=self.config.features_root,
            )
            if not feature_exists(row["ea_id"], modality, feature_root=self.config.features_root):
                continue
            feature_map[modality] = _display_path(npy_path, self.config.output_root)
            meta_map[modality] = _display_path(meta_path, self.config.output_root)
            statuses[modality] = str(_read_meta(meta_path).get("status") or "unknown")
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "ea_id": row["ea_id"],
            "segment_id": segment_id(row["ea_id"]),
            "source_dataset": row["source_dataset"],
            "source_id": row["source_id"],
            "source_split": row["source_split"],
            "source_index": _display_path(index_path, self.config.root),
            "requested_modalities": list(requested),
            "feature_paths": feature_map,
            "meta_paths": meta_map,
            "feature_status": statuses,
        }

    def _process_row(self, index_path: Path, row: Mapping[str, str]) -> dict[str, Any]:
        ea_id = row["ea_id"]
        requested = self.config.modalities
        existing = [
            modality
            for modality in requested
            if self.config.skip_existing
            and feature_exists(ea_id, modality, feature_root=self.config.features_root)
        ]
        pending = [modality for modality in requested if modality not in existing]
        LOGGER.info("[%s] pending=%s existing=%s", ea_id, pending, existing)

        media, resolution_errors = self._resolve_media(row, pending)
        written: list[str] = []
        filtered: list[str] = []
        failed: dict[str, str] = {}
        face_quality: dict[str, Any] = {}

        for modality in existing:
            existing_status = str(self._existing_meta(ea_id, modality).get("status") or "unknown")
            if existing_status == "ok":
                continue
            if modality == "micro" and existing_status in {
                "not_usable_for_micro",
                "filtered_face_quality",
            }:
                filtered.append(modality)
            else:
                failed[modality] = f"existing extractor status={existing_status}"

        for modality in pending:
            if modality == "micro":
                if not _bool_value(row.get("usable_for_micro", ""), default=True):
                    filtered.append(modality)
                    face_quality = {
                        "usable_for_micro": False,
                        "filter_reason": "source_index_usable_for_micro=false",
                    }
                    continue
                if "video" in resolution_errors:
                    failed[modality] = resolution_errors["video"]
                    continue
                if self.config.face_filter:
                    try:
                        face_quality = dict(
                            self.face_quality_checker(
                                media["video"],
                                max_frames=self.config.face_sample_frames,
                                min_detect_rate=self.config.min_face_detect_rate,
                                min_face_ratio=self.config.min_face_ratio,
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        failed[modality] = f"face quality check failed: {exc}"
                        continue
                    if not bool(face_quality.get("usable_for_micro")):
                        filtered.append(modality)
                        continue

            resolution_key = "video" if modality == "macro" else modality
            if resolution_key in resolution_errors:
                failed[modality] = resolution_errors[resolution_key]
                continue

            try:
                extractor = self._extractor(modality)
                if modality == "text":
                    vector, meta = extractor.extract(media["text"])
                elif modality == "speech":
                    vector, meta = extractor.extract(media["audio"])
                else:
                    vector, meta = extractor.extract(media["video"])
                meta = {
                    **dict(meta),
                    "source_dataset": row["source_dataset"],
                    "source_id": row["source_id"],
                    "source_split": row["source_split"],
                    "source_index": _display_path(index_path, self.config.root),
                }
                if modality == "micro" and face_quality:
                    meta["face_quality"] = face_quality
                save_feature(
                    ea_id,
                    modality,
                    np.asarray(vector),
                    meta,
                    feature_root=self.config.features_root,
                    path_root=self.config.output_root,
                )
                written.append(modality)
                status = str(meta.get("status") or "unknown")
                if status != "ok":
                    failed[modality] = f"extractor status={status}"
            except Exception as exc:  # noqa: BLE001
                failed[modality] = str(exc)
                LOGGER.debug(
                    "Extraction failed for %s/%s\n%s", ea_id, modality, traceback.format_exc()
                )

        status = "failed" if failed and not (written or existing) else "partial" if failed else "ok"
        quality_row = {
            "source_index": _display_path(index_path, self.config.root),
            "ea_id": ea_id,
            "source_dataset": row["source_dataset"],
            "requested_modalities": ";".join(requested),
            "written_modalities": ";".join(written),
            "existing_modalities": ";".join(existing),
            "filtered_modalities": ";".join(filtered),
            "failed_modalities": ";".join(sorted(failed)),
            "face_frames_sampled": face_quality.get("frames_sampled", ""),
            "face_frames_detected": face_quality.get("frames_with_face", ""),
            "face_detect_rate": face_quality.get("face_detect_rate", ""),
            "mean_face_ratio": face_quality.get("mean_face_ratio", ""),
            "micro_eligible": face_quality.get("usable_for_micro", ""),
            "micro_filter_reason": face_quality.get("filter_reason", ""),
            "status": status,
            "failure_reasons": " | ".join(
                f"{modality}: {reason}" for modality, reason in sorted(failed.items())
            ),
        }
        return quality_row

    def run(self) -> dict[str, Any]:
        rows = load_index_rows(self.config.index_paths, limit=self.config.limit)
        manifest_path = self.config.resolved_manifest_path
        manifest = _load_manifest(manifest_path)
        quality_rows: list[dict[str, Any]] = []
        try:
            for index_path, row in rows:
                quality = self._process_row(index_path, row)
                quality_rows.append(quality)
                manifest[row["ea_id"]] = self._manifest_record(
                    index_path, row, self.config.modalities
                )
        finally:
            for extractor in self.extractors.values():
                close = getattr(extractor, "close", None)
                if callable(close):
                    close()

        _write_quality_report(self.config.resolved_report_path, quality_rows)
        _write_manifest(manifest_path, manifest)
        failures = sum(bool(row["failed_modalities"]) for row in quality_rows)
        filtered = sum(bool(row["filtered_modalities"]) for row in quality_rows)
        return {
            "samples": len(quality_rows),
            "failed_samples": failures,
            "filtered_samples": filtered,
            "report_path": self.config.resolved_report_path,
            "manifest_path": manifest_path,
        }
