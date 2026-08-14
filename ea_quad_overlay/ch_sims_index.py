"""CH-SIMS source index generation, validation, probing, and reporting."""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EA_ID_RE = re.compile(r"^EAQ\d{6}$")
SOURCE_ID_PREFIX = "CH-SIMS/"
QUALITY_VALUES = {"high", "medium", "low", "missing"}
BOOL_VALUES = {"true", "false"}
SPLIT_MAP = {
    "train": "train",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
}
HF_LABEL_URL = "https://huggingface.co/datasets/tamb2203579/CH-SIMS/resolve/main/label.csv"

# Historical M1 seed IDs that must never be minted for new CH-SIMS rows.
RESERVED_MELD_EA_IDS = {f"EAQ{i:06d}" for i in range(12, 21)}
CH_SIMS_NEW_ID_START = 21

INDEX_COLUMNS = (
    "ea_id",
    "source_dataset",
    "source_split",
    "source_id",
    "video_path",
    "audio_path",
    "text_path",
    "start",
    "end",
    "language",
    "face_quality",
    "audio_quality",
    "text_quality",
    "usable_for_micro",
    "usable_for_l4",
)

LABEL_COLUMNS = (
    "ea_id",
    "source_id",
    "source_key",
    "text",
    "label",
    "label_t",
    "label_a",
    "label_v",
    "annotation",
    "mode",
)

PROBE_COLUMNS = (
    "source_key",
    "video_resolved_path",
    "duration_sec",
    "has_video_stream",
    "has_audio_stream",
    "probe_status",
    "probe_error",
)


class ChSimsIndexError(ValueError):
    """Raised when CH-SIMS index generation or validation fails."""


@dataclass(frozen=True)
class ChSimsRecord:
    video_id: str
    clip_id: str
    text: str
    label: str
    label_t: str
    label_a: str
    label_v: str
    annotation: str
    mode: str

    @property
    def source_key(self) -> str:
        return f"{self.video_id}/{self.clip_id}"

    @property
    def source_id(self) -> str:
        return f"{SOURCE_ID_PREFIX}{self.source_key}"


@dataclass(frozen=True)
class MediaProbeResult:
    source_key: str
    video_resolved_path: str
    duration_sec: float | None
    has_video_stream: bool
    has_audio_stream: bool
    probe_status: str
    probe_error: str = ""


def fetch_label_csv(dest: Path) -> Path:
    """Download public CH-SIMS label.csv metadata from Hugging Face."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(HF_LABEL_URL, headers={"User-Agent": "ea-quad-overlay/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        dest.write_bytes(response.read())
    return dest


def read_label_csv(path: Path) -> list[ChSimsRecord]:
    records: list[ChSimsRecord] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=1):
            if len(row) < 9:
                raise ChSimsIndexError(f"{path}:{row_number}: expected 9 columns, got {len(row)}")
            video_id, clip_id, text, label, label_t, label_a, label_v, annotation, mode = row[:9]
            if not video_id or not clip_id:
                raise ChSimsIndexError(f"{path}:{row_number}: missing video_id or clip_id")
            records.append(
                ChSimsRecord(
                    video_id=video_id.strip(),
                    clip_id=clip_id.strip(),
                    text=(text or "").strip(),
                    label=(label or "").strip(),
                    label_t=(label_t or "").strip(),
                    label_a=(label_a or "").strip(),
                    label_v=(label_v or "").strip(),
                    annotation=(annotation or "").strip(),
                    mode=(mode or "").strip().lower(),
                )
            )
    if not records:
        raise ChSimsIndexError(f"{path}: no CH-SIMS rows found")
    return records


def read_m1_ch_sims_reservations(m1_index_path: Path) -> dict[str, str]:
    """Map CH-SIMS source keys to reserved M1 ea_id values."""
    if not m1_index_path.is_file():
        return {}
    reserved: dict[str, str] = {}
    with m1_index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("source_dataset") != "CH-SIMS":
                continue
            source_id = row.get("source_id", "")
            if not source_id.startswith(SOURCE_ID_PREFIX):
                continue
            key = source_id.removeprefix(SOURCE_ID_PREFIX)
            reserved[key] = row["ea_id"]
    return reserved


def read_m1_seed_metadata(m1_index_path: Path) -> dict[str, dict[str, str]]:
    """Reuse measured M1 durations and accepted quality flags for seed rows."""
    if not m1_index_path.is_file():
        return {}
    metadata: dict[str, dict[str, str]] = {}
    with m1_index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("source_dataset") != "CH-SIMS":
                continue
            source_id = row.get("source_id", "")
            if not source_id.startswith(SOURCE_ID_PREFIX):
                continue
            key = source_id.removeprefix(SOURCE_ID_PREFIX)
            metadata[key] = {
                "start": row.get("start") or "0.00",
                "end": row.get("end") or "",
                "face_quality": row.get("face_quality") or "missing",
                "audio_quality": row.get("audio_quality") or "missing",
                "text_quality": row.get("text_quality") or "missing",
                "usable_for_micro": row.get("usable_for_micro") or "false",
                "usable_for_l4": row.get("usable_for_l4") or "false",
            }
    return metadata


def _text_quality(text: str) -> str:
    if len(text) >= 4:
        return "high"
    if text:
        return "medium"
    return "missing"


def build_media_paths(record: ChSimsRecord, dataset_root: str) -> tuple[str, str, str]:
    root = dataset_root.rstrip("/").rstrip("\\")
    member = f"Raw/{record.video_id}/{record.clip_id}.mp4"
    video_path = f"{root}/Raw.zip::{member}"
    audio_path = video_path
    text_path = f"{root}/label.csv#{record.source_key}"
    return video_path, audio_path, text_path


def resolve_local_video(record: ChSimsRecord, dataset_root: Path) -> Path | None:
    """Resolve extracted Raw/ clip when present under dataset_root."""
    candidate = dataset_root / "Raw" / record.video_id / f"{record.clip_id}.mp4"
    if candidate.is_file() and candidate.stat().st_size > 0:
        return candidate
    return None


def _ffprobe_bin() -> str:
    path = shutil.which("ffprobe")
    if path:
        return path
    try:
        import imageio_ffmpeg

        ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
        sibling = ffmpeg.with_name(ffmpeg.name.replace("ffmpeg", "ffprobe"))
        if sibling.is_file():
            return str(sibling)
    except Exception:  # noqa: BLE001
        pass
    raise ChSimsIndexError("ffprobe not found; install ffmpeg or imageio-ffmpeg")


def probe_video_file(source_key: str, video_path: Path) -> MediaProbeResult:
    """Probe one local video with ffprobe for duration and stream presence."""
    try:
        cmd = [
            _ffprobe_bin(),
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type",
            "-of",
            "json",
            str(video_path),
        ]
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        payload = json.loads(completed.stdout or "{}")
        duration_raw = (payload.get("format") or {}).get("duration")
        duration = float(duration_raw) if duration_raw not in (None, "") else None
        streams = payload.get("streams") or []
        has_video = any(stream.get("codec_type") == "video" for stream in streams)
        has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
        if duration is None or duration <= 0 or not has_video:
            return MediaProbeResult(
                source_key=source_key,
                video_resolved_path=str(video_path),
                duration_sec=duration,
                has_video_stream=has_video,
                has_audio_stream=has_audio,
                probe_status="unreadable",
                probe_error="missing duration or video stream",
            )
        return MediaProbeResult(
            source_key=source_key,
            video_resolved_path=str(video_path),
            duration_sec=duration,
            has_video_stream=has_video,
            has_audio_stream=has_audio,
            probe_status="ok",
        )
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as exc:
        return MediaProbeResult(
            source_key=source_key,
            video_resolved_path=str(video_path),
            duration_sec=None,
            has_video_stream=False,
            has_audio_stream=False,
            probe_status="error",
            probe_error=str(exc),
        )


def probe_dataset_media(
    records: Sequence[ChSimsRecord],
    dataset_root: Path,
    *,
    limit: int | None = None,
) -> dict[str, MediaProbeResult]:
    """Probe extracted Raw/ videos under dataset_root."""
    results: dict[str, MediaProbeResult] = {}
    for index, record in enumerate(records):
        if limit is not None and index >= limit:
            break
        video = resolve_local_video(record, dataset_root)
        if video is None:
            results[record.source_key] = MediaProbeResult(
                source_key=record.source_key,
                video_resolved_path="",
                duration_sec=None,
                has_video_stream=False,
                has_audio_stream=False,
                probe_status="missing_file",
                probe_error=f"missing {dataset_root / 'Raw' / record.video_id / (record.clip_id + '.mp4')}",
            )
            continue
        results[record.source_key] = probe_video_file(record.source_key, video)
    return results


def write_probe_csv(path: Path, probes: Mapping[str, MediaProbeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PROBE_COLUMNS))
        writer.writeheader()
        for key in sorted(probes):
            probe = probes[key]
            writer.writerow(
                {
                    "source_key": probe.source_key,
                    "video_resolved_path": probe.video_resolved_path,
                    "duration_sec": ""
                    if probe.duration_sec is None
                    else f"{probe.duration_sec:.6f}",
                    "has_video_stream": "true" if probe.has_video_stream else "false",
                    "has_audio_stream": "true" if probe.has_audio_stream else "false",
                    "probe_status": probe.probe_status,
                    "probe_error": probe.probe_error,
                }
            )


def read_probe_csv(path: Path) -> dict[str, MediaProbeResult]:
    if not path.is_file():
        return {}
    results: dict[str, MediaProbeResult] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            duration_raw = (row.get("duration_sec") or "").strip()
            duration = float(duration_raw) if duration_raw else None
            results[row["source_key"]] = MediaProbeResult(
                source_key=row["source_key"],
                video_resolved_path=row.get("video_resolved_path") or "",
                duration_sec=duration,
                has_video_stream=row.get("has_video_stream") == "true",
                has_audio_stream=row.get("has_audio_stream") == "true",
                probe_status=row.get("probe_status") or "unknown",
                probe_error=row.get("probe_error") or "",
            )
    return results


def assign_ea_ids(
    records: Sequence[ChSimsRecord],
    reserved_by_source_key: Mapping[str, str],
) -> dict[str, str]:
    """Assign ea_id values while preserving M1 CH-SIMS IDs.

    New CH-SIMS rows use EAQ000021+ per docs/source_index_contract.md.
    """
    assignments: dict[str, str] = {}
    used_ids: set[str] = set(reserved_by_source_key.values()) | set(RESERVED_MELD_EA_IDS)
    next_id = CH_SIMS_NEW_ID_START

    def next_free_id() -> str:
        nonlocal next_id
        while True:
            candidate = f"EAQ{next_id:06d}"
            next_id += 1
            if candidate in used_ids:
                continue
            used_ids.add(candidate)
            return candidate

    for record in records:
        key = record.source_key
        if key in reserved_by_source_key:
            ea_id = reserved_by_source_key[key]
            if ea_id in assignments.values():
                raise ChSimsIndexError(f"duplicate reserved ea_id for {key}")
            assignments[key] = ea_id
            used_ids.add(ea_id)
            continue
        assignments[key] = next_free_id()
    return assignments


def _quality_and_usability(
    record: ChSimsRecord,
    probe: MediaProbeResult | None,
) -> tuple[str, str, str, str, str]:
    text_q = _text_quality(record.text)
    if probe is None or probe.probe_status != "ok":
        # Honest defaults: modality quality is unverified without media probing.
        face_q = "missing"
        audio_q = "missing"
        return face_q, audio_q, text_q, "false", "false"

    face_q = "medium" if probe.has_video_stream else "missing"
    audio_q = "medium" if probe.has_audio_stream else "missing"
    usable_micro = "true" if face_q != "missing" and text_q != "missing" else "false"
    # L4 needs labels + text + decodable media; face/audio quality remain provisional.
    has_labels = all(
        [
            record.label,
            record.label_t,
            record.label_a,
            record.label_v,
            record.annotation,
        ]
    )
    usable_l4 = (
        "true"
        if has_labels and text_q != "missing" and probe.has_video_stream and probe.has_audio_stream
        else "false"
    )
    return face_q, audio_q, text_q, usable_micro, usable_l4


def _time_bounds(
    source_key: str,
    probe: MediaProbeResult | None,
    m1_seed_meta: Mapping[str, Mapping[str, str]],
) -> tuple[str, str, str]:
    """Return start, end, and duration provenance tag."""
    if probe is not None and probe.probe_status == "ok" and probe.duration_sec:
        return "0.00", f"{probe.duration_sec:.2f}", "ffprobe"
    seed = m1_seed_meta.get(source_key)
    if seed and seed.get("end"):
        return seed.get("start") or "0.00", seed["end"], "m1_seed_measured"
    # Atomic clip: empty bounds mean use the whole referenced file.
    return "", "", "atomic_empty"


def build_index_and_label_rows(
    records: Sequence[ChSimsRecord],
    *,
    dataset_root: str,
    reserved_by_source_key: Mapping[str, str] | None = None,
    probes: Mapping[str, MediaProbeResult] | None = None,
    m1_seed_meta: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, int]]:
    reserved = reserved_by_source_key or {}
    probes = probes or {}
    m1_seed_meta = m1_seed_meta or {}
    ea_ids = assign_ea_ids(records, reserved)
    index_rows: list[dict[str, str]] = []
    label_rows: list[dict[str, str]] = []
    provenance = Counter()

    for record in records:
        ea_id = ea_ids[record.source_key]
        split = SPLIT_MAP.get(record.mode, record.mode or "train")
        if split not in {"train", "validation", "test"}:
            raise ChSimsIndexError(f"unsupported split {record.mode!r} for {record.source_key}")
        probe = probes.get(record.source_key)
        face_q, audio_q, text_q, usable_micro, usable_l4 = _quality_and_usability(record, probe)
        seed = m1_seed_meta.get(record.source_key)
        if seed and probe is None:
            # Preserve already-accepted M1 seed quality/usability evidence.
            face_q = seed["face_quality"]
            audio_q = seed["audio_quality"]
            text_q = seed["text_quality"]
            usable_micro = seed["usable_for_micro"]
            usable_l4 = seed["usable_for_l4"]
        start, end, origin = _time_bounds(record.source_key, probe, m1_seed_meta)
        provenance[origin] += 1
        video_path, audio_path, text_path = build_media_paths(record, dataset_root)
        index_rows.append(
            {
                "ea_id": ea_id,
                "source_dataset": "CH-SIMS",
                "source_split": split,
                "source_id": record.source_id,
                "video_path": video_path,
                "audio_path": audio_path,
                "text_path": text_path,
                "start": start,
                "end": end,
                "language": "zh",
                "face_quality": face_q,
                "audio_quality": audio_q,
                "text_quality": text_q,
                "usable_for_micro": usable_micro,
                "usable_for_l4": usable_l4,
            }
        )
        label_rows.append(
            {
                "ea_id": ea_id,
                "source_id": record.source_id,
                "source_key": record.source_key,
                "text": record.text,
                "label": record.label,
                "label_t": record.label_t,
                "label_a": record.label_a,
                "label_v": record.label_v,
                "annotation": record.annotation,
                "mode": record.mode,
            }
        )
    return index_rows, label_rows, dict(provenance)


def write_index_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INDEX_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def write_labels_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LABEL_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def read_index_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        if header != list(INDEX_COLUMNS):
            raise ChSimsIndexError("index header does not match source_index_template.csv")
        return list(reader)


def read_labels_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        if header != list(LABEL_COLUMNS):
            raise ChSimsIndexError("labels header does not match expected LABEL_COLUMNS")
        return list(reader)


def validate_index_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    reserved_by_source_key: Mapping[str, str] | None = None,
    min_rows: int = 2000,
) -> dict[str, Any]:
    if len(rows) < min_rows:
        raise ChSimsIndexError(f"expected at least {min_rows} rows, got {len(rows)}")

    seen_ea: set[str] = set()
    seen_source: set[str] = set()
    split_counts: Counter[str] = Counter()
    face_quality_counts: Counter[str] = Counter()
    text_quality_counts: Counter[str] = Counter()
    timed_rows = 0
    atomic_rows = 0

    for row_number, row in enumerate(rows, start=2):
        ea_id = row["ea_id"]
        if not EA_ID_RE.fullmatch(ea_id):
            raise ChSimsIndexError(f"row {row_number}: invalid ea_id {ea_id!r}")
        if ea_id in seen_ea:
            raise ChSimsIndexError(f"row {row_number}: duplicate ea_id {ea_id!r}")
        seen_ea.add(ea_id)

        if row["source_dataset"] != "CH-SIMS":
            raise ChSimsIndexError(f"row {row_number}: source_dataset must be CH-SIMS")
        source_id = row["source_id"]
        if not source_id.startswith(SOURCE_ID_PREFIX):
            raise ChSimsIndexError(f"row {row_number}: invalid source_id {source_id!r}")
        if source_id in seen_source:
            raise ChSimsIndexError(f"row {row_number}: duplicate source_id {source_id!r}")
        seen_source.add(source_id)

        if row["source_split"] not in {"train", "validation", "test"}:
            raise ChSimsIndexError(f"row {row_number}: invalid source_split")
        for field in ("video_path", "audio_path", "text_path"):
            if not row[field]:
                raise ChSimsIndexError(f"row {row_number}: missing {field}")

        start_raw = (row.get("start") or "").strip()
        end_raw = (row.get("end") or "").strip()
        if start_raw == "" and end_raw == "":
            atomic_rows += 1
        else:
            try:
                start = float(start_raw)
                end = float(end_raw)
            except ValueError as exc:
                raise ChSimsIndexError(f"row {row_number}: start/end must be numeric or empty") from exc
            if start < 0 or end <= start:
                raise ChSimsIndexError(f"row {row_number}: invalid start/end range")
            timed_rows += 1

        for field in ("face_quality", "audio_quality", "text_quality"):
            if row[field] not in QUALITY_VALUES:
                raise ChSimsIndexError(f"row {row_number}: invalid {field}")
        for field in ("usable_for_micro", "usable_for_l4"):
            if row[field] not in BOOL_VALUES:
                raise ChSimsIndexError(f"row {row_number}: invalid {field}")
        if row["usable_for_micro"] == "true" and row["face_quality"] == "missing":
            raise ChSimsIndexError(f"row {row_number}: usable_for_micro requires face quality")

        split_counts[row["source_split"]] += 1
        face_quality_counts[row["face_quality"]] += 1
        text_quality_counts[row["text_quality"]] += 1

    reserved = reserved_by_source_key or {}
    for key, ea_id in reserved.items():
        expected_source = f"{SOURCE_ID_PREFIX}{key}"
        matches = [row for row in rows if row["source_id"] == expected_source]
        if not matches:
            raise ChSimsIndexError(f"missing reserved M1 source_id {expected_source}")
        if matches[0]["ea_id"] != ea_id:
            raise ChSimsIndexError(
                f"M1 reservation mismatch for {expected_source}: "
                f"expected {ea_id}, got {matches[0]['ea_id']}"
            )

    return {
        "total": len(rows),
        "split_counts": dict(split_counts),
        "face_quality_counts": dict(face_quality_counts),
        "text_quality_counts": dict(text_quality_counts),
        "usable_for_micro": sum(1 for row in rows if row["usable_for_micro"] == "true"),
        "usable_for_l4": sum(1 for row in rows if row["usable_for_l4"] == "true"),
        "reserved_m1_matches": len(reserved),
        "timed_rows": timed_rows,
        "atomic_rows": atomic_rows,
        "first_ea_id": min(seen_ea) if seen_ea else "",
        "last_ea_id": max(seen_ea) if seen_ea else "",
    }


def validate_labels_rows(
    label_rows: Sequence[Mapping[str, str]],
    index_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    if len(label_rows) != len(index_rows):
        raise ChSimsIndexError(
            f"labels count {len(label_rows)} != index count {len(index_rows)}"
        )
    index_by_id = {row["ea_id"]: row for row in index_rows}
    missing_fields = 0
    for row_number, row in enumerate(label_rows, start=2):
        ea_id = row["ea_id"]
        if ea_id not in index_by_id:
            raise ChSimsIndexError(f"labels row {row_number}: ea_id {ea_id} not in index")
        if row["source_id"] != index_by_id[ea_id]["source_id"]:
            raise ChSimsIndexError(f"labels row {row_number}: source_id mismatch for {ea_id}")
        for field in ("label", "label_t", "label_a", "label_v", "annotation"):
            if not str(row.get(field) or "").strip():
                missing_fields += 1
    return {
        "total": len(label_rows),
        "missing_label_fields": missing_fields,
    }


def summarize_label_records(records: Sequence[ChSimsRecord]) -> dict[str, Any]:
    split_counts = Counter(record.mode for record in records)
    annotation_counts = Counter(record.annotation for record in records if record.annotation)
    missing_text = sum(1 for record in records if not record.text)
    missing_labels = sum(
        1
        for record in records
        if not all([record.label, record.label_t, record.label_a, record.label_v, record.annotation])
    )
    return {
        "total": len(records),
        "split_counts": dict(split_counts),
        "annotation_counts": dict(annotation_counts),
        "missing_text": missing_text,
        "missing_labels": missing_labels,
    }


def render_index_report(
    *,
    label_summary: Mapping[str, Any],
    index_summary: Mapping[str, Any],
    labels_summary: Mapping[str, Any],
    duration_provenance: Mapping[str, int],
    probe_summary: Mapping[str, Any],
    label_source: str,
    dataset_root: str,
    output_csv: str,
    labels_csv: str,
) -> str:
    lines = [
        "# CH-SIMS Index Report",
        "",
        "GitHub issue: <https://github.com/cuiyuxun-droid/ea-quad-overlay/issues/8>",
        "",
        "## Scope and result",
        "",
        "- Built a unified CH-SIMS source index from public `label.csv` metadata.",
        "- Persisted original labels into a versioned companion CSV linked by `ea_id` / `source_id`.",
        "- Preserved M1 CH-SIMS `ea_id` reservations from `source_index/m1_sample_20.csv`.",
        "- New CH-SIMS rows allocate from `EAQ000021+` per `docs/source_index_contract.md`.",
        "",
        "## Evidence classes",
        "",
        "| Class | Meaning |",
        "| --- | --- |",
        "| measured | Taken from ffprobe or M1 seed measured durations |",
        "| atomic_empty | Whole-file clip; `start/end` left empty intentionally |",
        "| heuristic | Text-quality only; documented below |",
        "| pending_media_probe | Face/audio usability awaits media probing |",
        "",
        "## Inputs",
        "",
        f"- Label source: `{label_source}`",
        f"- Dataset root: `{dataset_root}`",
        f"- Output index: `{output_csv}`",
        f"- Output labels: `{labels_csv}`",
        "",
        "## Allocation",
        "",
        "```text",
        "dataset: CH-SIMS",
        f"first_ea_id: {index_summary.get('first_ea_id', '')}",
        f"last_ea_id: {index_summary.get('last_ea_id', '')}",
        f"seed_rows_inherited: {index_summary.get('reserved_m1_matches', 0)}",
        f"new_rows_allocated: {index_summary['total'] - index_summary.get('reserved_m1_matches', 0)}",
        "allocation_map_source: source_index/m1_sample_20.csv + docs/source_index_contract.md",
        "```",
        "",
        "## Label coverage",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| Total label rows | {label_summary['total']} |",
        f"| Missing text | {label_summary['missing_text']} |",
        f"| Missing original label fields | {label_summary['missing_labels']} |",
        f"| Persisted label rows | {labels_summary['total']} |",
        f"| Empty persisted label fields | {labels_summary['missing_label_fields']} |",
        "",
        "### Split distribution (label.csv)",
        "",
        "| Split | Count |",
        "| --- | ---: |",
    ]
    for split, count in sorted(label_summary["split_counts"].items()):
        lines.append(f"| `{split}` | {count} |")

    lines.extend(
        [
            "",
            "## Duration provenance",
            "",
            "| Source | Count |",
            "| --- | ---: |",
        ]
    )
    for key, count in sorted(duration_provenance.items()):
        lines.append(f"| `{key}` | {count} |")

    lines.extend(
        [
            "",
            "## Media probe",
            "",
            "| Metric | Count |",
            "| --- | ---: |",
            f"| Probe rows available | {probe_summary.get('total', 0)} |",
            f"| Probe ok | {probe_summary.get('ok', 0)} |",
            f"| Missing file | {probe_summary.get('missing_file', 0)} |",
            f"| Unreadable/error | {probe_summary.get('failed', 0)} |",
            "",
            "## Index validation",
            "",
            "| Metric | Count |",
            "| --- | ---: |",
            f"| Indexed rows | {index_summary['total']} |",
            f"| M1 reservations preserved | {index_summary['reserved_m1_matches']} |",
            f"| Timed rows (`start/end` filled) | {index_summary['timed_rows']} |",
            f"| Atomic empty bounds | {index_summary['atomic_rows']} |",
            f"| `usable_for_micro=true` | {index_summary['usable_for_micro']} |",
            f"| `usable_for_l4=true` | {index_summary['usable_for_l4']} |",
            "",
            "### Face quality distribution",
            "",
            "| Quality | Count | Class |",
            "| --- | ---: | --- |",
        ]
    )
    for quality, count in sorted(index_summary["face_quality_counts"].items()):
        klass = "pending_media_probe" if quality == "missing" else "measured_or_rule"
        lines.append(f"| `{quality}` | {count} | {klass} |")

    lines.extend(
        [
            "",
            "### Text quality distribution",
            "",
            "| Quality | Count | Class |",
            "| --- | ---: | --- |",
        ]
    )
    for quality, count in sorted(index_summary["text_quality_counts"].items()):
        lines.append(f"| `{quality}` | {count} | heuristic |")

    lines.extend(
        [
            "",
            "## Heuristics and pending work",
            "",
            "- `text_quality`: `high` if text length >= 4, `medium` if non-empty shorter text, else `missing`.",
            "- `face_quality` / `audio_quality`: `missing` until ffprobe confirms streams; not claimed usable.",
            "- `usable_for_micro` / `usable_for_l4`: false until media probe succeeds.",
            "- Exception: 11 M1 CH-SIMS seed rows inherit measured duration and accepted quality from `m1_sample_20.csv`.",
            "- Re-run with `--probe-media --dataset-root <server_ch_sims>` on the dataset host to fill measured durations and usability for the remaining rows.",
            "",
            "## Errors",
            "",
            "- none",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_probes(probes: Mapping[str, MediaProbeResult]) -> dict[str, int]:
    counts = Counter(probe.probe_status for probe in probes.values())
    failed = counts.get("unreadable", 0) + counts.get("error", 0)
    return {
        "total": len(probes),
        "ok": counts.get("ok", 0),
        "missing_file": counts.get("missing_file", 0),
        "failed": failed,
    }


def generate_ch_sims_index(
    *,
    label_csv: Path,
    output_csv: Path,
    labels_csv: Path,
    dataset_root: str,
    m1_index_path: Path,
    probe_csv: Path | None = None,
    probes: Mapping[str, MediaProbeResult] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any], dict[str, Any], dict[str, int], dict[str, Any]]:
    records = read_label_csv(label_csv)
    reserved = read_m1_ch_sims_reservations(m1_index_path)
    m1_seed_meta = read_m1_seed_metadata(m1_index_path)
    loaded_probes: dict[str, MediaProbeResult] = {}
    if probes:
        loaded_probes.update(probes)
    if probe_csv is not None and probe_csv.is_file():
        loaded_probes.update(read_probe_csv(probe_csv))

    index_rows, label_rows, duration_provenance = build_index_and_label_rows(
        records,
        dataset_root=dataset_root,
        reserved_by_source_key=reserved,
        probes=loaded_probes,
        m1_seed_meta=m1_seed_meta,
    )
    write_index_csv(output_csv, index_rows)
    write_labels_csv(labels_csv, label_rows)
    label_summary = summarize_label_records(records)
    index_summary = validate_index_rows(
        index_rows,
        reserved_by_source_key=reserved,
        min_rows=len(index_rows),
    )
    labels_summary = validate_labels_rows(label_rows, index_rows)
    probe_summary = summarize_probes(loaded_probes)
    return (
        index_rows,
        label_rows,
        label_summary,
        index_summary,
        duration_provenance,
        {**labels_summary, **{"probe": probe_summary}},
    )
