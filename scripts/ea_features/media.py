"""Resolve video/audio/text assets from the unified source index."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs


ZIP_MEMBER_RE = re.compile(r"^(?P<zip>.+\.zip)::(?P<member>.+)$", re.IGNORECASE)
CH_SIMS_POINTER_RE = re.compile(r"^(.+#)?(?P<key>video_\d+/\d+)$")
MELD_POINTER_RE = re.compile(
    r"^(?P<csv>.+\.csv)#Dialogue_ID=(?P<dialog>\d+)&Utterance_ID=(?P<utt>\d+)$",
    re.IGNORECASE,
)
MUSTARD_POINTER_RE = re.compile(
    r"^(?P<json>.+\.json)#utterance_id=(?P<uid>[^#]+)$",
    re.IGNORECASE,
)


@dataclass
class ResolvedMedia:
    ea_id: str
    video_path: Path | None
    audio_path: Path | None
    text: str
    language: str
    start: float
    end: float
    cache_dir: Path


class MediaResolver:
    """Materialize source_index rows into local video/audio/text payloads."""

    def __init__(self, cache_root: Path | None = None) -> None:
        root = Path(__file__).resolve().parents[2]
        self.cache_root = cache_root or (root / ".cache" / "media")
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def resolve_row(self, row: dict[str, str]) -> ResolvedMedia:
        ea_id = row["ea_id"]
        sample_cache = self.cache_root / ea_id
        sample_cache.mkdir(parents=True, exist_ok=True)

        video_path = self.resolve_video(row.get("video_path", ""), sample_cache)
        audio_path = self.resolve_audio(
            row.get("audio_path", ""),
            video_path,
            sample_cache,
            start=float(row.get("start") or 0.0),
            end=float(row.get("end") or 0.0),
        )
        text = self.resolve_text(row.get("text_path", ""), row.get("source_dataset", ""))
        return ResolvedMedia(
            ea_id=ea_id,
            video_path=video_path,
            audio_path=audio_path,
            text=text,
            language=(row.get("language") or "unknown").strip(),
            start=float(row.get("start") or 0.0),
            end=float(row.get("end") or 0.0),
            cache_dir=sample_cache,
        )

    def resolve_video(self, video_spec: str, sample_cache: Path) -> Path | None:
        if not video_spec:
            return None
        local = self.materialize_path(video_spec, sample_cache, preferred_name="video.mp4")
        return local

    def resolve_audio(
        self,
        audio_spec: str,
        video_path: Path | None,
        sample_cache: Path,
        start: float = 0.0,
        end: float = 0.0,
    ) -> Path | None:
        out_wav = sample_cache / "audio.wav"
        if out_wav.is_file() and out_wav.stat().st_size > 0:
            return out_wav

        # Prefer extracting wav from video when audio_spec points at the same video/zip member.
        source_for_ffmpeg: Path | None = None
        if audio_spec:
            materialized = self.materialize_path(audio_spec, sample_cache, preferred_name="audio_src")
            if materialized is not None:
                if materialized.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}:
                    shutil.copy2(materialized, out_wav)
                    return out_wav
                source_for_ffmpeg = materialized
        if source_for_ffmpeg is None and video_path is not None:
            source_for_ffmpeg = video_path
        if source_for_ffmpeg is None:
            return None

        extract_wav_with_ffmpeg(source_for_ffmpeg, out_wav, start=start, end=end)
        return out_wav if out_wav.is_file() else None

    def materialize_path(
        self,
        path_spec: str,
        sample_cache: Path,
        preferred_name: str | None = None,
    ) -> Path | None:
        path_spec = path_spec.strip()
        if not path_spec:
            return None

        match = ZIP_MEMBER_RE.match(path_spec)
        if match:
            zip_path = Path(match.group("zip"))
            member = match.group("member").replace("\\", "/")
            # Prefer already-extracted sibling directory, e.g. Raw.zip + Raw/
            extracted_candidate = zip_path.parent / member
            if extracted_candidate.is_file():
                return extracted_candidate
            if not zip_path.is_file():
                raise FileNotFoundError(f"zip not found: {zip_path}")
            suffix = Path(member).suffix or ".bin"
            digest = hashlib.sha1(f"{zip_path}::{member}".encode("utf-8")).hexdigest()[:12]
            out_name = preferred_name or f"{digest}{suffix}"
            if not out_name.endswith(suffix):
                out_name = f"{out_name}{suffix}"
            out_path = sample_cache / out_name
            if out_path.is_file() and out_path.stat().st_size > 0:
                return out_path
            sample_cache.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                with zf.open(member) as src, out_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            return out_path

        path = Path(path_spec)
        if not path.is_file():
            raise FileNotFoundError(f"media file not found: {path}")
        return path

    def resolve_text(self, text_spec: str, source_dataset: str = "") -> str:
        text_spec = (text_spec or "").strip()
        if not text_spec:
            return ""

        meld = MELD_POINTER_RE.match(text_spec)
        if meld:
            return read_meld_utterance(
                Path(meld.group("csv")),
                dialogue_id=int(meld.group("dialog")),
                utterance_id=int(meld.group("utt")),
            )

        mustard = MUSTARD_POINTER_RE.match(text_spec)
        if mustard:
            return read_mustard_utterance(
                Path(mustard.group("json")),
                utterance_id=mustard.group("uid").strip(),
            )

        if "#" in text_spec:
            csv_path_str, pointer = text_spec.split("#", 1)
            csv_path = Path(csv_path_str)
            pointer = pointer.strip()
            if csv_path.suffix.lower() == ".json":
                qs = parse_qs(pointer)
                if "utterance_id" in qs:
                    return read_mustard_utterance(csv_path, qs["utterance_id"][0])
            if csv_path.suffix.lower() == ".csv" and csv_path.is_file():
                # CH-SIMS style: label.csv#video_0001/0001
                if re.fullmatch(r"video_\d+/\d+", pointer):
                    return read_ch_sims_text(csv_path, pointer)
                # Generic query style already handled by MELD regex; try key lookup.
                qs = parse_qs(pointer)
                if "Dialogue_ID" in qs and "Utterance_ID" in qs:
                    return read_meld_utterance(
                        csv_path,
                        dialogue_id=int(qs["Dialogue_ID"][0]),
                        utterance_id=int(qs["Utterance_ID"][0]),
                    )
                return read_ch_sims_text(csv_path, pointer)

        path = Path(text_spec)
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
        return text_spec


def extract_wav_with_ffmpeg(
    source: Path,
    out_wav: Path,
    start: float = 0.0,
    end: float = 0.0,
) -> None:
    ffmpeg_bin = "ffmpeg"
    try:
        import imageio_ffmpeg

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        pass

    cmd = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error"]
    if start > 0:
        cmd.extend(["-ss", f"{start:.3f}"])
    cmd.extend(["-i", str(source)])
    if end > start:
        cmd.extend(["-t", f"{max(end - start, 0.05):.3f}"])
    cmd.extend(["-ac", "1", "-ar", "16000", str(out_wav)])
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg not found. Install system ffmpeg or `pip install imageio-ffmpeg`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed for {source}") from exc


def _open_csv_text(path: Path):
    """Yield an open text handle, trying common encodings."""
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            handle = path.open(newline="", encoding=encoding)
            handle.read(1024)
            handle.seek(0)
            return handle
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            try:
                handle.close()
            except Exception:  # noqa: BLE001
                pass
    raise RuntimeError(f"unable to decode {path}: {last_error}")


def read_ch_sims_text(label_csv: Path, key: str) -> str:
    """Read CH-SIMS utterance text for key like video_0001/0001.

    Official CH-SIMS label.csv is often headerless:
    video_id,clip_id,text,label_value,...,sentiment,split
    """
    key = key.strip().replace("\\", "/")
    with _open_csv_text(label_csv) as handle:
        sample = handle.read(4096)
        handle.seek(0)
        first_line = sample.splitlines()[0] if sample else ""
        has_header = any(
            token in first_line.lower()
            for token in ("text", "video", "id", "sentiment", "clip")
        ) and not first_line.lower().startswith("video_")

        if has_header:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            for row in reader:
                video_id = _first_nonempty(row, "video_id", "Video_ID", "video", "Video")
                clip_id = _first_nonempty(row, "clip_id", "Clip_ID", "clip", "Clip")
                candidates = [
                    row.get("id"),
                    row.get("ID"),
                ]
                if video_id and clip_id:
                    candidates.append(f"{video_id}/{clip_id}")
                candidates = [_normalize_key(candidate) for candidate in candidates]
                if any(_keys_match(candidate, key) for candidate in candidates if candidate):
                    for text_col in ("text", "Text", "chinese", "transcription", "sentence"):
                        if row.get(text_col):
                            return str(row[text_col]).strip()
                    for col in fieldnames:
                        if col.lower() in {"text", "chinese", "transcription", "sentence"}:
                            return str(row.get(col) or "").strip()
        else:
            reader = csv.reader(handle)
            for row in reader:
                if len(row) < 3:
                    continue
                row_key = f"{row[0].strip()}/{row[1].strip()}"
                if row_key == key:
                    return row[2].strip()

    raise KeyError(f"CH-SIMS text not found for key={key!r} in {label_csv}")


def _first_nonempty(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value:
            return str(value).strip()
    return ""


def _normalize_key(value: object) -> str:
    return str(value or "").strip().replace("\\", "/")


def _keys_match(candidate: str, key: str) -> bool:
    return candidate == key or candidate.endswith(f"/{key}")


def read_meld_utterance(csv_path: Path, dialogue_id: int, utterance_id: int) -> str:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                d = int(float(row.get("Dialogue_ID", row.get("dialogue_id", -1))))
                u = int(float(row.get("Utterance_ID", row.get("utterance_id", -1))))
            except (TypeError, ValueError):
                continue
            if d == dialogue_id and u == utterance_id:
                text = row.get("Utterance") or row.get("utterance") or row.get("text") or ""
                return str(text).strip()
    raise KeyError(
        f"MELD utterance not found Dialogue_ID={dialogue_id} Utterance_ID={utterance_id} in {csv_path}"
    )


def read_mustard_utterance(json_path: Path, utterance_id: str) -> str:
    """Lookup MUStARD utterance text from sarcasm_data.json#utterance_id=..."""
    if not json_path.is_file():
        raise FileNotFoundError(f"MUStARD json not found: {json_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"MUStARD json must be an object: {json_path}")
    item = payload.get(utterance_id)
    if not isinstance(item, dict):
        raise KeyError(f"MUStARD utterance_id={utterance_id!r} not found in {json_path}")
    text = item.get("utterance") or item.get("text") or ""
    return str(text).strip()


def read_source_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
