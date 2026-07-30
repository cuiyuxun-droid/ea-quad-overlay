#!/usr/bin/env python
"""Pull M1 seed media/annotations from AutoDL and write a local source index.

Usage:
  $env:EA_SSH_PASSWORD = '<password>'
  python scripts/pull_m1_data_local.py
"""

from __future__ import annotations

import csv
import os
import posixpath
import re
import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "source_index" / "m1_sample_20.csv"
LOCAL_DATA = ROOT / "data" / "m1"
LOCAL_INDEX = ROOT / "source_index" / "m1_sample_20_local.csv"
ZIP_RE = re.compile(r"^(?P<zip>.+\.zip)::(?P<member>.+)$", re.IGNORECASE)


def connect() -> paramiko.SSHClient:
    password = os.environ.get("EA_SSH_PASSWORD")
    if not password:
        raise SystemExit("Set EA_SSH_PASSWORD")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=os.environ.get("EA_SSH_HOST", "connect.westd.seetacloud.com"),
        port=int(os.environ.get("EA_SSH_PORT", "11482")),
        username=os.environ.get("EA_SSH_USER", "root"),
        password=password,
        timeout=30,
    )
    return client


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sftp_get(sftp: paramiko.SFTPClient, remote: str, local: Path) -> None:
    ensure_parent(local)
    if local.is_file() and local.stat().st_size > 0:
        print(f"skip existing {local}")
        return
    print(f"download {remote} -> {local}")
    sftp.get(remote, str(local))


def remote_exists(sftp: paramiko.SFTPClient, path: str) -> bool:
    try:
        sftp.stat(path)
        return True
    except OSError:
        return False


def materialize_video(sftp: paramiko.SFTPClient, video_spec: str, local_video: Path) -> None:
    if local_video.is_file() and local_video.stat().st_size > 0:
        print(f"skip existing {local_video}")
        return

    match = ZIP_RE.match(video_spec)
    if match:
        zip_path = match.group("zip")
        member = match.group("member").replace("\\", "/")
        # Prefer already-extracted sibling Raw/ on server
        extracted = posixpath.join(posixpath.dirname(zip_path), member)
        if remote_exists(sftp, extracted):
            sftp_get(sftp, extracted, local_video)
            return
        # Fallback: ask remote to extract one member via python
        print(f"extract remote zip member {zip_path}::{member}")
        cmd = (
            "/root/miniconda3/bin/python - <<'PY'\n"
            "import zipfile, pathlib\n"
            f"zip_path=pathlib.Path({zip_path!r})\n"
            f"member={member!r}\n"
            "out=pathlib.Path('/tmp')/pathlib.Path(member).name\n"
            "with zipfile.ZipFile(zip_path) as zf, zf.open(member) as src, out.open('wb') as dst:\n"
            "    dst.write(src.read())\n"
            "print(out)\n"
            "PY"
        )
        # Use transport from sftp
        raise RuntimeError(
            f"Extracted path missing on server: {extracted}. "
            "Ensure CH-SIMS Raw/ is extracted on AutoDL."
        )

    if not remote_exists(sftp, video_spec):
        raise FileNotFoundError(f"remote video missing: {video_spec}")
    sftp_get(sftp, video_spec, local_video)


def main() -> None:
    rows = list(csv.DictReader(INDEX.open(encoding="utf-8")))
    LOCAL_DATA.mkdir(parents=True, exist_ok=True)

    client = connect()
    try:
        sftp = client.open_sftp()
        try:
            # Shared annotation files
            ch_label_remote = "/root/autodl-tmp/data/datasets/ch_sims/label.csv"
            ch_label_local = LOCAL_DATA / "ch_sims" / "label.csv"
            sftp_get(sftp, ch_label_remote, ch_label_local)

            meld_ann_dir = "/root/autodl-tmp/data/datasets/meld/annotations"
            for name in ("train_sent_emo.csv", "dev_sent_emo.csv", "test_sent_emo.csv"):
                remote = f"{meld_ann_dir}/{name}"
                if remote_exists(sftp, remote):
                    sftp_get(sftp, remote, LOCAL_DATA / "meld" / "annotations" / name)

            out_rows: list[dict[str, str]] = []
            for row in rows:
                ea_id = row["ea_id"]
                dataset = row["source_dataset"]
                local_video = LOCAL_DATA / dataset.replace("-", "_").lower() / "videos" / f"{ea_id}.mp4"
                materialize_video(sftp, row["video_path"], local_video)

                # Remap text pointer
                text_spec = row["text_path"]
                if dataset == "CH-SIMS" and "#" in text_spec:
                    _, pointer = text_spec.split("#", 1)
                    local_text = f"{ch_label_local.as_posix()}#{pointer}"
                elif dataset == "MELD" and "#" in text_spec:
                    remote_csv, pointer = text_spec.split("#", 1)
                    local_csv = LOCAL_DATA / "meld" / "annotations" / Path(remote_csv).name
                    local_text = f"{local_csv.as_posix()}#{pointer}"
                else:
                    local_text = text_spec

                new_row = dict(row)
                video_path = str(local_video.resolve())
                new_row["video_path"] = video_path
                new_row["audio_path"] = video_path  # extract wav from video locally
                new_row["text_path"] = local_text
                out_rows.append(new_row)
                print(f"mapped {ea_id}")
        finally:
            sftp.close()
    finally:
        client.close()

    fieldnames = list(rows[0].keys())
    with LOCAL_INDEX.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {LOCAL_INDEX} ({len(out_rows)} rows)")
    print(f"Media root: {LOCAL_DATA}")


if __name__ == "__main__":
    main()
