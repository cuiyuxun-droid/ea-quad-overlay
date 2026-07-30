#!/usr/bin/env python
"""Upload local M1 features/reports back to AutoDL.

Usage:
  $env:EA_SSH_PASSWORD = '<password>'
  python scripts/push_m1_features_to_server.py
"""

from __future__ import annotations

import os
import posixpath
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
REMOTE = os.environ.get("EA_REMOTE_ROOT", "/root/ea-quad-overlay")


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


def mkdir_p(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts = []
    for part in remote_dir.strip("/").split("/"):
        parts.append(part)
        path = "/" + "/".join(parts)
        try:
            sftp.stat(path)
        except OSError:
            sftp.mkdir(path)


def upload_tree(sftp: paramiko.SFTPClient, local_dir: Path, remote_dir: str) -> int:
    count = 0
    if not local_dir.exists():
        return 0
    mkdir_p(sftp, remote_dir)
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        remote = posixpath.join(remote_dir, rel)
        mkdir_p(sftp, posixpath.dirname(remote))
        print(f"upload {path} -> {remote}")
        sftp.put(str(path), remote)
        count += 1
    return count


def main() -> None:
    client = connect()
    try:
        sftp = client.open_sftp()
        try:
            n_feat = upload_tree(sftp, ROOT / "features", f"{REMOTE}/features")
            n_rep = upload_tree(sftp, ROOT / "reports", f"{REMOTE}/reports")
        finally:
            sftp.close()
    finally:
        client.close()
    print(f"Uploaded feature files={n_feat}, report files={n_rep}")


if __name__ == "__main__":
    main()
