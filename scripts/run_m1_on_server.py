#!/usr/bin/env python
"""Upload M1 feature code to AutoDL and run extraction in one shot.

Usage (PowerShell):
  $env:EA_SSH_PASSWORD = '<password>'
  # CPU instance (current):
  python scripts/run_m1_on_server.py --device cpu --wait
  # Later, when GPU is available:
  python scripts/run_m1_on_server.py --device cuda --wait

Environment:
  EA_SSH_HOST      default connect.westd.seetacloud.com
  EA_SSH_PORT      default 11482
  EA_SSH_USER      default root
  EA_SSH_PASSWORD  required
  EA_REMOTE_ROOT   default /root/ea-quad-overlay
  EA_REMOTE_PYTHON default /root/miniconda3/bin/python
"""

from __future__ import annotations

import argparse
import os
import posixpath
import stat
import sys
import time
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
UPLOADS = [
    "requirements-features.txt",
    "source_index/m1_sample_20.csv",
    "source_index/source_index_template.csv",
    "scripts/extract_m1_features.py",
    "scripts/validate_m1_features.py",
    "scripts/ea_features/__init__.py",
    "scripts/ea_features/io_utils.py",
    "scripts/ea_features/media.py",
    "scripts/ea_features/face_utils.py",
    "scripts/ea_features/text_extract.py",
    "scripts/ea_features/speech_extract.py",
    "scripts/ea_features/macro_extract.py",
    "scripts/ea_features/micro_extract.py",
    "scripts/ea_features/data/haarcascade_frontalface_default.xml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--install-deps", action="store_true", help="pip install requirements")
    parser.add_argument("--wait", action="store_true", help="poll until extract finishes and pull results")
    parser.add_argument("--poll-seconds", type=int, default=20)
    return parser.parse_args()


def connect() -> paramiko.SSHClient:
    password = os.environ.get("EA_SSH_PASSWORD")
    if not password:
        raise SystemExit("Set EA_SSH_PASSWORD before running.")
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


def run(client: paramiko.SSHClient, command: str, timeout: int = 7200) -> tuple[int, str, str]:
    print(f"$ {command[:200]}", flush=True)
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out.strip():
        print(out[-6000:] if len(out) > 6000 else out, flush=True)
    if err.strip():
        print(err[-3000:] if len(err) > 3000 else err, file=sys.stderr, flush=True)
    return code, out, err


def mkdir_p(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts = []
    for part in remote_dir.strip("/").split("/"):
        parts.append(part)
        path = "/" + "/".join(parts)
        try:
            sftp.stat(path)
        except OSError:
            sftp.mkdir(path)


def upload(client: paramiko.SSHClient, remote_root: str) -> None:
    sftp = client.open_sftp()
    try:
        for rel in UPLOADS:
            remote = posixpath.join(remote_root, rel.replace("\\", "/"))
            mkdir_p(sftp, posixpath.dirname(remote))
            sftp.put(str(ROOT / rel), remote)
            print(f"upload {rel}", flush=True)
    finally:
        sftp.close()


def download_tree(sftp: paramiko.SFTPClient, remote_dir: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        entries = sftp.listdir_attr(remote_dir)
    except OSError:
        return
    for entry in entries:
        remote = posixpath.join(remote_dir, entry.filename)
        local = local_dir / entry.filename
        if stat.S_ISDIR(entry.st_mode):
            download_tree(sftp, remote, local)
        else:
            print(f"download {remote}", flush=True)
            local.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote, str(local))


def start_extract(client: paramiko.SSHClient, remote_root: str, py: str, device: str, skip_existing: bool) -> None:
    skip_flag = "--skip-existing" if skip_existing else ""
    cuda_env = "export CUDA_VISIBLE_DEVICES=\n" if device == "cpu" else ""
    wrapper = (
        "#!/bin/bash\n"
        f"cd {remote_root}\n"
        "export HF_ENDPOINT=https://hf-mirror.com\n"
        "export HF_HUB_DISABLE_XET=1\n"
        f"{cuda_env}"
        f"exec {py} -u scripts/extract_m1_features.py {skip_flag} --device {device} "
        "> /tmp/m1_extract.log 2>&1\n"
    )
    sftp = client.open_sftp()
    try:
        with sftp.file("/tmp/run_m1_extract.sh", "w") as handle:
            handle.write(wrapper)
        sftp.chmod("/tmp/run_m1_extract.sh", 0o755)
    finally:
        sftp.close()

    run(client, "pkill -f 'scripts/extract_m1_features.py' || true", timeout=20)
    time.sleep(1)
    transport = client.get_transport()
    assert transport is not None
    channel = transport.open_session()
    channel.exec_command("setsid /tmp/run_m1_extract.sh </dev/null >/tmp/m1_start.out 2>&1 &")
    time.sleep(2)
    channel.close()
    run(client, "pgrep -af 'extract_m1_features' || true; sleep 1; head -n 30 /tmp/m1_extract.log || true")


def main() -> None:
    args = parse_args()
    remote_root = os.environ.get("EA_REMOTE_ROOT", "/root/ea-quad-overlay")
    py = os.environ.get("EA_REMOTE_PYTHON", "/root/miniconda3/bin/python")
    skip_existing = not args.no_skip_existing

    client = connect()
    try:
        run(
            client,
            f"mkdir -p {remote_root}/features/{{text,speech,macro,micro}} "
            f"{remote_root}/reports {remote_root}/.cache/media "
            f"{remote_root}/scripts/ea_features/data",
            timeout=30,
        )
        upload(client, remote_root)

        if args.install_deps:
            code, _, _ = run(
                client,
                f"cd {remote_root} && {py} -m pip install -r requirements-features.txt",
                timeout=7200,
            )
            if code != 0:
                raise SystemExit(f"pip install failed: {code}")

        start_extract(client, remote_root, py, args.device, skip_existing)

        if not args.wait:
            print("Extract started on server. Re-run with --wait to poll and download.")
            return

        for i in range(240):
            status = run(
                client,
                "ps -eo pid,etime,pcpu,cmd | awk '/extract_m1_features.py/ && !/awk/ {print}'; "
                "echo '---'; "
                "tail -n 25 /tmp/m1_extract.log; "
                f"{py} - <<'PY'\n"
                "from pathlib import Path\n"
                "root=Path('/root/ea-quad-overlay/features')\n"
                "print('COUNTS', {m: len(list((root/m).glob('*.npy'))) if (root/m).exists() else 0 "
                "for m in ['text','speech','macro','micro']})\n"
                "PY",
                timeout=60,
            )
            running = "extract_m1_features.py" in status.split("---", 1)[0]
            print(f"iter={i} running={running}", flush=True)
            if not running and i > 0:
                break
            time.sleep(args.poll_seconds)

        run(client, f"cd {remote_root} && {py} scripts/validate_m1_features.py", timeout=180)

        sftp = client.open_sftp()
        try:
            download_tree(sftp, posixpath.join(remote_root, "features"), ROOT / "features")
            for name in ("m1_feature_check.md", "m1_feature_failures.csv"):
                remote = posixpath.join(remote_root, "reports", name)
                local = ROOT / "reports" / name
                try:
                    sftp.stat(remote)
                except OSError:
                    continue
                print(f"download {remote}", flush=True)
                local.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(remote, str(local))
        finally:
            sftp.close()
    finally:
        client.close()
    print("Done.")


if __name__ == "__main__":
    main()
