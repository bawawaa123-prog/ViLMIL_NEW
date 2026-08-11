#!/usr/bin/env python3
"""Resumable SMB upload of the contents of eval_results/yiyuan to GDPH."""

from __future__ import annotations

import argparse
import hashlib
import os
import runpy
import uuid
from dataclasses import dataclass
from pathlib import Path

import smbclient
from tqdm import tqdm


DEFAULT_SOURCE = Path(
    "/private/ljh-data/shared/ViLMIL/ViLa-MIL-main/eval_results/yiyuan"
)
DEFAULT_TARGET = r"\\172.22.194.201\metaverse\ljh\GDPH"
DEFAULT_CONFIG = Path(
    "/private/ljh-data/shared/data/tools/jiangxi/upload_shared_projects_to_nas.py"
)


@dataclass(frozen=True)
class UploadTask:
    local: Path
    remote: str
    size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload the contents of a local directory directly into an SMB directory. "
            "Interrupted files resume from their existing remote size."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--config-script", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--chunk-size-mb", type=int, default=8)
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Test write, resume, read-back verification, and cleanup; do not upload source files.",
    )
    return parser.parse_args()


def remote_join(root: str, relative: str) -> str:
    return root.rstrip("\\") + "\\" + relative.strip("\\")


def remote_size(path: str) -> int | None:
    if not smbclient.path.exists(path):
        return None
    if smbclient.path.isdir(path):
        raise IsADirectoryError(path)
    return smbclient.stat(path).st_size


def register_session(config_script: Path) -> None:
    if not config_script.is_file():
        raise FileNotFoundError(f"Credential config script does not exist: {config_script}")
    config = runpy.run_path(str(config_script))
    smbclient.register_session(
        config["DEFAULT_NAS_IP"],
        username=config["DEFAULT_NAS_USER"],
        password=config["DEFAULT_NAS_PASSWORD"],
    )


def test_resume_round_trip(target: str) -> None:
    smbclient.makedirs(target, exist_ok=True)
    test_path = remote_join(target, f".vilmil_resume_test_{uuid.uuid4().hex}.bin")
    payload = (b"ViLMIL SMB resumable transfer test\n" * 4096) + os.urandom(257)
    split = len(payload) // 2

    try:
        with smbclient.open_file(test_path, mode="wb") as remote:
            remote.write(payload[:split])
        if remote_size(test_path) != split:
            raise IOError("The partial test upload has an unexpected size")

        with smbclient.open_file(test_path, mode="ab") as remote:
            remote.write(payload[split:])
        if remote_size(test_path) != len(payload):
            raise IOError("The resumed test upload has an unexpected size")

        with smbclient.open_file(test_path, mode="rb") as remote:
            downloaded = remote.read()
        if hashlib.sha256(downloaded).digest() != hashlib.sha256(payload).digest():
            raise IOError("The read-back SHA-256 check failed")
    finally:
        if smbclient.path.exists(test_path):
            smbclient.remove(test_path)

    print("SMB transfer test passed: write, resume, SHA-256 read-back, and cleanup all succeeded.")


def scan_local(source: Path, target: str) -> tuple[list[UploadTask], list[str], int]:
    tasks: list[UploadTask] = []
    remote_dirs: list[str] = []
    total_bytes = 0

    for dirpath, _, filenames in os.walk(source):
        local_dir = Path(dirpath)
        relative_dir = local_dir.relative_to(source)
        if relative_dir == Path("."):
            remote_dir = target.rstrip("\\")
        else:
            remote_dir = remote_join(
                target, relative_dir.as_posix().replace("/", "\\")
            )
        remote_dirs.append(remote_dir)

        for filename in filenames:
            local = local_dir / filename
            size = local.stat().st_size
            tasks.append(UploadTask(local, remote_join(remote_dir, filename), size))
            total_bytes += size

    return tasks, remote_dirs, total_bytes


def upload(source: Path, target: str, chunk_size: int) -> None:
    if not source.is_dir():
        raise NotADirectoryError(f"Source directory does not exist: {source}")

    tasks, remote_dirs, total_bytes = scan_local(source, target)
    print(f"Source: {source}")
    print(f"Target: {target}")
    print(f"Files: {len(tasks)}; directories: {len(remote_dirs)}; bytes: {total_bytes}")

    # Create each directory once. This also preserves empty source directories.
    for remote_dir in tqdm(remote_dirs, desc="Create directories", unit="dir"):
        smbclient.makedirs(remote_dir, exist_ok=True)

    uploaded = resumed = skipped = failed = 0
    with tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Upload",
    ) as progress:
        for task in tasks:
            try:
                existing = remote_size(task.remote)
                if existing is not None and existing > task.size:
                    smbclient.remove(task.remote)
                    existing = None

                if existing == task.size:
                    progress.update(task.size)
                    skipped += 1
                    continue

                offset = existing or 0
                progress.update(offset)
                with task.local.open("rb") as local:
                    local.seek(offset)
                    with smbclient.open_file(
                        task.remote, mode="ab" if offset else "wb"
                    ) as remote:
                        while chunk := local.read(chunk_size):
                            remote.write(chunk)
                            progress.update(len(chunk))

                final_size = remote_size(task.remote)
                if final_size != task.size:
                    raise IOError(
                        f"size mismatch after upload: local={task.size}, remote={final_size}"
                    )
                if offset:
                    resumed += 1
                else:
                    uploaded += 1
            except Exception as exc:
                failed += 1
                print(f"\nFailed: {task.local} -> {task.remote}: {exc}")

    print(
        f"Finished: uploaded={uploaded}, resumed={resumed}, "
        f"skipped={skipped}, failed={failed}"
    )
    if failed:
        raise SystemExit(2)


def main() -> None:
    args = parse_args()
    if args.chunk_size_mb <= 0:
        raise ValueError("--chunk-size-mb must be positive")

    register_session(args.config_script)
    if args.test_only:
        test_resume_round_trip(args.target)
        return

    upload(args.source.resolve(), args.target, args.chunk_size_mb * 1024 * 1024)


if __name__ == "__main__":
    main()
