"""Resume and verify the public Qwen Image 2.1 model catalog."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time

from start_qwen_image import model_catalog, deployment_root

ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def unlink_transfer(path: Path) -> None:
    # Windows virus scanners can briefly retain a just-closed transfer file.
    for attempt in range(40):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.25)


@contextmanager
def model_lock(target: Path):
    with target.with_name(target.name + ".lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another downloader owns " + str(target)) from exc
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def download(entry: dict, folder: Path, proxy: str | None = None, chunk_root: Path | None = None) -> dict:
    target = (folder / entry["destination"]).resolve()
    if not target.is_relative_to(folder.resolve()):
        raise ValueError("Model destination escapes deployment directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    with model_lock(target):
        return download_locked(entry, target, proxy, chunk_root)


def download_locked(entry: dict, target: Path, proxy: str | None, chunk_root: Path | None) -> dict:
    if target.is_file() and target.stat().st_size == entry["size"] and digest(target) == entry["sha256"]:
        print(f"VERIFIED {entry['role']}", flush=True)
        return {**entry, "path": str(target), "verified": True}
    part = target.with_name(target.name + ".download")
    print(f"DOWNLOADING {entry['role']} ({entry['size'] / 1e9:.2f} GB)", flush=True)
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required for resumable model downloads")
    command = [curl, "--fail", "--location", "--silent", "--show-error", "--retry", "4",
               "--retry-delay", "2", "--connect-timeout", "30", "--speed-time", "120",
               "--speed-limit", "1024", "--continue-at", "-", "--output", str(part)]
    # Schannel's certificate chain verification remains enabled; its offline
    # revocation lookup fails on this host even for publicly trusted hosts.
    if curl.lower().endswith(".exe"):
        command.append("--ssl-no-revoke")
    if proxy:
        command += ["--proxy", proxy]
    if entry.get("parallel_ranges"):
        # The overseas host limits individual connections; bounded parallel
        # byte ranges speed it up while the final hash verifies the whole file.
        chunks_dir = (chunk_root or target.parent) / (target.name + ".chunks")
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunk_size = 64 * 1024 * 1024
        ranges = [(offset, min(offset + chunk_size, entry["size"]) - 1)
                  for offset in range(0, entry["size"], chunk_size)]

        def fetch(bounds: tuple[int, int]) -> Path:
            start, end = bounds
            chunk = chunks_dir / f"{start:012d}.part"
            expected = end - start + 1
            if chunk.exists() and chunk.stat().st_size == expected:
                return chunk
            tail = chunk.with_suffix(".tail")
            for attempt in range(10):
                have = chunk.stat().st_size if chunk.exists() else 0
                if have == expected:
                    break
                if have > expected:
                    chunk.replace(chunk.with_suffix(".invalid"))
                    have = 0
                unlink_transfer(tail)
                headers = chunk.with_suffix(".headers")
                chunk_command = [curl, "--fail", "--location", "--silent", "--show-error",
                                 "--connect-timeout", "30", "--max-time", "180",
                                 "--range", f"{start + have}-{end}",
                                 "--max-filesize", str(expected - have), "--dump-header", str(headers),
                                 "--output", str(tail)]
                if curl.lower().endswith(".exe"):
                    chunk_command.append("--ssl-no-revoke")
                if proxy:
                    chunk_command += ["--proxy", proxy]
                separator = "&" if "?" in entry["url"] else "?"
                result = subprocess.run(chunk_command + [entry["url"] + separator + "download=true"])
                if tail.exists() and 0 < tail.stat().st_size <= expected - have:
                    returned = re.findall(r"content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)",
                                          headers.read_text(encoding="utf-8", errors="replace"), re.I)
                    if not returned or int(returned[-1][0]) != start + have or int(returned[-1][2]) != entry["size"]:
                        raise RuntimeError(f"Server did not return the requested byte range {start + have}-{end}")
                    with chunk.open("ab") as sink, tail.open("rb") as stream:
                        shutil.copyfileobj(stream, sink, 16 * 1024 * 1024)
                    unlink_transfer(tail)
                if result.returncode and not chunk.exists():
                    time.sleep(min(2 * (attempt + 1), 10))
            if not chunk.exists() or chunk.stat().st_size != expected:
                raise RuntimeError(f"Incomplete model range {start}-{end}")
            print(f"RANGE COMPLETE {entry['role']} {start // chunk_size + 1}/{len(ranges)}", flush=True)
            return chunk

        with ThreadPoolExecutor(max_workers=max(1, min(int(entry.get("range_workers", 24)), 64))) as ranges_pool:
            chunks = list(ranges_pool.map(fetch, ranges))
        with part.open("wb") as combined:
            for chunk in chunks:
                with chunk.open("rb") as stream:
                    shutil.copyfileobj(stream, combined, 16 * 1024 * 1024)
    else:
        subprocess.run(command + [entry["url"]], check=True)
    if part.stat().st_size != entry["size"] or digest(part) != entry["sha256"]:
        raise RuntimeError(f"Size/SHA256 mismatch for {entry['role']}; retained download for inspection")
    part.replace(target)
    if entry.get("parallel_ranges"):
        # Remove only our numbered transfer files after the complete model
        # passed SHA256; leave unrelated files in the directory untouched.
        for chunk in chunks_dir.iterdir():
            if chunk.is_file() and chunk.stem.isdecimal() and chunk.suffix in {".part", ".tail", ".headers", ".invalid"}:
                unlink_transfer(chunk)
        if not any(chunks_dir.iterdir()):
            chunks_dir.rmdir()
    print(f"VERIFIED {entry['role']}", flush=True)
    return {**entry, "path": str(target), "verified": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--proxy", help="Optional HTTP/SOCKS proxy for this download only")
    parser.add_argument("--roles", nargs="+", help="Download selected catalog roles only")
    parser.add_argument("--chunk-root", type=Path, help="Directory for resumable download ranges; defaults to the deployment disk")
    args = parser.parse_args()
    catalog = model_catalog()
    entries = catalog["models"]
    if args.roles:
        unknown = set(args.roles) - {e["role"] for e in entries}
        if unknown:
            parser.error("Unknown model roles: " + ", ".join(sorted(unknown)))
        entries = [e for e in entries if e["role"] in args.roles]
    folder = args.root.expanduser().resolve() if args.root else deployment_root()
    folder.mkdir(parents=True, exist_ok=True)
    chunk_root = args.chunk_root.expanduser().resolve() if args.chunk_root else None
    if chunk_root:
        chunk_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(folder).free < sum(e["size"] for e in entries if not (folder / e["destination"]).exists()):
        raise RuntimeError("Insufficient free disk space for model weights")
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 4))) as pool:
        verified = list(pool.map(lambda e: download(e, folder, args.proxy, chunk_root), entries))
    receipt_path = folder / "models-verified.json"
    if args.roles and receipt_path.is_file():
        previous = json.loads(receipt_path.read_text(encoding="utf-8"))
        paths = {e["path"] for e in verified}
        by_path = {str(folder / e["destination"]): e for e in catalog["models"]}
        verified += [{**e, "role": by_path[e["path"]]["role"]} for e in previous.get("models", [])
                     if e["path"] not in paths and e["path"] in by_path]
    receipt = {"schema_version": "1.0", "models": verified, "elapsed_seconds": round(time.time() - started, 2)}
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "verified_models": len(verified), "deployment_root": str(folder)}), flush=True)


if __name__ == "__main__":
    main()
