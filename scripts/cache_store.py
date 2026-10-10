#!/usr/bin/env python3
"""Small content-addressed proxy/thumbnail cache for the local workflow.

The cache only owns rebuildable artifacts below ``outputs/cache/content``.
Source files and approved deliveries are never copied into it.  Entries are
addressed by source digest, operation parameters, dimensions and software
version, and are evicted oldest-first when the byte budget is exceeded.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any


def sha256_file(path: Path | str, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class ContentCache:
    """File-backed cache with atomic writes and an explicit byte budget."""

    def __init__(self, root: Path | str, *, max_bytes: int = 2 * 1024 * 1024 * 1024,
                 software_version: str = "cosplay-cache-1"):
        project_root = Path(root).resolve()
        self.root = project_root / "outputs" / "cache" / "content"
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "cache.json"
        self.max_bytes = max(0, int(max_bytes))
        self.software_version = str(software_version)

    @staticmethod
    def _canonical(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

    def source_hash(self, source: Path | str) -> str:
        value = Path(source)
        if value.is_file():
            return sha256_file(value)
        text = str(source)
        if len(text) == 64 and all(char in "0123456789abcdefABCDEF" for char in text):
            return text.lower()
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def make_key(self, source: Path | str, operation: str, *, params: dict[str, Any] | None = None,
                 dimensions: Any = None) -> str:
        payload = {
            "source_sha256": self.source_hash(source), "operation": str(operation),
            "params": params or {}, "dimensions": dimensions,
            "software_version": self.software_version,
        }
        return hashlib.sha256(self._canonical(payload).encode("utf-8")).hexdigest()

    make_cache_key = make_key

    def entry_dir(self, key: str) -> Path:
        key = str(key).strip().lower()
        if len(key) < 16 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError("invalid cache key")
        return self.root / key[:2] / key

    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.is_file():
            return {"schema_version": "1.0", "entries": {}}
        try:
            value = json.loads(self.index_path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and isinstance(value.get("entries", {}), dict):
                return value
        except (OSError, json.JSONDecodeError, TypeError):
            pass
        return {"schema_version": "1.0", "entries": {}}

    def _save_index(self, value: dict[str, Any]) -> None:
        temp = self.index_path.with_name(f".cache-{uuid.uuid4().hex}.tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, self.index_path)

    def _record(self, key: str, *, source_sha256: str | None = None,
                operation: str | None = None, params: dict[str, Any] | None = None,
                dimensions: Any = None, artifact: str, path: Path) -> None:
        index = self._load_index()
        entries = index.setdefault("entries", {})
        item = entries.setdefault(key, {"key": key, "artifacts": {}, "bytes": 0})
        item.update({"last_access": time.time(), "updated_at": time.time()})
        if source_sha256 is not None:
            item["source_sha256"] = source_sha256
        if operation is not None:
            item["operation"] = operation
        if params is not None:
            item["params"] = params
        if dimensions is not None:
            item["dimensions"] = dimensions
        item.setdefault("software_version", self.software_version)
        old = item["artifacts"].get(artifact)
        old_size = int(old.get("bytes", 0)) if isinstance(old, dict) else 0
        size = path.stat().st_size
        item["artifacts"][artifact] = {"path": path.name, "bytes": size, "updated_at": time.time()}
        item["bytes"] = max(0, int(item.get("bytes", 0)) - old_size + size)
        self._save_index(index)

    def put_bytes(self, key: str, artifact: str, data: bytes, *, metadata: dict[str, Any] | None = None) -> Path:
        if not artifact or "/" in artifact or "\\" in artifact or artifact in {"source", "delivery", "final"}:
            raise ValueError("invalid or protected cache artifact")
        entry = self.entry_dir(key)
        entry.mkdir(parents=True, exist_ok=True)
        target = entry / artifact
        temp = entry / f".{artifact}.{uuid.uuid4().hex}.tmp"
        temp.write_bytes(data)
        os.replace(temp, target)
        record = metadata or {}
        self._record(key, source_sha256=record.get("source_sha256"), operation=record.get("operation"),
                     params=record.get("params"), dimensions=record.get("dimensions"), artifact=artifact, path=target)
        entry_meta = entry / "cache.json"
        entry_meta.write_text(json.dumps({"key": key, "artifact": artifact, **record}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.evict()
        return target

    def put_image(self, key: str, image: Any, *, artifact: str = "proxy", format: str | None = None,
                  metadata: dict[str, Any] | None = None) -> Path:
        from io import BytesIO
        output_format = (format or ("JPEG" if artifact in {"proxy", "thumbnail"} else "PNG")).upper()
        buffer = BytesIO()
        save_kwargs = {"quality": 92} if output_format == "JPEG" else {}
        image_to_save = image.convert("RGB") if output_format == "JPEG" and getattr(image, "mode", "RGB") not in {"RGB", "L"} else image
        image_to_save.save(buffer, format=output_format, **save_kwargs)
        suffix = ".jpg" if output_format == "JPEG" else ".png"
        name = artifact if artifact.endswith((".jpg", ".jpeg", ".png", ".webp")) else artifact + suffix
        return self.put_bytes(key, name, buffer.getvalue(), metadata=metadata)

    def get(self, key: str, artifact: str = "proxy") -> Path | None:
        entry = self.entry_dir(key)
        index = self._load_index()
        item = (index.get("entries") or {}).get(key)
        if not isinstance(item, dict):
            return None
        info = (item.get("artifacts") or {}).get(artifact)
        path = entry / str(info.get("path")) if isinstance(info, dict) else entry / artifact
        if not path.is_file():
            return None
        item["last_access"] = time.time()
        self._save_index(index)
        return path

    get_path = get

    def find_latest_proxy(self, source: Path | str, *, params: dict[str, Any] | None = None) -> Path | None:
        """Return the newest proxy matching the requested decoding/display params."""
        source_sha256 = self.source_hash(source)
        index = self._load_index()
        candidates: list[tuple[float, Path]] = []
        for key, item in (index.get("entries") or {}).items():
            if not isinstance(item, dict) or item.get("source_sha256") != source_sha256:
                continue
            if item.get("operation") != "proxy":
                continue
            if params and any((item.get("params") or {}).get(name) != value for name, value in params.items()):
                continue
            artifacts = item.get("artifacts") or {}
            for info in artifacts.values():
                if not isinstance(info, dict):
                    continue
                path = self.entry_dir(key) / str(info.get("path", ""))
                if path.is_file():
                    candidates.append((float(item.get("last_access", 0)), path))
        if not candidates:
            return None
        candidates.sort(key=lambda value: value[0], reverse=True)
        return candidates[0][1]

    def put_proxy(self, source: Path | str, image: Any, *, params: dict[str, Any] | None = None,
                  dimensions: Any = None, artifact: str = "proxy") -> dict[str, Any]:
        key = self.make_key(source, "proxy", params=params, dimensions=dimensions)
        source_hash = self.source_hash(source)
        path = self.put_image(key, image, artifact=artifact,
                              metadata={"source_sha256": source_hash, "operation": "proxy",
                                        "params": params or {}, "dimensions": dimensions})
        return {"key": key, "path": str(path), "cache_hit": False, "source_sha256": source_hash}

    def get_proxy(self, source: Path | str, *, params: dict[str, Any] | None = None,
                  dimensions: Any = None, artifact: str = "proxy") -> dict[str, Any] | None:
        key = self.make_key(source, "proxy", params=params, dimensions=dimensions)
        lookup = artifact if artifact.endswith((".jpg", ".png", ".webp")) else artifact + ".jpg"
        path = self.get(key, lookup)
        return {"key": key, "path": str(path), "cache_hit": True} if path else None

    def total_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.root.glob("**/*") if path.is_file() and path.name != "cache.json")

    def evict(self) -> list[str]:
        index = self._load_index()
        removed: list[str] = []
        while self.total_bytes() > self.max_bytes:
            entries = index.get("entries") or {}
            candidates = [(float(item.get("last_access", 0)), key) for key, item in entries.items() if isinstance(item, dict)]
            if not candidates:
                break
            _, key = min(candidates)
            shutil.rmtree(self.entry_dir(key), ignore_errors=True)
            entries.pop(key, None)
            removed.append(key)
            self._save_index(index)
        return removed


CacheStore = ContentCache
