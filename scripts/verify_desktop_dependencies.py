"""Verify the pinned desktop build environment and preserved license bytes."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path, PurePosixPath
import platform
import re
import struct
import sys


class VerificationError(ValueError):
    """A public, path-free build-gate failure."""


def canonical_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def read_pins(path):
    pins = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s;=]+)", line)
        if not match:
            raise VerificationError("Desktop requirements must contain exact distribution==version pins")
        name, version = match.groups()
        key = canonical_name(name)
        if key in pins:
            raise VerificationError("Duplicate desktop distribution pin: " + name)
        pins[key] = {"name": name, "version": version}
    if not pins:
        raise VerificationError("Desktop distribution pins are empty")
    return pins


def verify_python(version=None, bits=None, machine=None):
    version = tuple(sys.version_info[:2]) if version is None else tuple(version)
    bits = struct.calcsize("P") * 8 if bits is None else bits
    machine = platform.machine() if machine is None else machine
    if version != (3, 12) or bits != 64 or machine.lower() not in {"amd64", "x86_64"}:
        raise VerificationError("The desktop build requires Python 3.12 x64")


def verify_versions(pins, version_for=metadata.version):
    verified = {}
    for key, row in sorted(pins.items()):
        try:
            actual = version_for(row["name"])
        except metadata.PackageNotFoundError:
            raise VerificationError("Missing pinned distribution: " + row["name"]) from None
        if actual != row["version"]:
            raise VerificationError("Pinned version mismatch: " + row["name"])
        verified[key] = actual
    return verified


def checked_relative(value, prefix=None):
    if not isinstance(value, str) or "\\" in value:
        raise VerificationError("Manifest path must use a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"..", "."} or ":" in part for part in path.parts):
        raise VerificationError("Manifest path must stay inside its source tree")
    if prefix is not None and path.parts[0] != prefix:
        raise VerificationError("License record must stay inside LICENSES")
    return path


def verify_license_inventory(root):
    directory = root / "LICENSES"
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
        records = manifest["texts"]
    except (OSError, ValueError, KeyError, TypeError):
        raise VerificationError("License manifest is missing or invalid") from None
    if not isinstance(records, list) or not records:
        raise VerificationError("License manifest has no text records")
    names = set()
    for row in records:
        try:
            relative = checked_relative(row["target"], "LICENSES")
            expected = row["sha256"]
            size = row["bytes"]
        except (KeyError, TypeError):
            raise VerificationError("License text record is incomplete") from None
        name = relative.as_posix()
        if name in names:
            raise VerificationError("Duplicate license text record")
        names.add(name)
        path = root.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(directory.resolve()):
            raise VerificationError("License path escapes LICENSES")
        if not re.fullmatch(r"[a-f0-9]{64}", str(expected)) or not isinstance(size, int) or isinstance(size, bool):
            raise VerificationError("License text size or SHA-256 is invalid")
        if not path.is_file():
            raise VerificationError("Preserved license text is missing: " + name)
        data = path.read_bytes()
        if len(data) != size or hashlib.sha256(data).hexdigest() != expected:
            raise VerificationError("Preserved license text changed: " + name)
    support = {"LICENSES/README.md", "LICENSES/manifest.json", "LICENSES/dependency-sources.json"}
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix.lower() in {".py", ".pyc", ".pyo", ".pyd", ".dll", ".exe"}:
            raise VerificationError("Code or cache files cannot be preserved as license texts")
        if relative.as_posix() not in names | support:
            raise VerificationError("Unrecorded license text: " + relative.as_posix())
    return records


def verify_installed_license_texts(records, pins, distribution_for=metadata.distribution):
    checked = 0
    marker = "installed wheel RECORD: "
    for row in records:
        origin = row.get("origin", "")
        if not origin.startswith(marker):
            continue
        component = row["component"]
        pin = pins.get(canonical_name(component))
        if pin is None or row.get("version") != pin["version"]:
            raise VerificationError("License inventory differs from the desktop pin: " + component)
        relative = checked_relative(origin[len(marker):])
        try:
            distribution = distribution_for(component)
            path = Path(distribution.locate_file(relative.as_posix()))
            data = path.read_bytes()
        except (metadata.PackageNotFoundError, OSError):
            raise VerificationError("Installed wheel license is missing: " + component) from None
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise VerificationError("Installed wheel license differs from the preserved text: " + component)
        checked += 1
    if not checked:
        raise VerificationError("License inventory has no installed-wheel text bindings")
    return checked


def verify(root, *, import_modules=True):
    verify_python()
    pins = read_pins(root / "scripts/requirements-desktop.txt")
    versions = verify_versions(pins)
    records = verify_license_inventory(root)
    wheel_texts = verify_installed_license_texts(records, pins)
    if import_modules:
        for name in ("PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets", "numpy", "scipy", "PIL", "tifffile", "rawpy", "onnxruntime", "PyInstaller"):
            try:
                importlib.import_module(name)
            except Exception:
                raise VerificationError("Pinned native dependency cannot load: " + name) from None
    return {"ok": True, "python": "3.12", "architecture": "x64", "pinned_distributions": versions,
            "license_texts": len(records), "installed_wheel_license_texts": wheel_texts,
            "native_imports_checked": import_modules}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    try:
        result = verify(args.root.resolve())
    except VerificationError as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2
    except Exception:
        print(json.dumps({"ok": False, "error": "Desktop dependency verification could not complete"}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
