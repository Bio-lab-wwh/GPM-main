"""Resolve project inputs and serialize portable artifact references."""

import hashlib
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    return resolved.as_posix()


def resolve_project_path(value: str | Path | None) -> Path | None:
    if not value:
        return None
    return PROJECT_ROOT / Path(value)


def manifest_artifact_path(value: str | Path, manifest_path: Path) -> Path:
    return manifest_path.parent / Path(value)


def file_identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "sha256": _file_digest(path.resolve(), stat.st_size, stat.st_mtime_ns),
    }


@lru_cache(maxsize=16)
def _file_digest(path: Path, size: int, mtime_ns: int) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
