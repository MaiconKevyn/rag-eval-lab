from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def sha256_of_file(path: str | Path) -> str:
    """Stream-hash a file. Used to fingerprint the corpus for cache + benchmark versioning."""
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(_HASH_CHUNK_SIZE), b""):
            h.update(block)
    return h.hexdigest()


def sha256_of_files(paths: list[str | Path]) -> str:
    """Hash a set of files. Order-independent: sorted by absolute path before concatenation."""
    h = hashlib.sha256()
    for p in sorted(Path(x).resolve() for x in paths):
        h.update(sha256_of_file(p).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any, *, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent, default=str),
        encoding="utf-8",
    )
    return path
