from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(path.resolve())
    config["_config_sha256"] = sha256(path)
    return config


def sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(config: dict[str, Any], key: str) -> Path:
    config_dir = Path(config["_config_path"]).parent
    root = config_dir.parent
    return (root / config["paths"][key]).resolve()
