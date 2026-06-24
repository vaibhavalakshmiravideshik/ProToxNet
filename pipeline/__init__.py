"""Shared helpers for ProToxNet pipeline modules."""

from __future__ import annotations

import os
from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_data_dir() -> Path:
    override = os.environ.get("PROTOXNET_DRIVE")
    data_dir = Path(override).expanduser() if override else get_repo_root() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


__all__ = ["get_repo_root", "get_data_dir"]
