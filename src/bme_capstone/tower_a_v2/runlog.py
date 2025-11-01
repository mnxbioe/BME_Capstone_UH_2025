"""Run folder + configuration snapshot utilities for Tower A v2."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _git_short_sha(root: Path) -> str:
    try:
        sha = (
            subprocess.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        return sha
    except Exception:
        return "nogit"


def start_run(slug: str, config: Dict[str, Any], *, root: Optional[Path] = None) -> Path:
    root_path = root or Path.cwd()
    runs_dir = root_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    sha = _git_short_sha(root_path)
    safe_slug = "".join(ch for ch in slug if ch.isalnum() or ch in ("-", "_"))[:32]
    run_dir = runs_dir / f"{timestamp}_{safe_slug}_{sha}"
    run_dir.mkdir(parents=True, exist_ok=False)

    config_dir = run_dir / "config"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)

    return run_dir


__all__ = ["start_run"]

