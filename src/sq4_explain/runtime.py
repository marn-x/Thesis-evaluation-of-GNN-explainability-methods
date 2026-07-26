"""Cross-cutting runtime concerns: logging, seeding, device choice, manifests.

The manifest written by :func:`write_manifest` is the audit trail for a run.
Every result file produced by this project sits next to one, so a reader can
reconstruct which settings, library versions and seed produced a number.
"""

from __future__ import annotations

import json
import platform
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from sq4_explain.config import LoggingSettings, Settings


def setup_logging(settings: LoggingSettings, log_file: Path) -> None:
    """Replace the default loguru sink with a console and a rotating file sink."""
    logger.remove()
    logger.add(sys.stderr, level=settings.level, colorize=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        level=settings.level,
        rotation=settings.rotation,
        retention=settings.retention,
        encoding="utf-8",
    )


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch, and ask cuDNN for deterministic kernels.

    Determinism is not cosmetic here: run-to-run variation of the explanation
    is the quantity SQ4 measures, so variation from the framework has to be
    ruled out before variation from the method can be attributed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(preference: str = "auto") -> torch.device:
    """Return the best available device, or the one explicitly requested."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def environment_fingerprint() -> dict[str, str]:
    """Collect the library versions that could change a numerical result."""
    fingerprint = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    for name in ("torch_geometric", "captum", "numpy", "scipy"):
        try:
            module = __import__(name)
        except ImportError:  # pragma: no cover - optional at import time
            fingerprint[name] = "not installed"
        else:
            fingerprint[name] = getattr(module, "__version__", "unknown")
    return fingerprint


def write_manifest(
    destination: Path,
    settings: Settings,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the run manifest (settings, versions, timestamp) next to the results."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": environment_fingerprint(),
        "settings": json.loads(settings.model_dump_json()),
        **(extra or {}),
    }
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote run manifest to {}", destination)
    return destination
