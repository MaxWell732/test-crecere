"""Per-stage logging to data/logs/<stage>.log (+ concise console output)."""

from __future__ import annotations

import logging
import sys

from . import paths

_FMT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def get_logger(stage: str) -> logging.Logger:
    """Logger for a stage; the file handler follows `paths.LOGS` (tests redirect it)."""
    logger = logging.getLogger(f"extract.{stage}")
    target = paths.LOGS / f"{stage}.log"
    if getattr(logger, "_extract_target", None) == target:
        return logger
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    paths.ensure_dir(paths.LOGS)
    fh = logging.FileHandler(target, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FMT))
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-7s %(name)s | %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger._extract_target = target  # type: ignore[attr-defined]
    return logger
