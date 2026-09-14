"""Stage `download`: bring the provider's WAVs into data/raw/<key>/ (spec §2.1).

Source: the local, read-only folder `paths.raw_source` (as delivered). Files are copied — never moved or modified
at the source — and each copy is verified by sha256.
"""

from __future__ import annotations

import os
import shutil
import time

from . import paths
from .cache import StageContext, StageResult, now_iso, write_stage_meta
from .config import load_config, sha256_file, stage_hash
from .logging_utils import get_logger


def run(force: bool = False) -> StageResult:
    cfg = load_config()
    logger = get_logger("download")
    ctx = StageContext(stage="download", cfg=cfg, key=stage_hash("download", cfg), logger=logger)
    result = StageResult()
    started, t0 = now_iso(), time.time()
    src_root = (paths.ROOT / cfg["paths"]["raw_source"]).resolve()
    expected = cfg["paths"]["expected_files_per_group"]

    for gkey, g in cfg["paths"]["raw_groups"].items():
        dst = paths.ensure_dir(paths.RAW / gkey)
        src = src_root / g["folder"]
        if src.is_dir():
            ctx.stats[f"source_{gkey}"] = "local"
            for f in sorted(src.glob("*.wav")):
                target = dst / f.name
                if target.exists() and not force and target.stat().st_size == f.stat().st_size:
                    result.skipped.append(f.name)
                    continue
                try:
                    tmp = dst / f".{f.name}.part"
                    shutil.copyfile(f, tmp)
                    if sha256_file(tmp) != sha256_file(f):
                        raise OSError("sha256 mismatch after copy")
                    os.replace(tmp, target)
                    result.ok.append(f.name)
                except Exception as exc:  # noqa: BLE001
                    result.failed[f.name] = f"{type(exc).__name__}: {exc}"
                    logger.error("copy failed for a file in %s: %s", gkey, exc)
        else:
            result.failed[gkey] = f"source folder not found: {src}"
        n = len(list(dst.glob("*.wav")))
        ctx.stats[f"n_files_{gkey}"] = n
        if n != expected:
            logger.warning("%s: %d wav files, expected %d", gkey, n, expected)

    write_stage_meta("download", ctx, result, started, time.time() - t0)
    logger.info("download: copied=%d already_present=%d failed=%d | %s", len(result.ok), len(result.skipped),
                len(result.failed), {k: v for k, v in ctx.stats.items() if k.startswith("n_files")})
    return result
