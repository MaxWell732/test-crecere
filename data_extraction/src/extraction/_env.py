"""Process environment: confine caches to the project and expose pip CUDA libs to CTranslate2.

Must run before torch / transformers / huggingface_hub / ctranslate2 are imported.
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

from .paths import CACHE

_REEXEC_FLAG = "_EXTRACTION_ENV_READY"


def confine_caches() -> None:
    """Force every model download / cache / temp file under ./.cache (overrides user-level settings)."""
    targets = {
        "HF_HOME": CACHE / "huggingface",
        "TORCH_HOME": CACHE / "torch",
        "XDG_CACHE_HOME": CACHE / "xdg",
        "MPLCONFIGDIR": CACHE / "matplotlib",
        "NUMBA_CACHE_DIR": CACHE / "numba",
        "TMPDIR": CACHE / "tmp",
    }
    for var, path in targets.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[var] = str(path)
    os.environ.pop("HF_HUB_CACHE", None)
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ.pop("SENTENCE_TRANSFORMERS_HOME", None)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    import tempfile

    tempfile.tempdir = str(targets["TMPDIR"])


def nvidia_lib_dirs() -> list[str]:
    site = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    if not site.is_dir():
        return []
    return [str(p) for p in sorted(site.glob("*/lib")) if p.is_dir()]


def ensure_cuda_libs(argv: list[str]) -> None:
    """Re-exec once with LD_LIBRARY_PATH containing the venv's nvidia/*/lib dirs (spec §5.4).

    The dynamic loader only reads LD_LIBRARY_PATH at process start, so setting it in-process is not enough
    for ctranslate2's lazy dlopen of libcublas / libcudnn.
    """
    if os.environ.get(_REEXEC_FLAG) == "1":
        return
    dirs = nvidia_lib_dirs()
    current = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    os.environ[_REEXEC_FLAG] = "1"
    if not dirs or all(d in current for d in dirs):
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join(dirs + [p for p in current if p not in dirs])
    os.execv(sys.executable, [sys.executable, "-m", "extraction.cli", *argv])
