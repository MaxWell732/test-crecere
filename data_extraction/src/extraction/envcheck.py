"""`extract check-env`: setup verification (spec §5.1, §5.3, §5.4). Prints PASS/FAIL per check; never prints tokens."""

from __future__ import annotations

import os
import platform
import shutil
import sys
from importlib import metadata

from . import paths


def _line(ok: bool | None, name: str, detail: str = "") -> bool | None:
    tag = {True: "PASS", False: "FAIL", None: "INFO"}[ok]
    print(f"[{tag}] {name}{': ' + detail if detail else ''}")
    return ok


def run() -> int:
    failures = 0
    _line(sys.version_info[:2] == (3, 12), "python 3.12", platform.python_version()) or (failures := failures + 1)
    _line(str(paths.CACHE) in os.environ.get("HF_HOME", ""), "HF_HOME confined", os.environ.get("HF_HOME", ""))
    _line(sys.prefix.startswith(str(paths.ROOT)), "venv inside project", sys.prefix)

    # torch (§5.3)
    try:
        import torch

        cuda = torch.cuda.is_available()
        archs = torch.cuda.get_arch_list() if cuda else []
        _line(cuda, "torch.cuda.is_available()", f"torch {torch.__version__}") or (failures := failures + 1)
        _line("sm_61" in archs, "sm_61 in torch.cuda.get_arch_list() (literal §5.3 check)", " ".join(archs))
        from .diarize import cuda_kernels_ok

        cap = torch.cuda.get_device_capability(0) if cuda else None
        _line(cuda_kernels_ok(), "kernels for device capability (sm_6x ≤ 6.1) + cuDNN conv on cuda",
              f"capability {cap}; see docs/decisions.md D-007") or (failures := failures + 1)
        if cuda:
            try:
                a = torch.randn(256, 256, device="cuda")
                (a @ a).sum().item()
                _line(True, "cuda matmul", torch.cuda.get_device_name(0))
            except Exception as exc:  # noqa: BLE001
                failures += 1
                _line(False, "cuda matmul", str(exc)[:200])
    except Exception as exc:  # noqa: BLE001
        failures += 1
        _line(False, "import torch", str(exc)[:200])

    # ctranslate2 (§5.4)
    try:
        import ctranslate2

        n = ctranslate2.get_cuda_device_count()
        _line(n >= 1, "ctranslate2 cuda devices", f"{n} (ctranslate2 {ctranslate2.__version__})") or (failures := failures + 1)
        types = ctranslate2.get_supported_compute_types("cuda") if n else set()
        _line("int8" in types, "int8 in ctranslate2 cuda compute types", str(sorted(types))) or (failures := failures + 1)
    except Exception as exc:  # noqa: BLE001
        failures += 1
        _line(False, "import ctranslate2", str(exc)[:200])
    _line(None, "LD_LIBRARY_PATH nvidia dirs", str(sum("nvidia" in p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":"))))

    for mod in ["pyannote.audio", "silero_vad", "faster_whisper", "parselmouth", "pysentimiento", "sentence_transformers",
                "torchcodec"]:
        try:
            __import__(mod)
            ver = None
            for dist in (mod.replace("_", "-"), mod, {"parselmouth": "praat-parselmouth"}.get(mod, mod)):
                try:
                    ver = metadata.version(dist)
                    break
                except metadata.PackageNotFoundError:
                    continue
            _line(True, f"import {mod}", ver or "")
        except Exception as exc:  # noqa: BLE001
            ok = False if mod != "torchcodec" else None  # torchcodec is optional: audio is passed in memory
            failures += 0 if ok is None else 1
            _line(ok, f"import {mod}", f"{type(exc).__name__}: {str(exc)[:160]}")

    _line(None, "ffmpeg on PATH", shutil.which("ffmpeg") or "absent (not required: see docs/decisions.md)")
    token_path = paths.CACHE / "huggingface" / "token"
    _line(token_path.exists() or bool(os.environ.get("HF_TOKEN")), "Hugging Face token (H2)",
          "present" if token_path.exists() else "absent -> run scripts/hf_login.sh")
    print(f"\n{failures} failing check(s)")
    return 1 if failures else 0
