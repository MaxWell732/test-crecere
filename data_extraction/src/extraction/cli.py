"""Entry point: `uv run extract <stage> [--calls C001,C002] [--force]`.

Composite commands (`audio-all`, `features-all`) run each stage as its own subprocess so that GPU stages never
share a process (spec §1.6).
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys

from . import _env

# stage name -> (module, function, accepts calls/force)
STAGE_COMMANDS: dict[str, tuple[str, str, str]] = {
    "download": ("extraction.download", "run", "force"),
    "inventory": ("extraction.inventory", "run", "calls"),
    "preprocess": ("extraction.preprocess", "run", "calls"),
    "vad": ("extraction.vad", "run", "calls"),
    "asr": ("extraction.asr", "run", "calls"),
    "diarize": ("extraction.diarize", "run", "calls"),
    "turns": ("extraction.turns", "run", "calls"),
    "roles-render": ("extraction.roles", "render", "calls"),
    "roles-validate": ("extraction.roles", "validate", "calls"),
    "roles-review-sheet": ("extraction.roles", "build_review_sheet", "calls"),
    "canonical": ("extraction.canonical", "run", "calls"),
    "features-dynamics": ("extraction.features.dynamics", "run", "calls"),
    "features-prosody": ("extraction.features.prosody", "run", "calls"),
    "features-lexical": ("extraction.features.lexical", "run", "calls"),
    "features-embeddings": ("extraction.features.embeddings", "run", "calls"),
    "objection-qa": ("extraction.features.embeddings", "objection_qa", "none"),
    "validation-select": ("extraction.validation", "select", "force"),
    "annotate-status": ("extraction.annotation", "progress", "pass"),
    "annotate-validate": ("extraction.annotation", "validate", "pass"),
    "validate": ("extraction.validation", "run", "none"),
    "tables": ("extraction.tables", "run", "none"),
}
COMPOSITES = {
    "audio-all": ["download", "inventory", "preprocess", "vad", "asr", "diarize", "turns"],
    "features-all": ["features-dynamics", "features-prosody", "features-lexical", "features-embeddings"],
}
HELP = {
    "check-env": "verify Python/torch/CUDA/ctranslate2/model packages (§5.3–5.4)",
    "download": "copy + sha256-verify raw WAVs into data/raw",
    "inventory": "E0 anonymous IDs, QC, censor detection, manifest",
    "preprocess": "E1 16 kHz model copy",
    "vad": "E2 Silero VAD, SNR, speech ratio",
    "asr": "E3 faster-whisper (GPU process)",
    "diarize": "E4 pyannote diarization (GPU process)",
    "turns": "E5 word/speaker fusion, turns, backchannels, interruptions, latency",
    "roles-render": "E6 render role-annotation inputs",
    "roles-validate": "E6 validate role annotations (schema, turn_ids, quotes)",
    "roles-review-sheet": "E6 build data/interim/review/roles_review.csv for H5",
    "canonical": "E7 canonical JSON + rendered transcripts (requires approved role review)",
    "features-dynamics": "E8a", "features-prosody": "E8b", "features-lexical": "E8c", "features-embeddings": "E8d",
    "objection-qa": "E8d QA: cluster objection quotes typed `other`",
    "validation-select": "E10 seeded dev/gold/WER selection + templates",
    "annotate-status": "E9 progress table", "annotate-validate": "E9 validate annotation files",
    "validate": "E10 metrics + variable reliability", "tables": "E11 final tables + data dictionary",
    "audio-all": "download → E0 … E5, each stage in its own process",
    "features-all": "E8a–E8d, each in its own process",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="extract", description="Collections AI-vs-human audio extraction pipeline")
    sub = p.add_subparsers(dest="command", required=True, metavar="<stage>")
    for name in ["check-env", *STAGE_COMMANDS, *COMPOSITES]:
        sp = sub.add_parser(name, help=HELP.get(name, ""))
        sp.add_argument("--calls", help="comma-separated call_ids, e.g. C001,C002")
        sp.add_argument("--force", action="store_true", help="recompute even if cached outputs match")
        if name == "asr":
            sp.add_argument("--device", choices=["cuda", "cpu"], help="runtime device override (not hashed)")
        if name in ("annotate-status", "annotate-validate"):
            sp.add_argument("--pass", dest="pass_name", default="a", choices=["a", "b", "rerun_a", "roles"])
    return p


def _dispatch(args: argparse.Namespace) -> int:
    calls = [c.strip() for c in args.calls.split(",")] if args.calls else None
    if args.command == "check-env":
        from . import envcheck

        return envcheck.run()
    if args.command in COMPOSITES:
        for stage in COMPOSITES[args.command]:
            cmd = [sys.executable, "-m", "extraction.cli", stage]
            if args.calls:
                cmd += ["--calls", args.calls]
            if args.force:
                cmd.append("--force")
            print(f"==> {stage}", flush=True)
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                print(f"stage {stage} exited with {rc}; stopping {args.command}", file=sys.stderr)
                return rc
        return 0
    module, func, mode = STAGE_COMMANDS[args.command]
    fn = getattr(importlib.import_module(module), func)
    if args.command == "asr" and getattr(args, "device", None):
        result = fn(calls=calls, force=args.force, device=args.device)
    elif mode == "calls":
        result = fn(calls=calls, force=args.force)
    elif mode == "force":
        result = fn(force=args.force)
    elif mode == "pass":
        result = fn(pass_name=args.pass_name, calls=calls)
    else:
        result = fn()
    if isinstance(result, int):
        return result
    failed = getattr(result, "failed", None)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _env.ensure_cuda_libs(argv)  # may re-exec this process with LD_LIBRARY_PATH set
    _env.confine_caches()
    return _dispatch(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
