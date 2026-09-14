#!/usr/bin/env bash
# H2 — Hugging Face login with the token stored inside this project (./.cache/huggingface/token), not in $HOME.
# Before running: create a READ token at https://huggingface.co/settings/tokens and accept the terms at
#   https://huggingface.co/pyannote/speaker-diarization-community-1
#   https://huggingface.co/pyannote/speaker-diarization-3.1   and   https://huggingface.co/pyannote/segmentation-3.0
# When asked "Add token as git credential?" answer n (that would write to ~/.git-credentials).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT_ROOT"
uv run hf auth login
uv run hf auth whoami
