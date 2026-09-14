#!/usr/bin/env bash
# §5.5 smoke test: run the given stages on the 3 smoke calls (data/interim/smoke_calls.txt, 2 + 1 across groups),
# sampling GPU memory once per second. Usage:  scripts/smoke_test.sh asr diarize turns
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT_ROOT"
CALLS=$(tr -d '[:space:]' < data/interim/smoke_calls.txt)
[ -n "$CALLS" ] || { echo "data/interim/smoke_calls.txt is empty"; exit 1; }
for stage in "$@"; do
  log="data/logs/smoke_vram_${stage}.log"
  : > "$log"
  ( while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits >> "$log"; sleep 1; done ) &
  sampler=$!
  echo "==> $stage on $CALLS"
  set +e
  uv run extract "$stage" --calls "$CALLS" --force
  rc=$?
  set -e
  kill "$sampler" 2>/dev/null || true
  echo "    exit=$rc  peak VRAM MiB,util%: $(sort -t, -k1 -n "$log" | tail -1)"
  [ "$rc" -eq 0 ] || exit "$rc"
done
