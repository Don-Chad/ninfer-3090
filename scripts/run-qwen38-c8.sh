#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
server="${NINFER_SERVER:-$root/ninfer-serve}"
if [[ ! -x "$server" && -x "$root/../build-sm86/apps/ninfer-serve" ]]; then
  server="$root/../build-sm86/apps/ninfer-serve"
fi
model="${1:-${NINFER_MODEL_DIR:-$root/models}/qwen3_8_27b.ninfer}"

if [[ ! -x "$server" ]]; then
  printf 'Missing executable: %s\n' "$server" >&2
  printf '%s\n' 'Set NINFER_SERVER or build build-sm86/apps/ninfer-serve.' >&2
  exit 1
fi
if [[ ! -f "$model" ]]; then
  printf 'Missing model: %s\n' "$model" >&2
  printf '%s\n' 'Run download-qwen38.sh or pass the model path.' >&2
  exit 1
fi

printf '%s\n' 'Starting Qwen3.8-27B at http://127.0.0.1:8080/v1'
printf '%s\n' 'Profile: up to eight requests, 8K context, MTP3, ReplaySSM'
exec "$server" "$model" \
  --host 127.0.0.1 --port 8080 \
  --max-context 8192 --kv-capacity 16384 \
  --max-concurrency 8 --max-pending-requests 32 \
  --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
