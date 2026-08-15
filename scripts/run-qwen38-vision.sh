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

printf '%s\n' 'Starting Qwen3.8-27B Vision at http://127.0.0.1:8080/v1'
printf '%s\n' 'Tested RTX 3090 profile: one request, 32K context, ReplaySSM and MTP3'
exec "$server" "$model" \
  --host 127.0.0.1 --port 8080 \
  --max-context 32768 --kv-capacity 32768 \
  --max-concurrency 1 --max-pending-requests 8 \
  --prefill-chunk 512 --kv-dtype int8 --default-max-tokens 1024 \
  --vision --spec mtp --draft-tokens 3 --lm-head-draft
