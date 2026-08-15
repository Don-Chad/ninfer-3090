#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model_dir="${NINFER_MODEL_DIR:-$root/models}"
model="$model_dir/qwen3_8_27b.ninfer"

mkdir -p -- "$model_dir"
printf '%s\n' 'Downloading Qwen3.8-27B NInfer model...'
if ! curl -L -C - --fail --output "$model" \
  'https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/main/qwen3_8_27b.ninfer'; then
  printf '%s\n' 'Download failed. Run this script again to resume.' >&2
  exit 1
fi
printf 'Model ready: %s\n' "$model"
