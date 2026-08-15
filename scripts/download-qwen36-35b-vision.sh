#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model_dir="${NINFER_MODEL_DIR:-$root/models}"
model="$model_dir/qwen3_6_35b_a3b.ninfer"

mkdir -p -- "$model_dir"
printf '%s\n' 'Downloading the RTX 3090-compatible Qwen3.6-35B-A3B vision model...'
if ! curl -L -C - --fail --output "$model" \
  'https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/resolve/c8b8c1c0df4c74df3c190c6aa3a7f24dc614721c/qwen3_6_35b_a3b.ninfer'; then
  printf '%s\n' 'Download failed. Run this script again to resume.' >&2
  exit 1
fi
printf 'Model ready: %s\n' "$model"
