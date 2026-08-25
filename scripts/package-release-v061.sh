#!/usr/bin/env bash
set -euo pipefail

release_tag='0.6.1-rtx3090'
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="${NINFER_BUILD_ROOT:-$repo_root/build-v061-linux-vcpkg}"
dist_root="$repo_root/dist"
product_name="ninfer-rtx3090-linux-x64-$release_tag"
product_root="$dist_root/$product_name"
archive_name="$product_name.tar.gz"
archive_path="$dist_root/$archive_name"
checksum_path="$dist_root/SHA256SUMS-v0.6.1-linux.txt"

mkdir -p -- "$dist_root"
case "$product_root" in "$dist_root/$product_name") ;; *) exit 1 ;; esac
rm -rf -- "$product_root"
rm -f -- "$archive_path"
mkdir -- "$product_root"

for product in 'apps/ninfer:ninfer' 'apps/ninfer-serve:ninfer-serve' 'bench/ninfer_bench:ninfer_bench'; do
  source_path="$build_root/${product%%:*}"
  destination="$product_root/${product#*:}"
  [[ -f "$source_path" ]] || { printf 'Missing release product: %s\n' "$source_path" >&2; exit 1; }
  cp -- "$source_path" "$destination"
done
cp -- "$repo_root/VERSION" "$repo_root/LICENSE" "$repo_root/RELEASE_NOTES_0.6.1.md" "$product_root/"
cp -- "$repo_root/docs/rtx-3090-linux.md" "$product_root/README.md"
cp -- "$repo_root"/scripts/{download-qwen38.sh,run-qwen38-c1.sh,run-qwen38-c8.sh,run-qwen38-vision.sh,download-qwen36-35b-vision.sh,run-qwen36-35b-vision.sh} "$product_root/"
(
  cd -- "$product_root"
  mapfile -d '' files < <(find . -maxdepth 1 -type f ! -name SHA256SUMS.txt -print0 | LC_ALL=C sort -z)
  sha256sum -- "${files[@]}" > SHA256SUMS.txt
)
tar -C "$dist_root" -czf "$archive_path" "$product_name"
(
  cd -- "$dist_root"
  sha256sum -- "$archive_name" > "$(basename -- "$checksum_path")"
)
du -h -- "$archive_path" "$checksum_path"
