#!/usr/bin/env bash
# Install (if missing) and configure ccache for NInfer builds on Linux and WSL. Run once per
# machine; safe to run again.
#
# Why: a full rebuild is 10+ minutes and a public-header edit recompiles nearly everything. The
# cache is shared by every build tree on the machine, so a tree that changes a few files rebuilds
# only those. cmake/CompilerCache.cmake turns it on automatically when ccache is found; this
# script installs ccache and writes the settings that make sharing work:
#
#   base_dir  ccache rewrites absolute paths under it to be relative, so the same source in two
#             trees produces the same command and hits the same entry. Without it every tree has
#             its own, unshared entries. In a git checkout it is the main checkout (which holds
#             every linked worktree); otherwise it is $HOME, where WSL builds live as `git archive`
#             snapshots (~/ninfer-rel-*), none of which has a .git to find the main checkout from.
#             A checkout on /mnt/c (a Windows worktree seen from WSL) is not built from WSL and
#             does not decide the base. Override with --base DIR.
#   hash_dir  false. The working directory is not hashed (it only matters for debug info).
#   max_size  the cache is one per-user directory shared by all trees. WSL has its own cache, apart
#             from the Windows one: the objects are different.
#
# Installing needs the system package manager; as root (or via sudo) it does so:
#   Debian/Ubuntu: apt-get install ccache      macOS: brew install ccache
#
#   scripts/setup-ccache.sh                 install if needed, then configure
#   scripts/setup-ccache.sh --max-size 50G  larger cache
#   scripts/setup-ccache.sh --base ~/work   share across every tree under ~/work
#
# After a build: ccache -s     (hit rate; a second tree at the same commit should hit nearly all)
set -euo pipefail

max_size="30G"
base=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-size) max_size="${2:?--max-size needs a value}"; shift 2 ;;
    --base) base="${2:?--base needs a directory}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ccache_bin="$(command -v ccache || true)"
if [[ -z "$ccache_bin" && -x "$HOME/.local/bin/ccache" ]]; then
  ccache_bin="$HOME/.local/bin/ccache"
fi
if [[ -z "$ccache_bin" ]]; then
  echo "ccache is not installed; installing it."
  if [[ "$(id -u)" -eq 0 ]]; then
    sudo_cmd=()
  elif command -v sudo >/dev/null 2>&1; then
    sudo_cmd=(sudo)
  else
    echo "Install ccache first (apt-get install ccache / brew install ccache), then re-run." >&2
    exit 1
  fi
  if command -v apt-get >/dev/null 2>&1; then
    "${sudo_cmd[@]}" apt-get install -y ccache
  elif command -v brew >/dev/null 2>&1; then
    brew install ccache
  else
    echo "No supported package manager; install ccache and re-run." >&2
    exit 1
  fi
  ccache_bin="$(command -v ccache)"
fi

if [[ -z "$base" ]]; then
  base="$HOME"
  if common="$(git -C "$root" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"; then
    candidate="$(cd -- "$common/.." && pwd)"
    [[ "$candidate" == /mnt/* ]] || base="$candidate"
  fi
fi
base="$(cd -- "$base" && pwd)"

"$ccache_bin" --set-config "base_dir=$base"
"$ccache_bin" --set-config "hash_dir=false"
"$ccache_bin" --set-config "max_size=$max_size"

echo
"$ccache_bin" --version | head -n 1
"$ccache_bin" --show-config | grep -E "cache_dir|base_dir|hash_dir|max_size"
echo
echo "Reconfigure a build tree to pick it up; CMake prints 'Compiler cache: ...' when it is on."
