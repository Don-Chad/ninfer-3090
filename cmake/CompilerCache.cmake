# Compiler cache (ccache) for C, C++ and CUDA, shared by every worktree of this checkout.
#
# A full rebuild here is 10+ minutes, mostly a few multi-minute CUDA translation units, and a public
# header edit (include/ninfer/types.h) recompiles nearly everything. Work is done in several git
# worktrees at once, and any one of them changes only a few files, so the cache has to be shared:
#
#   * The cache directory is ccache's per-user default, already common to every tree.
#   * ccache hashes absolute paths unless told otherwise, so the same source in two worktrees would
#     never hit. CCACHE_BASEDIR rewrites absolute paths under it to be relative to the build
#     directory; every tree has the same <tree>/build-ninja beside <tree>/src layout, so identical
#     sources then produce identical commands. The base is the main checkout (the parent of git's
#     common directory), which contains every worktree under .claude/worktrees/.
#   * CCACHE_NOHASHDIR keeps the working directory out of the hash (it only matters for debug info,
#     which Release builds do not emit).
#
# Run scripts/setup-ccache.{ps1,sh} once per machine to install ccache and set its size.
# Release is the default and is cached. MSVC /Zi (Debug, RelWithDebInfo) writes a shared PDB and
# cannot be cached by ccache.
#
# NINFER_COMPILER_CACHE=OFF disables this; a launcher the caller already set is left alone.

option(NINFER_COMPILER_CACHE "Use ccache for C, C++ and CUDA compilation when it is installed" ON)

if(NOT NINFER_COMPILER_CACHE)
  return()
endif()

if(CMAKE_C_COMPILER_LAUNCHER OR CMAKE_CXX_COMPILER_LAUNCHER OR CMAKE_CUDA_COMPILER_LAUNCHER)
  message(STATUS "Compiler cache: using the compiler launcher already configured")
  return()
endif()

# scripts/setup-ccache.ps1 installs ccache.exe flat into this directory when no package manager
# provides one.
set(_ninfer_ccache_hints "$ENV{LOCALAPPDATA}/ccache-bin" "$ENV{HOME}/.local/bin")
find_program(NINFER_CCACHE_PROGRAM ccache HINTS ${_ninfer_ccache_hints})
unset(_ninfer_ccache_hints)

if(NOT NINFER_CCACHE_PROGRAM)
  message(STATUS
    "Compiler cache: ccache not found; builds are uncached. Run scripts/setup-ccache.ps1 "
    "(Windows) or scripts/setup-ccache.sh (Linux) once, then reconfigure.")
  return()
endif()

# The main checkout is the parent of git's common directory, also when this is a linked worktree.
# An explicit CCACHE_BASEDIR in the environment wins. Outside git (a `git archive` snapshot, as on
# WSL) nothing is passed and the base_dir in ccache's own config applies, which
# scripts/setup-ccache.sh sets to $HOME there so sibling snapshots share.
set(_ninfer_ccache_base "$ENV{CCACHE_BASEDIR}")
if(NOT _ninfer_ccache_base)
  find_package(Git QUIET)
  if(GIT_FOUND)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_SOURCE_DIR}" rev-parse --path-format=absolute
              --git-common-dir
      OUTPUT_VARIABLE _ninfer_git_common
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE _ninfer_git_status)
    if(_ninfer_git_status EQUAL 0 AND _ninfer_git_common)
      get_filename_component(_ninfer_ccache_base "${_ninfer_git_common}/.." ABSOLUTE)
    endif()
  endif()
endif()

# The root CMakeLists adds -Xcompiler=-ffile-prefix-map=<source dir>=. to every CUDA compile on
# GCC/Clang so the host half of a .cu embeds relative paths. ccache rewrites an argument that is a
# path under the base, but not one wrapped in -Xcompiler=, so the absolute source directory would
# make every worktree's CUDA key different. The remapping yields the same output in every tree,
# so leaving that one option out of the hash is correct. (C and C++ spell it plainly, which
# ccache already normalizes.)
set(_ninfer_ccache_launcher
  "${CMAKE_COMMAND}" -E env CCACHE_NOHASHDIR=1
  "CCACHE_IGNOREOPTIONS=-Xcompiler=-ffile-prefix-map=*")
if(_ninfer_ccache_base)
  list(APPEND _ninfer_ccache_launcher "CCACHE_BASEDIR=${_ninfer_ccache_base}")
  set(_ninfer_ccache_base_note "base ${_ninfer_ccache_base}")
else()
  set(_ninfer_ccache_base_note "base_dir from ccache config; run scripts/setup-ccache.* if unset")
endif()
list(APPEND _ninfer_ccache_launcher "${NINFER_CCACHE_PROGRAM}")
set(CMAKE_C_COMPILER_LAUNCHER "${_ninfer_ccache_launcher}")
set(CMAKE_CXX_COMPILER_LAUNCHER "${_ninfer_ccache_launcher}")
set(CMAKE_CUDA_COMPILER_LAUNCHER "${_ninfer_ccache_launcher}")
message(STATUS "Compiler cache: ${NINFER_CCACHE_PROGRAM} (${_ninfer_ccache_base_note})")
unset(_ninfer_ccache_base_note)
unset(_ninfer_ccache_launcher)
unset(_ninfer_ccache_base)
unset(_ninfer_git_common)
unset(_ninfer_git_status)
