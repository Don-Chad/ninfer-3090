# Writes ${OUT} with the build version. Run as a script (cmake -P) on every build so the commit
# suffix tracks HEAD; configure_file leaves the output untouched when the text is unchanged, so an
# unchanged version recompiles nothing.
#
# Inputs: SRC_DIR (repository root), IN (template), OUT (generated source).

file(STRINGS "${SRC_DIR}/VERSION" _version LIMIT_COUNT 1)
string(STRIP "${_version}" _version)
if(NOT _version)
  message(FATAL_ERROR "${SRC_DIR}/VERSION is empty")
endif()

set(NINFER_BUILD_VERSION "${_version}")
find_package(Git QUIET)
# .git is a directory in a checkout and a file in a worktree; a git archive has neither.
if(GIT_FOUND AND EXISTS "${SRC_DIR}/.git")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${SRC_DIR}" rev-parse --short=9 HEAD
    OUTPUT_VARIABLE _commit OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _commit_status ERROR_QUIET)
  if(_commit_status EQUAL 0 AND _commit)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" -C "${SRC_DIR}" describe --tags --exact-match HEAD
      OUTPUT_VARIABLE _tag OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _tag_status ERROR_QUIET)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" -C "${SRC_DIR}" status --porcelain --untracked-files=no
      OUTPUT_VARIABLE _dirty OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(NOT (_tag_status EQUAL 0 AND _tag STREQUAL "v${_version}"))
      string(APPEND NINFER_BUILD_VERSION "+${_commit}")
    endif()
    if(_dirty)
      string(APPEND NINFER_BUILD_VERSION "-dirty")
    endif()
  endif()
endif()

configure_file("${IN}" "${OUT}" @ONLY)
