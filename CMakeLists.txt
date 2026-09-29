cmake_minimum_required(VERSION 3.28)

# The current implementation supports Ampere sm_86 and early Ada sm_89, plus sm_80 (GA100, e.g.
# CMP 170HX) as an unmeasured compatibility target: it shares the sm_86 instruction set the
# kernels use, but none of its route tables have been swept. Set and validate the
# architecture before CUDA compiler detection so CMake never injects a generic
# host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^(80|86|89)$")
  message(FATAL_ERROR
    "NInfer supports CMAKE_CUDA_ARCHITECTURES=80, 86 or 89; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)

# NINFER_SM8X_COMPAT gates every Ampere/early-Ada compatibility path. ninfer_ops also sets it
# PUBLIC for its own consumers, but that link is PRIVATE through ninfer_engine, so apps and tests
# would not otherwise see it. Defining it globally keeps a single consistent answer for every
# target, including apps/ninfer-perplexity, whose upstream default KV profile is FP8.
if(CMAKE_CUDA_ARCHITECTURES MATCHES "^(80|86|89)$")
  add_compile_definitions(NINFER_SM8X_COMPAT=1)
endif()

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI, server and perplexity evaluator" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(MSVC)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
  # /Zc:__cplusplus makes MSVC report the standard it is actually compiling. Without it the macro
  # stays at 199711L even under /std:c++20, and CUDA's <vector_types.h> then hides the constexpr
  # dim3 constructors behind its own __cplusplus >= 201103L test, which fails to compile every
  # translation unit that builds a dim3 in a constant expression.
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:__cplusplus>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>
    # spdlog's bundled fmt asserts the source/execution charset is UTF-8; MSVC defaults to the
    # active code page instead. Needed by every translation unit that includes fmt/spdlog
    # headers, not just spdlog's own .cpp files, so this is global rather than per-target.
    $<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
  # CUDA's static runtime requests LIBCMT even when the application and vcpkg dependencies use
  # the project-wide /MD runtime. Keep one CRT in the final process. $<HOST_LINK:...> keeps this
  # off the CUDA relocatable device-link step, which forwards it to cl.exe without the /link
  # passthrough separator and only produces a harmless but noisy D9002 "unknown option" warning.
  add_link_options($<HOST_LINK:/NODEFAULTLIB:LIBCMT>)
else()
  # Schedule dispatch in this project is a switch per Op mapping a schedule id to a kernel launch,
  # and a missing `return` there is close to invisible: the Op runs the intended kernel and then
  # the next case's kernel over the top of it. Both compute the same result, so the output stays
  # correct, every test passes, and the only symptom is that the Op costs twice what it should.
  # Two of these shipped in the last catch-up merge (q4_q5 attention input, q5 linear add) and
  # both were dormant because the live route table never selected the affected schedule.
  #
  # MSVC has no cheap equivalent -- C26819 needs /analyze -- so on Windows this is caught by
  # review and by comparing an Op's measured cost against its chosen kernel's. On every other
  # toolchain, and therefore in the direction upstream merges arrive from, it is a build error.
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:-Wimplicit-fallthrough>
    $<$<COMPILE_LANGUAGE:C,CXX>:-Werror=implicit-fallthrough>)

  # Keep the builder's absolute paths out of the shipped binaries. `__FILE__` and nvcc's own source
  # references otherwise embed wherever the tree happened to sit: the v0.9.0 Linux binaries carry
  # `/home/ash/ninfer-rel/src/...` about 200 times, and v0.8.1 carried `/mnt/c/ninfer-fork/...` 405
  # times. Not a secret -- the file names are public and no path contains a home directory -- but
  # it makes every assertion message shorter and more readable, and it makes two builds of the same
  # commit from different directories produce closer output.
  #
  # Verified on this box under real Linux g++: `/tmp/ftest/sub/a.cpp` becomes `./sub/a.cpp`, and
  # the absolute prefix disappears from `strings` entirely.
  #
  # **There is no working MSVC equivalent, and this was measured rather than assumed.** MSVC takes
  # `__FILE__` from the path as written on the command line, and CMake writes absolute paths;
  # `/d1trimfile:`, the usual suggestion, is undocumented and had *no effect* on `__FILE__` when
  # tested against 14.44.35207 with an absolute source path. So the Windows binaries keep their
  # ~466 occurrences until something better turns up.
  #
  # -Xcompiler reaches the host half of a .cu translation unit. Device-side `__FILE__` comes from
  # nvcc's own frontend and has no documented remapping flag, so it is not covered here.
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:-ffile-prefix-map=${PROJECT_SOURCE_DIR}=.>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffile-prefix-map=${PROJECT_SOURCE_DIR}=.>)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS ninfer_link=1)
  set(CMAKE_JOB_POOL_LINK ninfer_link)
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
  message(FATAL_ERROR
    "NInfer requires CUDA 12.8 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()

# Tests also exercise product input, logging and protocol adapters.
set(NINFER_BUILD_PRODUCT_SUPPORT OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_PRODUCT_SUPPORT ON)
endif()

include(cmake/Dependencies.cmake)
include(cmake/NinferTargets.cmake)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
  add_subdirectory(tools)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
