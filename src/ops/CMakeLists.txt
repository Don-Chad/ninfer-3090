# Compile policy is owned here; each semantic family explicitly contributes sources.
add_library(ninfer_ops STATIC)
ninfer_internal_includes(ninfer_ops)
ninfer_cuda_archive(ninfer_ops)
target_link_libraries(ninfer_ops PUBLIC ninfer_core)
# The opt-in cuBLAS prefill route hands its GEMM to cuBLAS; see
# ops/linear_swiglu/q4cublas/w4_cublas_prefill.h for why that route exists.
target_link_libraries(ninfer_ops PRIVATE CUDA::cublas)

# Warp-specialized SM120 NVFP4 kernels use setmaxnreg to transfer registers from
# producer to consumer warpgroups. RDC makes ptxas discard that contract. These
# self-contained kernels therefore use a non-RDC archive and host launchers.
# The SM120 tensor-core qualifiers do not exist on sm_86/sm_89, so this fork does not create
# the archive there; family manifests add to it only when the target exists.
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  add_library(ninfer_nvfp4_non_rdc STATIC)
  ninfer_internal_includes(ninfer_nvfp4_non_rdc)
  ninfer_cuda_non_rdc_archive(ninfer_nvfp4_non_rdc)
  target_link_libraries(ninfer_nvfp4_non_rdc PRIVATE ninfer_core CUDA::cudart CUDA::cuda_driver)
  target_link_libraries(ninfer_ops PRIVATE ninfer_nvfp4_non_rdc)
endif()

include(basic_sources.cmake)
include("${CMAKE_CURRENT_LIST_DIR}/attn_input_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/candidate_selector/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/context_kv_materialize/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/dynamic_grouped_conv/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_gating_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_input_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/kv_cache/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_add/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_pair/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_swiglu/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_topk/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/rmsnorm_rope/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/softmax_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sparse_moe/sources.cmake")

if(CMAKE_CUDA_ARCHITECTURES MATCHES "^(80|86|89)$")
  get_target_property(ninfer_ops_sources ninfer_ops SOURCES)
  # NVFP4 W4A4 uses mma.sync...kind::mxf4nvf4.block_scale; FP8 A8 uses
  # mma.sync...kind::f8f6f4. Both qualifiers are Blackwell-only, and sm_80/86/89 have no FP8 or FP4
  # tensor-core path to fall back to, so the owning translation units cannot be compiled here.
  # The A16 dequantizing routes for both formats are kept and remain fully functional.
  list(FILTER ninfer_ops_sources EXCLUDE REGEX "nvfp4/.+_a4\\.cu$")
  list(FILTER ninfer_ops_sources EXCLUDE REGEX "fp8/.+_a8\\.cu$")
  set_property(TARGET ninfer_ops PROPERTY SOURCES ${ninfer_ops_sources})
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/nvfp4_sm86_stubs.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/fp8_sm86_stubs.cpp")
  target_compile_definitions(ninfer_ops PUBLIC NINFER_SM8X_COMPAT=1)
endif()
