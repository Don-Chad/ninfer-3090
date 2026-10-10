target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/causal_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/bf16/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/bf16/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/packed_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/context_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/sliding_window_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/launch.cu"
)

# Quantized causal attention. sm_80/86/89 (NINFER_SM8X_COMPAT) runs this fork's small-T, chunked
# small-T and prompt kernels for the INT8 family (INT8-G64, rk8v4, rk4v4), FP8, NVFP4 and K8V4.
# Upstream's per-format fp8/, int8/, nvfp4/ and k8v4/ directories need sm_120a instructions
# (block-scaled FP8/FP4 MMA, TMA, mbarrier) and lack the rk8v4/rk4v4 codings; they are listed only
# for an sm_120a build, which this fork's top-level CMakeLists does not currently accept.
if(CMAKE_CUDA_ARCHITECTURES MATCHES "^(80|86|89)$")
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_fp8.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_nvfp4.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_k8v4.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_i8_fa2.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_fp8.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_nvfp4.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_k8v4.cu"
  )
else()
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/fp8/launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/fp8/tiled_launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/fp8/plan.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/int8/launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/int8/plan.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/nvfp4/launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/nvfp4/plan.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/k8v4/launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/k8v4/tiled_launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/k8v4/plan.cpp"
  )
  if(TARGET ninfer_nvfp4_non_rdc)
    target_sources(ninfer_nvfp4_non_rdc PRIVATE
      "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/nvfp4/tiled_launch.cu"
    )
  endif()
endif()
