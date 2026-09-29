# This target deliberately receives no src/, CUDA, artifact, kernel, or target
# include root. It proves that the public product headers stand alone.
add_executable(ninfer_public_api_test "${CMAKE_CURRENT_LIST_DIR}/../test_public_api.cpp")
target_include_directories(ninfer_public_api_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME ninfer_public_api_test COMMAND ninfer_public_api_test)

ninfer_add_test(ninfer_wide_math_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_wide_math.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_multi_gpu_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_multi_gpu.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/../test_stage_plan.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/../test_kv_cache_ranks.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/../test_stage_link.cu"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_arena_ranks_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_arena_ranks.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_host_kv_clamp_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../host/test_host_kv_clamp.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_device_buffer_visibility_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_device_buffer_visibility.cu"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_vmm_graph_remap_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_vmm_graph_remap.cu"
  LIBRARIES ninfer_core CUDA::cuda_driver)

ninfer_add_test(ninfer_evictable_kv_pool_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_evictable_kv_pool.cu"
  LIBRARIES ninfer_core CUDA::cuda_driver)

ninfer_add_test(ninfer_evictable_weight_pool_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_evictable_weight_pool.cu"
  LIBRARIES ninfer_core)

set_tests_properties(
  ninfer_arena_ranks_test
  ninfer_device_buffer_visibility_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_device_test       SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_device.cpp"
  LIBRARIES ninfer_core)

set_tests_properties(ninfer_device_test PROPERTIES
  ENVIRONMENT_MODIFICATION "NINFER_CUDA_SYNC=unset:")
set(sync_modes spin blocking yield auto)
set(sync_flags 1 4 2 0)
foreach(mode flags IN ZIP_LISTS sync_modes sync_flags)
  add_test(NAME ninfer_device_sync_${mode}_test COMMAND ninfer_tests ninfer_device_test ${flags})
  set_tests_properties(ninfer_device_sync_${mode}_test PROPERTIES
    ENVIRONMENT "NINFER_CUDA_SYNC=${mode}" SKIP_RETURN_CODE 77)
endforeach()
add_test(NAME ninfer_device_sync_invalid_test COMMAND ninfer_tests ninfer_device_test --invalid-sync)
set_tests_properties(ninfer_device_sync_invalid_test PROPERTIES
  ENVIRONMENT "NINFER_CUDA_SYNC=invalid")

# A present-but-empty NINFER_CUDA_SYNC is only reachable through CTest's ENVIRONMENT property on
# POSIX. On Windows, CTest sets test-process environment variables the same way `_putenv`/
# SetEnvironmentVariable does: an empty value deletes the variable instead of setting it, so the
# child process sees NINFER_CUDA_SYNC unset (defaulting to spin) rather than empty-and-invalid --
# confirmed by running the same binary with the same env var set outside CTest's launch path, where
# it correctly throws. There is no Windows-side way to express "set to empty" through this
# mechanism, so this case is POSIX-only.
if(NOT WIN32)
  add_test(NAME ninfer_device_sync_empty_test COMMAND ninfer_tests ninfer_device_test --invalid-sync)
  set_tests_properties(ninfer_device_sync_empty_test PROPERTIES
    ENVIRONMENT "NINFER_CUDA_SYNC=")
endif()

ninfer_add_test(ninfer_decode_graph_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_decode_graph.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_tensor_test       SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_tensor.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_arena_test        SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_arena.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_materialization_budget_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_materialization_budget.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_kv_cache_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_kv_cache.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_state_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_state_store.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_gdn_replay_records_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_gdn_replay_records.cpp"
  LIBRARIES ninfer_core)

set_tests_properties(
  ninfer_device_test
  ninfer_decode_graph_test
  ninfer_arena_test
  ninfer_kv_cache_test
  ninfer_state_store_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_host_timing_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_host_timing.cpp"
  LIBRARIES ninfer_core)

# Standalone: the chat-template test runs it as an executable path.
ninfer_add_test(ninfer_jinja_test STANDALONE
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../text/test_jinja.cpp"
  LIBRARIES ninfer_jinja ninfer::json)

add_test(NAME ninfer_chat_templates_test
  COMMAND ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tests/text/test_chat_templates.py
          $<TARGET_FILE:ninfer_jinja_test>)
