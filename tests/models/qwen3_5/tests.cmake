ninfer_add_test(ninfer_qwen3_5_loading_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading_real.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen3_5_loading_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading.cpp"
  LIBRARIES ninfer_model_loading)

set_tests_properties(
  ninfer_qwen3_5_loading_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_frontend_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_frontend.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer_core ninfer::json)

ninfer_add_test(ninfer_qwen3_5_graft_loader_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_graft_loader.cpp"
  LIBRARIES ninfer_engine ninfer_core ninfer::json)

ninfer_add_test(ninfer_qwen3_5_runtime_mechanisms_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_runtime_mechanisms.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_state_image_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image.cpp"
  LIBRARIES ninfer_engine ninfer_core)

set_tests_properties(
  ninfer_qwen3_5_state_image_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_state_image_layout_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image_layout.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_context_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_context_store.cpp"
  LIBRARIES ninfer_engine ninfer_core)

set_tests_properties(
  ninfer_qwen3_5_context_store_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_prefix_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_prefix_real.cpp"
  LIBRARIES ninfer_engine)

set_tests_properties(
  ninfer_qwen3_5_prefix_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_score_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_score_real.cpp"
  LIBRARIES ninfer_engine)

set_tests_properties(
  ninfer_qwen3_5_score_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_vision_workspace_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_workspace.cpp"
  LIBRARIES ninfer_model_runtime ninfer_engine)

set_tests_properties(
  ninfer_qwen3_5_vision_workspace_test
  PROPERTIES SKIP_RETURN_CODE 77)

# k=7 graph=1 optimized=1 batch=2 kv=int8 vision=0 state_slots=1. The argv fallbacks are k=15,
# batch=8 and 3 state slots, which want about 6.3 GB of runtime reservation and cannot fit beside the
# 20.4 GB artifact on a 24 GB card. This configuration exercises the same DFlash2 accept/rollback
# path and fits, so the test is coverage rather than a standing failure.
ninfer_add_test(ninfer_qwen3_5_dflash2_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash2_real.cpp"
  LIBRARIES ninfer_engine
  TEST_ARGS 7 1 1 2 int8 0 1)

set_tests_properties(
  ninfer_qwen3_5_dflash2_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_moe_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_moe_real.cpp"
  LIBRARIES ninfer_engine)

set_tests_properties(
  ninfer_qwen3_5_moe_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_stages_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_stages_real.cpp"
  LIBRARIES ninfer_engine)

set_tests_properties(
  ninfer_qwen3_5_stages_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_dflash_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash_real.cpp"
  LIBRARIES ninfer_engine)

set_tests_properties(
  ninfer_qwen3_5_dflash_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_tool_call_parser_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../../test_tool_call_parser.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_visual_scatter_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_visual_scatter.cpp"
  LIBRARIES ninfer_engine ninfer_core)

set_tests_properties(
  ninfer_qwen3_5_visual_scatter_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_mtp_graph_profiles_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_graph_profiles.cpp"
  LIBRARIES ninfer_model_runtime ninfer_ops)

ninfer_add_test(ninfer_qwen3_5_mlp_a8_decode_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mlp_a8_decode_wiring.cpp"
  LIBRARIES ninfer_model_runtime ninfer_ops)

ninfer_add_test(ninfer_qwen3_5_lookup_draft_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_lookup_draft.cpp"
  LIBRARIES ninfer_core)
