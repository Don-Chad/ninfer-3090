ninfer_add_test(ninfer_qwen3_5_loading_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading_real.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen3_5_loading_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen3_5_frontend_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_frontend.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer_core ninfer::json)

ninfer_add_test(ninfer_qwen3_5_graft_loader_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_graft_loader.cpp"
  LIBRARIES ninfer_engine ninfer_core ninfer::json)

# The reasoning-loop guard's repeated-passage measure (CPU only).
ninfer_add_test(ninfer_qwen3_5_reasoning_loop_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_reasoning_loop.cpp"
  LIBRARIES ninfer_model_runtime)

ninfer_add_test(ninfer_qwen3_5_runtime_mechanisms_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_runtime_mechanisms.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_state_image_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_state_image_layout_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image_layout.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_context_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_context_store.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_native_transactions_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_native_transactions.cpp"
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_core)

ninfer_add_test(ninfer_qwen3_5_prefix_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_prefix_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_context_store_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_context_store_real.cpp"
  LIBRARIES ninfer_engine ninfer_s3_object_store)

ninfer_add_test(ninfer_qwen3_5_preemption_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_preemption_real.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_recovery_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_recovery_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_grammar_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_grammar_real.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_tools_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_tools_real.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_score_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_score_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_vision_workspace_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_workspace.cpp"
  LIBRARIES ninfer_model_runtime ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_vision_overlay_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_vision_overlay_real.cpp"
  LIBRARIES ninfer_engine)

# k=7 graph=1 optimized=1 batch=2 kv=int8 vision=0 state_slots=1. The argv fallbacks are k=15,
# batch=8 and 3 state slots, which want about 6.3 GB of runtime reservation and cannot fit beside the
# 20.4 GB artifact on a 24 GB card. This configuration exercises the same DFlash2 accept/rollback
# path and fits, so the test is coverage rather than a standing failure.
ninfer_add_test(ninfer_qwen3_5_dflash2_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash2_real.cpp"
  LIBRARIES ninfer_engine
  TEST_ARGS 7 1 1 2 int8 0 1)

ninfer_add_test(ninfer_qwen3_5_dflash_prefill_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dflash_prefill_real.cpp"
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_core)

ninfer_add_test(ninfer_qwen3_5_moe_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_moe_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_dflash_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash_real.cpp"
  LIBRARIES ninfer_engine)

# Captures the 27B's own KV/state after a real prompt prefix into graft containers, then checks that
# the installed direct graft reproduces the replayed prefix's greedy output across backends,
# concurrency and cache pressure. NINFER_TEST_GRAFT_REFERENCE=<phantom-kv prefill_kv container> also
# compares the captured tensors with an independently produced (HF) graft for layout agreement.
ninfer_add_test(ninfer_qwen3_5_graft_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_graft_real.cpp"
  LIBRARIES ninfer_engine ninfer_model_runtime ninfer_model_loading ninfer_core ninfer::json)
ninfer_add_test(ninfer_qwen3_5_stages_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_stages_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_test(ninfer_tool_call_parser_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../../test_tool_call_parser.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_visual_scatter_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_visual_scatter.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_bundle_command(ninfer_qwen3_5_prefix_real_command ninfer_tests
  ninfer_qwen3_5_prefix_real_test)
add_test(NAME ninfer_qwen3_5_agent_continuation_real_test
  COMMAND ${ninfer_qwen3_5_prefix_real_command})
set_tests_properties(ninfer_qwen3_5_agent_continuation_real_test PROPERTIES
  ENVIRONMENT "NINFER_PREFIX_REAL_SCENARIO=agent-continuation")

# A real-model test owns the single GPU while its artifact is resident.
set(ninfer_qwen3_5_real_tests
  ninfer_qwen3_5_loading_real_test
  ninfer_qwen3_5_native_transactions_test
  ninfer_qwen3_5_prefix_real_test
  ninfer_qwen3_5_agent_continuation_real_test
  ninfer_qwen3_5_context_store_real_test
  ninfer_qwen3_5_preemption_real_test
  ninfer_qwen3_5_recovery_real_test
  ninfer_qwen3_5_grammar_real_test
  ninfer_qwen3_5_tools_real_test
  ninfer_qwen3_5_score_real_test
  ninfer_qwen3_5_vision_workspace_test
  ninfer_qwen3_5_vision_overlay_real_test
  ninfer_qwen3_5_dflash2_real_test
  ninfer_qwen3_5_dflash_prefill_real_test
  ninfer_qwen3_5_moe_real_test
  ninfer_qwen3_5_dflash_real_test
  ninfer_qwen3_5_graft_real_test
  ninfer_qwen3_5_stages_real_test)
set_tests_properties(${ninfer_qwen3_5_real_tests} PROPERTIES
  SKIP_RETURN_CODE 77
  RUN_SERIAL TRUE
  LABELS "gpu;real")

set_tests_properties(
  ninfer_qwen3_5_state_image_test
  ninfer_qwen3_5_context_store_test
  ninfer_qwen3_5_visual_scatter_test
  PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")

ninfer_add_test(ninfer_qwen3_5_mtp_graph_profiles_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_graph_profiles.cpp"
  LIBRARIES ninfer_model_runtime ninfer_ops)

ninfer_add_test(ninfer_qwen3_5_mlp_a8_decode_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mlp_a8_decode_wiring.cpp"
  LIBRARIES ninfer_model_runtime ninfer_ops)

ninfer_add_test(ninfer_qwen3_5_ngram_proposer_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_proposer.cpp"
  LIBRARIES ninfer_core)
# STANDALONE: the Python schema oracle drives this probe by executable path.
ninfer_add_test(ninfer_qwen3_5_tool_constraints_test
  STANDALONE
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_tool_constraints.cpp"
  LIBRARIES ninfer_model_runtime ninfer_grammar ninfer::json)

add_test(NAME ninfer_qwen3_5_tool_schema_oracle_test
  COMMAND ${CMAKE_COMMAND} -E env
    "PYTHONUTF8=1" "NINFER_TOOL_PROBE=$<TARGET_FILE:ninfer_qwen3_5_tool_constraints_test>"
    ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/test_tool_schema.py")
