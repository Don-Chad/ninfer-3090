set(ninfer_op_tests
  add_bias
  gelu
  silu_mul
  residual_add
  sigmoid_mul
  rmsnorm
  rmsnorm_pack_tail
  gated_rmsnorm
  l2norm
  gated_delta_net
  kimi_delta_attention
  causal_conv1d_silu
  layer_norm
  embedding
  argmax
  gdn_gating
  gdn_gating_proj
  rope
  vision_pos_embed
  sampling
  scalar
  cast
  prepare_ragged_prefix
  scatter
  scatter_bf16_batch
  target_logprobs
  position)
foreach(op IN LISTS ninfer_op_tests)
  ninfer_add_op_test(ninfer_${op}_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_${op}.cpp"
    LIBRARIES ninfer_ops)
endforeach()

ninfer_add_op_test(ninfer_linear_topk_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_linear_topk.cu"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_candidate_selector_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_candidate_selector.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_softmax_attention_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/causal_cache.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/plain_and_packed.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/context.cpp"
  LIBRARIES ninfer_ops)

add_test(NAME ninfer_softmax_attention_nvfp4_test
  COMMAND ninfer_tests ninfer_softmax_attention_test --nvfp4-only)

add_test(NAME ninfer_softmax_attention_k8v4_test
  COMMAND ninfer_tests ninfer_softmax_attention_test --k8v4-only)

# Upstream's DFlash2 verification sweep. ~460 s: six storage families (five upstream, plus this
# fork's rk8v4) over every narrow width, batch and base offset, so it gets a timeout well clear of
# the default 1500 s only if the machine is slow -- left at the default here because it finishes in
# under a third of it.
add_test(NAME ninfer_softmax_attention_dflash2_test
  COMMAND ninfer_tests ninfer_softmax_attention_test --dflash2-only)
# INT8-G64 FA2 prompt route: full prefill chunks and key splits over 4K-40K histories against the
# FP64 oracle at sampled query rows.
add_test(NAME ninfer_softmax_attention_int8_prompt_test
  COMMAND ninfer_tests ninfer_softmax_attention_test --int8-prompt-only)

set_tests_properties(
  ninfer_softmax_attention_nvfp4_test
  ninfer_softmax_attention_k8v4_test
  ninfer_softmax_attention_dflash2_test
  ninfer_softmax_attention_int8_prompt_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_op_test(ninfer_sliding_window_attention_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sliding_window_attention.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_kv_cache_append_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kv_cache_append.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_kv_plane_types_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kv_plane_types.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_rmsnorm_rope_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_rmsnorm_rope.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_context_kv_materialize_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_context_kv_materialize.cpp"
  LIBRARIES ninfer_ops)

add_test(NAME ninfer_kv_cache_append_nvfp4_test
  COMMAND ninfer_tests ninfer_kv_cache_append_test --nvfp4-only)

add_test(NAME ninfer_kv_cache_append_k8v4_test
  COMMAND ninfer_tests ninfer_kv_cache_append_test --k8v4-only)

set_tests_properties(
  ninfer_kv_cache_append_nvfp4_test
  ninfer_kv_cache_append_k8v4_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_op_test(ninfer_prepare_masked_block_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prepare_masked_block.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_sparse_moe_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sparse_moe.cpp"
  LIBRARIES ninfer_ops)

# The routing sort on its own, exhaustively. The Op test above covers SparseMoe end to end and
# would not localise a network that mis-sorts a minority of score permutations; this one runs the
# device function over all 8! orderings and all 3^8 tie patterns.
ninfer_add_op_test(ninfer_sparse_moe_route_network_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sparse_moe_route_network.cu"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_mtp_pack_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_pack.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_mtp_round_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_round.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_speculative_round_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_speculative_round.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_attn_input_proj_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_attn_input_proj.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gdn_input_proj_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_proj.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gdn_input_small_t_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_small_t.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_attn_input_small_t_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_attn_input_small_t.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_dynamic_grouped_conv_prepare_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dynamic_grouped_conv_prepare.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_linear_dynamic_grouped_conv_add_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_linear_dynamic_grouped_conv_add.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gdn_input_proj_conv_snapshot_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_proj_conv_snapshot.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gdn_input_proj_conv_record_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_proj_conv_record.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gated_delta_net_replay_record_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gated_delta_net_replay_record.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gdn_replay_fold_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_replay_fold.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gdn_state_fp16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_state_fp16.cpp"
  LIBRARIES ninfer_ops)

# Width 1 above covers the recurrent (T < 16) launcher only; this width-16 case is the smallest
# call that dispatches into the chunked staging/narrowing path added alongside FP16 state support.
ninfer_add_op_test(ninfer_gdn_state_fp16_chunked_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_state_fp16.cpp"
  LIBRARIES ninfer_ops
  TEST_ARGS 16)

ninfer_add_op_test(ninfer_vocabulary_transcode_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vocabulary_transcode.cpp"
  LIBRARIES ninfer_ops ninfer_artifact)

# Pure host resolution -- no device work, so it runs on a machine with no GPU at all.
ninfer_add_op_test(ninfer_route_coverage_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_route_coverage.cpp"
  LIBRARIES ninfer_ops)

include("${CMAKE_CURRENT_LIST_DIR}/linear/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_add/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_pair/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_swiglu/tests.cmake")
