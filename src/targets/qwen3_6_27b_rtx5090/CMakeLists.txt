add_library(ninfer_qwen3_6_27b_rtx5090 STATIC
  impl/package.cpp
  impl/diagnostic/activation_dump_access.cpp
  impl/frontend/chat_template.cpp
  impl/frontend/frontend.cpp
  impl/frontend/processor.cpp
  impl/frontend/tokenizer.cpp
  impl/load/bindings.cpp
  impl/program/layouts.cpp
  impl/program/program.cpp
  impl/program/request_plan.cpp
  impl/schedule/mtp.cpp
  impl/schedule/text_context.cpp
  impl/schedule/text_prefill.cpp
  impl/schedule/vision.cpp
  impl/schedule/vision_context.cpp
  impl/schedule/vision_control.cpp
  impl/state/state_store.cpp)

ninfer_internal_includes(ninfer_qwen3_6_27b_rtx5090)
target_include_directories(ninfer_qwen3_6_27b_rtx5090
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/impl
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/export>)
ninfer_cuda_archive(ninfer_qwen3_6_27b_rtx5090)
target_link_libraries(ninfer_qwen3_6_27b_rtx5090
  PRIVATE ninfer_artifact ninfer_ops ninfer_text ninfer_media_decode)
