add_library(ninfer_serve STATIC
  http_transport.cpp
  console_log.cpp
  request_validation.cpp
  openai_chat_request.cpp
  openai_chat_response.cpp
  openai_chat_http.cpp
  openai_common.cpp
  props.cpp
  openai_responses_request.cpp
  openai_responses_state.cpp
  openai_responses_response.cpp
  anthropic_messages_request.cpp
  anthropic_messages_response.cpp
  anthropic_messages_http.cpp
  openai_responses_store.cpp
  translate.cpp
  generation_service.cpp
  request_events.cpp
  operational_log.cpp
  request_log.cpp
  http_server.cpp
  load_report.cpp
  slots_report.cpp
  openai_responses_http.cpp
  serve_options.cpp
  serve_metrics.cpp
)
ninfer_internal_includes(ninfer_serve)
target_link_libraries(ninfer_serve
  PUBLIC ninfer_engine Threads::Threads ninfer::json ninfer::httplib
  PRIVATE ninfer_media_acquire ninfer_s3_object_store ninfer_product_logging ninfer_product_version ${NINFER_CUDART_TARGET})
