target_sources(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/chat_template.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/frontend.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/processor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/resources.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/tokenizer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/vision/bindings.cpp)

target_include_directories(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/export)
