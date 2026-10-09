add_library(ninfer_media_acquire STATIC
  media_acquire/acquire.cpp
)
ninfer_internal_includes(ninfer_media_acquire)
target_link_libraries(ninfer_media_acquire PRIVATE ${NINFER_CURL_TARGET})
if(WIN32)
  target_link_libraries(ninfer_media_acquire PRIVATE ws2_32)
endif()

add_library(ninfer_s3_object_store STATIC
  object_store/s3_object_store.cpp
)
ninfer_internal_includes(ninfer_s3_object_store)
target_include_directories(ninfer_s3_object_store PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(ninfer_s3_object_store PRIVATE ${NINFER_CURL_TARGET})
if(WIN32)
  target_link_libraries(ninfer_s3_object_store PRIVATE ws2_32)
endif()

add_library(ninfer_product_prompt_input STATIC
  prompt_input/prompt_input.cpp
)
ninfer_internal_includes(ninfer_product_prompt_input)
target_link_libraries(ninfer_product_prompt_input PRIVATE ninfer_media_acquire ninfer::json)

add_library(ninfer_product_logging STATIC
  logging/logging.cpp
  logging/pretty_format.cpp
  logging/startup_log.cpp
)
ninfer_internal_includes(ninfer_product_logging)
target_link_libraries(ninfer_product_logging PUBLIC spdlog::spdlog)

# The generated source is rewritten only when the version text changes, so the commit suffix
# following HEAD costs one tiny translation unit, not a rebuild of every consumer. The target
# runs on every build; a custom command alone would not notice a new commit.
set(NINFER_VERSION_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/version/version.cpp")
add_custom_target(ninfer_version_source
  COMMAND ${CMAKE_COMMAND}
    -DSRC_DIR=${PROJECT_SOURCE_DIR}
    -DIN=${CMAKE_CURRENT_SOURCE_DIR}/version/version.cpp.in
    -DOUT=${NINFER_VERSION_SOURCE}
    -P ${PROJECT_SOURCE_DIR}/cmake/NinferVersion.cmake
  BYPRODUCTS "${NINFER_VERSION_SOURCE}"
  VERBATIM)
add_library(ninfer_product_version STATIC "${NINFER_VERSION_SOURCE}")
add_dependencies(ninfer_product_version ninfer_version_source)
ninfer_internal_includes(ninfer_product_version)
