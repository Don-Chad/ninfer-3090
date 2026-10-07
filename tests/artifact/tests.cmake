ninfer_add_test(ninfer_artifact_reader_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_reader.cpp"
  LIBRARIES ninfer_artifact)

ninfer_add_test(ninfer_artifact_permute_row_split_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_permute_row_split.cpp"
  LIBRARIES ninfer_artifact)

set_tests_properties(
  ninfer_artifact_permute_row_split_test
  PROPERTIES SKIP_RETURN_CODE 77)

# Standalone: writer_interop.py runs it as an executable path, and the symbol wrapping below must not
# reach any other test.
if(MSVC)
  # MSVC has no link-time symbol wrapping; the CUDA fault-injection cases are compiled out.
  ninfer_add_test(ninfer_artifact_materialization_test STANDALONE
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_materialization.cpp"
    LIBRARIES ninfer_artifact)
else()
  ninfer_add_test(ninfer_artifact_materialization_test STANDALONE
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_materialization.cpp" "${CMAKE_CURRENT_LIST_DIR}/materialization_cuda_errors.cpp"
    LIBRARIES ninfer_artifact)
  target_compile_definitions(ninfer_artifact_materialization_test PRIVATE NINFER_TEST_LINK_WRAP=1)
  target_link_options(ninfer_artifact_materialization_test PRIVATE
    "LINKER:--wrap=cudaMalloc"
    "LINKER:--wrap=cudaMallocHost"
    "LINKER:--wrap=cudaHostAlloc"
    "LINKER:--wrap=cudaFree"
    "LINKER:--wrap=cudaFreeHost"
    "LINKER:--wrap=cudaEventCreateWithFlags"
    "LINKER:--wrap=cudaEventRecord"
    "LINKER:--wrap=cudaMemcpyAsync"
    "LINKER:--wrap=cudaStreamSynchronize")
endif()

add_test(NAME ninfer_artifact_writer_interop_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/writer_interop.py"
    $<TARGET_FILE:ninfer_artifact_materialization_test>)

set_tests_properties(
  ninfer_artifact_writer_interop_test
  PROPERTIES SKIP_RETURN_CODE 77)

set_tests_properties(
  ninfer_artifact_materialization_test
  PROPERTIES SKIP_RETURN_CODE 77)
