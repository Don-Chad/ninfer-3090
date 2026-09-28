# XGrammar compiles tokenizer-aware grammars and advances their token-level matchers for
# structured output. Only the C++ core is built, as a local static library: upstream's top-level
# project sets project-wide -Werror/-flto flags that must not reach NInfer's CUDA build.
file(GLOB NINFER_XGRAMMAR_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/support/*.cc")
add_library(ninfer_xgrammar STATIC ${NINFER_XGRAMMAR_SOURCES})
set_target_properties(ninfer_xgrammar PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(ninfer_xgrammar PUBLIC cxx_std_20)
target_compile_definitions(ninfer_xgrammar PUBLIC
  XGRAMMAR_ENABLE_CPPTRACE=0
  XGRAMMAR_ENABLE_INTERNAL_CHECK=0)
target_include_directories(ninfer_xgrammar
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  SYSTEM PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/picojson"
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dlpack/include")
target_link_libraries(ninfer_xgrammar PUBLIC Threads::Threads)
if(MSVC)
  # Third-party code: keep NInfer's warning policy out of it.
  target_compile_options(ninfer_xgrammar PRIVATE /W0)
else()
  target_compile_options(ninfer_xgrammar PRIVATE -w)
endif()
