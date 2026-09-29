# Vendored XGrammar

NInfer vendors the C++ core of XGrammar for structured (JSON) output:

- upstream: <https://github.com/mlc-ai/xgrammar>
- tag: `v0.2.5.post1`
- commit: `4546988d1cf51670a10eafb8795d37e4a2f0d833`
- license: Apache-2.0; see `LICENSE` and `NOTICE`

The committed `cpp/` and `include/` trees are unchanged upstream files, except that the Python
binding directory `cpp/tvm_ffi/` and upstream's `cpp/CMakeLists.txt` are omitted.
`3rdparty/picojson/picojson.h` is upstream's bundled copy (BSD-2-Clause, license text in the
header). `3rdparty/dlpack/include/dlpack/dlpack.h` is from dmlc/dlpack commit
`bbd2f4d32427e548797929af08cfe2a9cbb3cf12`, the revision of upstream's `3rdparty/dlpack`
submodule (Apache-2.0, `3rdparty/dlpack/LICENSE`). Python bindings, tests, the web/Swift
packages, cpptrace and googletest are not vendored.

The local `CMakeLists.txt` builds the static library `ninfer_xgrammar` without upstream's
project-wide compiler flags. NInfer does not fetch network content during configuration, so an
offline (for example Nix) build needs nothing beyond this tree.

To update: replace `cpp/` (minus `tvm_ffi/` and `CMakeLists.txt`), `include/xgrammar/`,
`picojson.h`, and `dlpack.h` from the new tag and its dlpack submodule revision, then update the
revisions above.
