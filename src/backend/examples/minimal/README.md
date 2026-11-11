# Minimal CPU backend fixture

These two source files are copied unchanged from the upstream NVIDIA Triton
minimal backend example present in `examples/minimal/src` in this workspace.
They retain their original copyright and BSD license notices. The example
implements an INT32 pass-through backend for an input `IN0` and output `OUT0`,
each with configured dimensions `[4]`.

They are bundled here so the Phase 2 demo and tests do not depend on the
workspace's pre-existing untracked `examples` or `backend` directories.
`tools/build/build_demo_backend.sh` compiles them against the headers and libraries
in `core/build/install`; no upstream CMake fetching is performed.
