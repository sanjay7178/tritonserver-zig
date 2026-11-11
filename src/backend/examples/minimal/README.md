# Minimal CPU backend fixture

These files are vendored unchanged from NVIDIA Triton backend commit
`efad1152ed836d336bb04cc4095ffe026e780bb7`,
[`examples/backends/minimal/src/{minimal.cc,libtriton_minimal.ldscript}`](https://github.com/triton-inference-server/backend/tree/efad1152ed836d336bb04cc4095ffe026e780bb7/examples/backends/minimal/src).
They retain their original copyright and BSD license notices. The example
implements an INT32 pass-through backend for an input `IN0` and output `OUT0`,
each with configured dimensions `[4]`.

They are bundled here so the Phase 2 demo and tests do not depend on the
workspace's pre-existing untracked `examples` or `backend` directories.
`tools/build/build_demo_backend.sh` compiles them against the headers and libraries
in `core/build/install`; no upstream CMake fetching is performed.

Both files match that revision byte for byte. SHA-256:

| File | SHA-256 |
| --- | --- |
| `minimal.cc` | `7d2169f0e4d71f77fd72c97e46803cc7044b733f756437060eca9badc0517116` |
| `libtriton_minimal.ldscript` | `470fbb783f0c2e9d44effc8f4c58da33ca7cfbbdf9ea6454f70b371486ef9de0` |

When updating, compare **both** files against the same backend commit and update
this pin and these hashes together. The [upstream provenance index](../../../../docs/upstream-provenance.md)
maps this fixture to the library contracts and regression tests.
