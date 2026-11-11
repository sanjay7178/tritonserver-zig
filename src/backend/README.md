# Backend boundary

This directory corresponds to NVIDIA Triton's
[backend repository](https://github.com/triton-inference-server/backend):
model execution implementations and examples for backend authors. It is
separate from `src/core` (server/model management) and `src/server` (the process
and command-line frontend).

Phase 2 uses the licensed upstream C++ minimal backend in `examples/minimal`
solely to validate real model loading and metadata. It is a reference dependency,
not a Zig-native backend implementation. Build it with `zig build demo-backend`
or run `just demo`; the shared library is generated under ignored `.zig-cache`.

Phase 5 adds the Zig-native pass-through plugin in `examples/zig_passthrough.zig`
and the public `t.backend` authoring helpers. Build it with `just backend`, or
load it in real Triton with `just backend-demo`. The optional CUDA copy module
is compiled only with `-Dbackend-cuda=true`; `just backend-cuda-test` verifies it
on local hardware. See [usage and ownership](../../docs/backend.md).

`src/core` loads plugins through Triton's existing C API; it never imports plugin source.
The pre-existing root-level `backend/` checkout remains an untouched upstream
C/C++ dependency, distinct from this project's source organization.
