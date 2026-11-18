# Native Triton gRPC

The Zig executable can start NVIDIA's gRPC frontend around the Triton core
it already owns. The frontend uses Triton's native request handling and
scheduler. In-process Zig callers can continue to use the Zig inference API.
The listener is opt-in. The example uses cleartext gRPC on loopback and inline
tensor data. HTTP, TLS configuration, and client shared memory are outside the
adapter's current scope.
[NVIDIA frontend bindings](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/customization_guide/tritonfrontend.html)

## Build and start

Use Linux with CMake 3.31.8 or newer, Ninja, a C++17 compiler, Git, curl, Python 3,
and the development packages for libb64, OpenSSL, RapidJSON, and zlib. Select a writable
Triton prefix containing `include/triton/core/tritonserver.h` and
`lib/libtritonserver.so`. The helper builds NVIDIA's pinned dependencies in
`.deps/native-grpc` and installs `libtritonzig_grpc.so` in the selected prefix.
`TRITON_GRPC_CACHE` changes the cache location. `CMAKE` selects a CMake executable,
and `CMAKE_BUILD_PARALLEL_LEVEL` controls parallelism.

```sh
tools/build/build_grpc_frontend.sh "$TRITON_PREFIX" cpu
zig build -Dgrpc=true -Dtriton-prefix="$TRITON_PREFIX" -Doptimize=ReleaseSafe
zig-out/bin/tritonserver_zig --grpc-listen 127.0.0.1:8001 \
    --model-repository /path/to/models --backend-directory /path/to/backends
```

Use `gpu` with a GPU-enabled Triton prefix. Load its runtime environment
before building and starting the executable so the GPU core and backend
can locate their dependencies. The frontend hands inline tensors to that
core, which owns GPU execution.
[NVIDIA Triton 2.69.0 release](https://github.com/triton-inference-server/server/releases/tag/v2.69.0)

## Validate

Install `tritonclient[grpc]==2.69.0` and NumPy in a Python environment. Build
the existing CPU test backend and run the integration check against the gRPC
executable linked with the CPU prefix.

```sh
zig build scheduler-backend -Dtriton-prefix="$TRITON_PREFIX"
python tests/integration/grpc_native.py \
    --server zig-out/bin/tritonserver_zig --triton-prefix "$TRITON_PREFIX" \
    --backend-directory "$PWD/.zig-cache/phase2/backends"
```

The check exercises NVIDIA's client, the passthrough tensor, port collision,
SIGTERM during inference, and the loaded Triton core path on Linux.

## Upgrade

The build helper pins the NVIDIA r26.05 server, common, and third-party
sources together. Rebuild the adapter when upgrading those sources or the
installed core. Run the CPU client check and verify inference on the selected
GPU runtime. The bridge borrows the server handle and keeps the
native frontend alive while Triton drains accepted work.
[Native endpoint](https://github.com/triton-inference-server/server/tree/r26.05/src/grpc)
