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

Use `gpu` with the GPU Triton prefix from the
[TensorRT setup](../tools/experiments/tensorrt_graphs/README.md). Source its
environment before building and before starting the executable so the GPU core
and backend can locate their dependencies. The frontend hands inline tensors
to that core, which owns GPU execution.

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

## Benchmark TensorRT over loopback

Run the [in-process benchmark](../tools/experiments/tensorrt_graphs/README.md)
first to generate the plan, fixtures, and report. Build the gRPC executable with
that same GPU prefix. Extract the stock `tritonserver` executable from the same
NVIDIA release archive and load the GPU environment for both processes.

```sh
unzip -j /path/to/tritonserver.zip tritonserver/bin/tritonserver -d .deps/stock
python -B tools/experiments/tensorrt_graphs/grpc_benchmark.py \
    --zig-server zig-out/bin/tritonserver_zig \
    --stock-server .deps/stock/tritonserver --triton-prefix "$TRITON_PREFIX" \
    --artifacts /path/to/in-process-results --output-dir .deps/grpc-results
```

The benchmark runs one request in flight and compares explicit batches of 1
and 8 with graphs off and on. It checks the plan and library digests against
the in-process run, verifies output against the same NumPy reference, and
counts backend `cudaGraphLaunch` calls in separate proof runs. The latency
includes Python tensor marshalling, the loopback RPC and response decoding.
See the [RTX 4050 gRPC measurements](results/tensorrt-graphs/README.md#loopback-grpc)
for the three-path comparison.

## Upgrade

The build helper pins the NVIDIA r26.05 server, common, and third-party
sources together. Rebuild the adapter when upgrading those sources or the
installed core. Run the CPU client check and repeat the TensorRT comparison on
the selected GPU runtime. The bridge borrows the server handle and keeps the
native frontend alive while Triton drains accepted work.
[Native endpoint](https://github.com/triton-inference-server/server/tree/r26.05/src/grpc)
