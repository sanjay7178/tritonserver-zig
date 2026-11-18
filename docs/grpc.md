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

## Set up the build environment with Pixi

On Ubuntu x86-64, use Pixi 0.63.2 or newer, `dpkg-deb`, and a host NVIDIA driver
compatible with CUDA 13.2. From the repository root, open the locked environment
for the C/C++ compiler, CMake, Ninja, `uv`, and FFmpeg.

```sh
pixi shell --locked
```

In the Pixi shell, download the repository's pinned Zig toolchain and extract
the `libb64` development package into the checkout.

```sh
mkdir -p .deps
zig_archive=zig-x86_64-linux-0.17.0-dev.1936+5a625d5f3.tar.xz
curl --fail --location -o ".deps/$zig_archive" "https://zig.squirl.dev/$zig_archive" &&
    printf '%s  %s\n' 38620c06ca57b175f7dab8f65c8001f9a67b0ff1987c344d4e64ccb72b57174f \
        ".deps/$zig_archive" | sha256sum --check &&
    tar -xf ".deps/$zig_archive" -C .deps
export PATH="$PWD/.deps/${zig_archive%.tar.xz}:$PATH"
zig version

b64_archive=.deps/libb64-dev_1.2-5_amd64.deb
curl --fail --location -o "$b64_archive" \
    https://archive.ubuntu.com/ubuntu/pool/universe/libb/libb64/libb64-dev_1.2-5_amd64.deb &&
    printf '%s  %s\n' 410e036ad4e4ba12792c974dcabb0449f85ca6d5905626e5560542cdd87f163b \
        "$b64_archive" | sha256sum --check &&
    dpkg-deb -x "$b64_archive" .deps/libb64
export CMAKE_INCLUDE_PATH="$PWD/.deps/libb64/usr/include${CMAKE_INCLUDE_PATH:+:$CMAKE_INCLUDE_PATH}"
export CMAKE_LIBRARY_PATH="$PWD/.deps/libb64/usr/lib/x86_64-linux-gnu${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}"
```

Install the pinned NVIDIA runtime, build the Zig server, and extract the
matching stock server once per host. The runtime setup creates the Python
environment used by the benchmarks. In a later shell, source its `env.sh`
again to use the installed runtime.
Restrict `pkg-config` to the Pixi environment during the native build so gRPC
discovers libraries from the same environment as the compiler. The dedicated
build cache keeps the Pixi objects separate from builds using a host compiler.
The explicit Zig target selects its bundled glibc support at the same 2.28
baseline as the Pixi C/C++ sysroot.

```sh
tools/experiments/tensorrt_graphs/setup.sh "$PWD/.deps/tensorrt-graphs"
source .deps/tensorrt-graphs/env.sh

PKG_CONFIG_PATH= \
    PKG_CONFIG_LIBDIR="$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/share/pkgconfig" \
    TRITON_GRPC_CACHE="$PWD/.deps/native-grpc-pixi" \
    tools/build/build_grpc_frontend.sh "$TRT_PREFIX" gpu
zig build -Dgrpc=true -Dtarget=x86_64-linux-gnu.2.28 \
    -Dtriton-prefix="$TRT_PREFIX" -Doptimize=ReleaseSafe
unzip -j "${TRITON_DOWNLOAD_CACHE:-.deps/tensorrt-graphs/downloads}/tritonserver.zip" \
    tritonserver/bin/tritonserver -d .deps/stock
```

## Benchmark TensorRT over loopback

Use the GPU runtime and both server executables from the build steps above.
Choose a physical GPU index or UUID from `nvidia-smi -L` and a fresh
output directory. Build the plan on each GPU you benchmark.

```sh
gpu=0
run=.deps/bench-gpu0
uv pip install --python "$TRT_PYTHON" 'tritonclient[grpc]==2.69.0'
"$TRT_PYTHON" -B tools/experiments/tensorrt_graphs/benchmark.py \
    --prepare-only --gpu "$gpu" --triton-prefix "$TRT_PREFIX" --output-dir "$run/fixed"
"$TRT_PYTHON" -B tools/experiments/tensorrt_graphs/grpc_benchmark.py \
    --zig-server zig-out/bin/tritonserver_zig \
    --stock-server .deps/stock/tritonserver --triton-prefix "$TRT_PREFIX" \
    --gpu "$gpu" --artifacts "$run/fixed" --output-dir "$run/fixed-results"
```

The benchmark compares explicit batches of 1 and 8 with graphs off and on.
It keeps one request in flight by default. Use `--concurrency 4` or
`--concurrency 8` to exercise multiple fixed worker lanes on both servers.
It checks the plan and library digests against the prepared artifacts,
verifies output against the same NumPy reference, and
counts backend `cudaGraphLaunch` calls in separate proof runs. The latency
includes Python tensor marshalling, the loopback RPC and response decoding.
See the [RTX 4050 gRPC measurements](results/tensorrt-graphs/README.md#loopback-grpc)
for the recorded comparison.

The generated plans and references can also be passed to the
[manual Zig TensorRT tests](../tests/fixtures/tensorrt/README.md).

### Benchmark a YOLO video pipeline

Use the same GPU build, stock server, and `gpu` and `run` variables shown above.
Pixi supplies FFmpeg and FFprobe. Add the video dependencies to the benchmark's
Python environment.

```sh
uv pip install --python "$TRT_PYTHON" 'tritonclient[grpc]==2.69.0' \
    'onnx==1.17.0' 'Pillow==11.3.0'
"$TRT_PYTHON" -B tools/experiments/tensorrt_graphs/video_setup.py \
    --output-dir "$run/video" --gpu "$gpu"
"$TRT_PYTHON" -B tools/experiments/tensorrt_graphs/video_benchmark.py \
    --zig-server zig-out/bin/tritonserver_zig \
    --stock-server .deps/stock/tritonserver --triton-prefix "$TRT_PREFIX" \
    --gpu "$gpu" --artifacts "$run/video" --output-dir "$run/video-results"
```

The setup uses published YOLOv8n weights and normalizes the pinned
1280×720 airport clip to a 1920×1080 frame input. It builds one TensorRT
plan per model on the selected GPU. Each frame batch makes two gRPC calls.
YOLOv8n detects objects at 640×640, then a deterministic regression fixture
predicts box offsets from up to four resized crops per frame.

Timed runs start with decoded RGB frames and include preprocessing, both
RPCs, class-aware NMS, and box refinement. The report gives frame throughput
and batch latency separately from per-model RPC timings. Paired trials switch
server order and verify matching detection counts and native Triton execution
counts. Separate proof runs compare detector tensors, regression outputs, and
refined boxes and count CUDA Graph launches across the two models.
[RTX 4050 video measurements](results/tensorrt-graphs/README.md#yolov8n-video-over-grpc)

## Upgrade

The build helper pins the NVIDIA r26.05 server, common, and third-party
sources together. Rebuild the adapter when upgrading those sources or the
installed core. Run the CPU client check and repeat the TensorRT comparison on
the selected GPU runtime. The bridge borrows the server handle and keeps the
native frontend alive while Triton drains accepted work.
[Native endpoint](https://github.com/triton-inference-server/server/tree/r26.05/src/grpc)
