# Native TensorRT CUDA Graph benchmark

The Zig client embeds GPU-enabled Triton and loads its native `tensorrt` backend.
Each run builds one FP32 TensorRT plan for the selected GPU. The workload has
three 512 × 512 matrix multiplications and two ReLU layers by default. It keeps
the tensor width fixed and exercises explicit request batches of 1 and 8 using
one optimization profile. For each batch it compares ordinary enqueue and
CUDA Graph replay using the same serialized plan. A NumPy reference checks two
distinct input fixtures in each mode.

## Reproduce

Use x86_64 Linux, an NVIDIA GPU and driver that support CUDA 13.2, the repository's
pinned Zig toolchain, Python 3.12 via `uv`, and a C compiler. The setup needs
`curl`, `unzip`, `sha256sum`, `dpkg-deb`, and several GB of disk space for the
release archive, Python wheels, and extracted files. It installs Triton 2.69.0,
TensorRT 10.16.1.11 and their required libraries into a directory you own.
[NVIDIA Triton 2.69.0 release](https://github.com/triton-inference-server/server/releases/tag/v2.69.0)

From the repository root, run these commands in Bash.

```sh
tools/experiments/tensorrt_graphs/setup.sh "$PWD/.deps/tensorrt-graphs"
source .deps/tensorrt-graphs/env.sh
"$TRT_PYTHON" -B tools/experiments/tensorrt_graphs/benchmark.py \
    --triton-prefix "$TRT_PREFIX" --gpu 0 \
    --output-dir "$PWD/.deps/tensorrt-results"
```

The installer accepts `TRITON_DOWNLOAD_CACHE` to reuse verified downloads from
another installation. With an existing compatible GPU Triton distribution,
provide its prefix, the matching CUDA and TensorRT libraries in
`LD_LIBRARY_PATH`, and a Python environment with NumPy, TensorRT and CUDA
bindings. The prefix must include `include/`, `lib/libtritonserver.so` and
`backends/tensorrt/libtriton_tensorrt.so`. A distribution that uses `lib64/` can
provide a `lib` symlink as the installer does. The core needs model statistics,
and the backend must support CUDA Graph capture.

The output directory receives `report.json`, the plan, model configs, fixtures,
and per-case logs. On another GPU, rerun the command to build a compatible plan
from the same weights, seed and shapes. Keep the default width and batch sizes
for a comparable workload. Use an otherwise idle GPU when comparing timing or
device utilization across runs.

Use `--prepare-only` to generate the plan, input fixtures, graph probe, and
model repositories without building or running the embedded timing client.
The resulting directory also supplies the [manual Zig GPU tests](../../../tests/fixtures/tensorrt/README.md)
and the [Python gRPC benchmark](../../../docs/grpc.md#benchmark-tensorrt-over-loopback).

## Measurements

- **Engine only.** TensorRT runs on one CUDA stream with device-resident input
  and output. Host latency covers enqueue or replay through stream
  synchronization. A separate pass uses CUDA events to time the inference
  interval. The interval can include gaps between operations. Isolated copies
  between pinned host and device buffers use separate CUDA events and appear as
  H2D and D2H. These copies are outside the engine throughput and host latency.
  [NVIDIA TensorRT benchmarking guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html)
- **Embedded Zig.** Host latency covers request construction, input copy,
  submission, response and release callbacks, and request deletion. Throughput
  is measured over the full timed loop, including response cleanup. The Triton
  C API provides mean queue, input, inference and output phase timings. The
  backend input/output phases can include host processing and synchronization.
  [Triton timing definitions](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/metrics.html)
- **Replay proof.** A separate run forwards `cudaGraphLaunch` to the CUDA
  runtime and counts successful calls originating from `libtriton_tensorrt.so`.
  It requires zero in the graphs-off case and one per request in the graphs-on
  case. This catches fallback to ordinary enqueue even when graph capture
  succeeds. The timed runs use the uninstrumented backend.
  [Triton TensorRT backend implementation](https://github.com/triton-inference-server/tensorrt_backend/blob/r26.05/src/instance_state.cc)
- **Utilization and memory.** The sampler reads whole-device GPU utilization
  and memory with `nvidia-smi` during the timed loops. A short run can have no
  samples, in which case the corresponding values are null. Peak memory is the
  highest sampled value. The report includes pre-run memory for context.
- **Batching.** Each request carries a full batch, with one request in flight on
  one GPU instance. The report gives both requests/s and samples/s. The results
  cover explicit batching and in-process inference.

See the [RTX 4050 validation run](../../../docs/results/tensorrt-graphs/README.md)
for the tested native stack and recorded measurements.
