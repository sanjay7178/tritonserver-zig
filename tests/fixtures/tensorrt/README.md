# Manual TensorRT GPU validation

These Zig tests use a Linux NVIDIA GPU, a GPU-enabled Triton prefix with
model statistics enabled, and a TensorRT plan built for the selected device.
Load the runtime environment for that prefix, then pass its installed libraries
and an existing fixture directory.

```sh
zig build tensorrt-test -Doptimize=ReleaseSafe \
  -Dtriton-prefix="$TRT_PREFIX" \
  -Dtensorrt-test-artifacts=/path/to/tensorrt-artifacts
```

The fixture layout matches a prepared benchmark run with width 512 and
explicit batches 1 and 8.

```text
tensorrt-artifacts/
  batch-1.bin
  batch-8.bin
  b1-graphs-0/models/bench/config.pbtxt
  b1-graphs-0/models/bench/1/model.plan
  b1-graphs-1/models/bench/config.pbtxt
  b1-graphs-1/models/bench/1/model.plan
  b8-graphs-0/models/bench/config.pbtxt
  b8-graphs-0/models/bench/1/model.plan
  b8-graphs-1/models/bench/config.pbtxt
  b8-graphs-1/models/bench/1/model.plan
```

Each `batch-{N}.bin` contains two FP32 input tensors followed by their two
reference output tensors. Each tensor has shape `[N, 512]` and uses little-endian
row-major data. The model uses `IN0` and `OUT0` as the TensorRT binding names.
Use the same plan for all four repositories and select graph capture with each
repository's Triton configuration. A prepared fixture can use symlinks to
share the plan.

The test binary verifies output shapes and values, Triton execution counters,
and successful `cudaGraphLaunch` calls from the TensorRT backend. It expects
four launches per graph-enabled case and zero per graph-disabled case. The
tests use Zig's test runner and run manually.
