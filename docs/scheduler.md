# Phase 4: software batching and scheduling

The scheduler sits above the embedded Triton C API. Concurrent callers submit
small requests to a bounded queue; worker threads combine compatible inputs
along the leading batch dimension, send one inference request to Triton, and
split its output into independently owned responses. Coalescing requires a
batching model that returns one output sample for each input sample. Nonbatching
models can use the worker queue with a batch cap of 1 and are dispatched alone.

This implements the Phase 4 scheduling mechanism. CPU fixtures verify batching,
result routing, errors and shutdown. The roadmap's performance target also
requires improved throughput and measured GPU utilization above 80 percent.
CPU tests cannot establish that target. The optional hardware experiments below
record CUDA provenance, numerical checks and measurements for synthetic models.

## Run the CPU demonstration

```sh
just scheduler-demo
just scheduler-bench --duration-seconds 1 --warmup-seconds 0
```

The demonstration uses `tests/fixtures/scheduler_models/phase4_fixed`, the
test-only `phase4_batch` backend, and the existing CPU Triton runtime. It submits
16 distinguishable `INT32[1,4]` jobs before collecting their results. Every
`OUT0` must match its caller's input; `BATCH_INFO` reports the backend execution
ID, combined sample count, and number of native requests in that execution.
The demo also checks the scheduler's largest dispatched batch is greater than
one. This verifies actual coalescing rather than merely concurrent native calls.

`phase4_dynamic` accepts variable nonbatch dimensions for compatibility tests;
`phase4_nonbatch` and `phase4_malformed` exercise nonbatching behavior and
malformed output handling. These are test fixtures, not production models.

## API and ownership

```zig
const t = @import("tritonserver_zig");

var scheduler = try t.Scheduler.init(allocator, &server, "my_model", 1, .{
    .max_batch_size = 8,
    .max_wait_us = 5_000,
    .worker_count = 2,
    .queue_capacity = 1024,
});
defer scheduler.deinit();

const input = try t.InputTensor.fromSlice(i32, "IN0", &.{ 1, 4 }, &.{ 1, 2, 3, 4 });
var future = try scheduler.submit(allocator, &.{input}, .{ .outputs = &.{"OUT0"} });
defer future.deinit();
var response = try future.wait();
defer response.deinit();
const values = try (try response.output("OUT0")).asSlice(i32);
_ = values;

// Stop accepting work, finish accepted work, and join worker threads.
scheduler.shutdown();
const statistics = scheduler.stats();
_ = statistics;
```

`InputTensor` is a borrowed view. `submit` validates and copies names, shapes
and data before returning. It is safe to reuse the caller's input buffer after
a successful submission. A submitted handle owns its completion state; call
`deinit` even if `wait` fails. A successful `wait` transfers an independently
owned `InferenceResponse` to the caller. Release that response separately.
`wait` is single-use. A future remains usable after scheduler shutdown or
deinitialization. Deinitializing an unfinished future waits for its accepted
work; it does not cancel that work. Do not copy owning handles and free both
copies.

Keep the `Server` at a stable address and alive until `scheduler.shutdown()`
returns. Do not poll, stop or delete the server concurrently with scheduler
work. Shut down and deinitialize the scheduler before deleting the server.
`Server.on_error` can run concurrently on worker threads and must be thread-safe.

The scheduler always uses `std.heap.c_allocator` for its internal cross-thread
storage. The allocator passed to `submit` is used only by its single waiting
thread when cloning the returned response. Separate allocators per waiter are
valid; an allocator shared by concurrent waiters must be thread-safe. The
allocator passed to scheduler initialization owns the scheduler state and model
snapshot. The benchmark uses `std.heap.c_allocator` throughout. Treat the
scheduler as a shared owner and avoid overlapping `deinit` with caller operations.

The model name and version belong to the scheduler. `max_batch_size = null`
uses the model's configured batch limit. An explicit cap must fit that limit.
Version `-1` selects the highest READY version at initialization and pins it for
the scheduler's lifetime. Poll/reload the repository after shutting it down.
The scheduler validates concrete shapes and fixed-width datatypes against model
metadata. Compatible jobs have matching tensor names, datatypes, nonbatch
dimensions and output selection. Incompatible inputs are not concatenated.
`max_wait_us` bounds how long an incomplete batch waits for compatible work;
a full batch can dispatch immediately. It is a queue batching delay, not an
end-to-end latency deadline or a cancellation timeout.
Zero wait dispatches each submission immediately without collecting a window.

Queue capacity limits accepted queued and executing logical requests; completed
responses retained by callers do not consume that capacity. Submission can fail
immediately when the queue is full or shutdown has begun. Submitted work reports native
inference errors through `wait`. Unsupported configurations and invalid output
shapes fail explicitly rather than yielding partial or misrouted tensors.

The supported scope is fixed-width CPU input/output buffers and ordinary
one-to-one models, with a leading batch dimension for coalescing. GPU models may consume these
host buffers through Triton's backend, but device-resident tensors, ragged
batching, sequence batching, decoupled responses and variable-length `BYTES`
tensors are outside this implementation. Models with native dynamic batching
enabled are rejected, so there is a single batching owner.
Shape tensors, optional inputs, generated batch inputs/outputs and batching
models without inputs are also rejected because ordinary concatenation cannot
preserve their semantics.

## Benchmark comparison

```sh
just scheduler-bench
```

The benchmark builds in ReleaseFast and compares two cases against the same
live server and workload, with the same number of workers and closed-loop
clients:

| Setting | Baseline | Batched default |
| --- | --- | --- |
| Maximum batch samples | 1 | 8 |
| Batch wait | 0 microseconds | 5,000 microseconds |
| Worker threads | 2 | 2 |
| Concurrent clients | 32 | 32 |
| Warmup per case | 2 seconds | 2 seconds |
| Measurement per case | 10 seconds | 10 seconds |

Each client has at most one outstanding request and submits again after its
previous response completes. Both cases use one-sample logical requests. A
shared start gate excludes thread creation from measurement; the elapsed time
includes draining the final in-flight requests. Warmup requests are excluded
from measured counters and latency samples. The baseline runs first; repeated
runs and alternating order in an external harness help assess thermal and
ordering effects for serious performance comparisons.

One JSON report is written to stdout. Triton and diagnostic logs use stderr.
It contains completed requests/samples, samples per second, end-to-end
nearest-rank latency p50/p95/p99 in microseconds, mean queue latency, dispatch
count, mean dispatched batch size, and the batched/baseline throughput ratio.
End-to-end latency starts before submission and includes validation/copies,
queueing, native inference and response creation. Queue latency uses the
scheduler's aggregate enqueue-to-worker-dispatch time per accepted request;
it excludes inference. These scopes must not be compared as equivalent timers.

`throughput_improved` is an observation from that run, not a test assertion.
A small pass-through CPU model can be slower with batching: copying, queueing,
allocation and delay may dominate its work. CI checks successful execution and
valid results; it does not assert a speedup. Short smoke runs are particularly
unsuitable for performance claims.

Options use separate arguments:

```text
--model-repository PATH    --backend-directory PATH
--model NAME               --version 1
--duration-seconds 10      --warmup-seconds 2
--concurrency 32           --workers 2
--max-batch-size 8         --max-wait-us 5000
--workload workload.json   --gpu-index 0
```

The default workload is `IN0: INT32[1,4] = [1,2,3,4]`, requesting `OUT0`.
Use a workload file to benchmark another model:

```json
{
  "inputs": [
    {"name": "IN0", "datatype": "FP32", "shape": [1, 4], "data": [1, 2, 3, 4]}
  ],
  "outputs": ["OUT0"]
}
```

Shapes must be fully concrete, positive and include a leading batch dimension
of 1; `data` is a flat row-major JSON array matching the entire tensor shape.
Multiple inputs are supported. Accepted datatype names are `BOOL`, `UINT8`,
`UINT16`, `UINT32`, `UINT64`, `INT8`, `INT16`, `INT32`, `INT64`, `FP16`, `FP32`
and `FP64`; values must fit their datatype. Boolean data uses JSON booleans.
`BYTES` and `BF16` workload encoding are unsupported and fail explicitly.
An empty `outputs` array requests the model's ordinary complete output set.
The file size is limited to 16 MiB. Models and requested output names still
undergo scheduler validation.

## Measure a GPU workload

Use a compatible GPU backend, a batched model repository and a representative
workload. Native `KIND_GPU` models require a GPU-enabled Triton core. Alternatively,
a `KIND_CPU` Python/custom backend can transport host tensors and explicitly run
CUDA internally, as the optional experiments below do. The CPU pass-through
fixture does not exercise a GPU. The build accepts
`-Dtriton-prefix=/path/to/triton/install` to select an external installation.
For example, build and install the executables against that installation, then
run the benchmark:

```sh
zig build -Doptimize=ReleaseFast -Dtriton-prefix=/path/to/triton/install
./zig-out/bin/scheduler-bench \
  --model-repository /path/to/gpu-models \
  --backend-directory /path/to/triton/backends \
  --model my_model --version 1 --workload workload.json \
  --duration-seconds 30 --warmup-seconds 5 --gpu-index 0
```

`--gpu-index` explicitly requests `nvidia-smi` sampling for the selected device
during each measurement. The benchmark checks availability before starting,
queries roughly every 200 ms, and reports the arithmetic mean and sample count.
A missing executable, unsupported query, nonexistent device, nonzero command
exit or invalid response causes a visible error. Commands have a three-second
timeout. Device utilization includes other processes and is sampled rather
than time-weighted; isolate the GPU and retain raw run reports when validating
performance. GPU sampling itself adds some overhead to both cases.

[NVIDIA defines this utilization metric](https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization)
as the portion of a trailing sampling window with GPU kernels active. Depending
on the device, that window spans roughly one-sixth of a second to one second.
Polling every 200 ms can therefore repeat or overlap hardware sampling windows;
the first query can include activity preceding measurement. The percentage
does not measure SM occupancy or prove that the benchmark process used the GPU.
Desktop graphics and unrelated GPU work also contribute. Do not subtract an
idle desktop percentage and treat the difference as process utilization.

Without GPU sampling, GPU fields and `observed_gpu_thresholds_met` are `null`.
With sampling, `observed_gpu_thresholds_met` is true only when the measured
throughput ratio is above 1 and the batched case's measured mean device
utilization is above 80 percent. This is a device-level observation. It does
not verify the model's CUDA execution or its numerical outputs.

The legacy `phase4_performance_target_met` field remains `null`, even when both
observed thresholds pass. `performance_target_qualification_requires` explains
that a qualified result additionally needs GPU execution provenance and output
correctness. Record backend/provider placement or a GPU execution profile,
compare real outputs against a reference using distinct inputs, and repeat
sufficiently long, representative runs with matched workers and concurrency
before claiming the roadmap's GPU performance target is met. Keep that evidence
with the experiment report; this generic benchmark alone does not establish it.

## Optional GPU experiments

Hardware validation on 2026-11-11 used an NVIDIA GeForce RTX 4050 Laptop GPU
(6,141 MiB VRAM), driver 595.104.02, the installed CUDA 12 runtime/cuBLAS 12,
and Zig ReleaseFast. `nvidia-smi` reports driver support for CUDA 13.2; that is
not the toolkit version used by these experiments. No power, clock or fan
settings were changed.

Two synthetic dense workloads exercised actual CUDA computation through the
existing CPU-only Triton core. Their `KIND_CPU` configuration describes host
tensor transport, while the optional backends explicitly execute on the GPU:

- ONNX Runtime 1.23.2: four MatMul/Relu layers with shared FP32 8192-by-8192
  weights. ORT profiling recorded all four MatMul and four Relu nodes on
  `CUDAExecutionProvider`, with CPU fallback and TF32 disabled. Uniform weights
  make the positive-input reference an independently computable row mean.
- cuBLAS: FP32 64-to-4096 projection, 16 hidden dense layers, and reduction to
  64 outputs, using pedantic math. Every CUDA/cuBLAS status is checked; there
  is no CPU compute fallback. Each output equals its input row's mean.

Numerical checks passed for a single sample, eight distinct one-sample requests
merged into one batch, and mixed request sizes 1/3/4 merged into one batch.
Both output shapes and every returned value were checked after Zig partitioned
the results. The ONNX profile and these distinct-output checks provide evidence
separate from device utilization.

Each measured case used one scheduler worker and one model instance, three
seconds of warmup and 15 seconds of measurement per baseline/batched case.
The baseline used batch limit 1/wait 0; the batched case used a 5 ms window.
Workers, concurrency, model and input workload matched within each row.

The [archived measurements and CUDA/correctness evidence](results/phase4-gpu/README.md)
retain the reports behind this table.

| Workload | Concurrency | Batch limit | Baseline samples/s | Batched samples/s | Ratio | Batched device busy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNX dense | 16 | 8 | 138.62 | 1,077.39 | 7.77x | 92.97% |
| ONNX dense | 64 | 32 | 148.82 | 3,431.90 | 23.06x | 73.77% |
| cuBLAS dense | 16 | 8 | 158.30 | 1,249.75 | 7.89x | 97.16% |

The two batch-eight cases observed the roadmap's throughput and device-busy
thresholds with accompanying CUDA/numerical evidence. Batch 32 increased
throughput further while reducing the device-busy percentage. Its concurrency
also differs, so the rows are not a controlled batch-size-only comparison.
These results describe these synthetic models, not trained-model accuracy or
a production performance guarantee.

Desktop graphics shared the device: the pre-test baseline averaged 23.3% busy,
and process monitoring also recorded compositor activity during inference.
Baseline always ran first; ordering, temperature and clocks were not controlled.
A spot reading reached 87 C during cuBLAS testing. GPU polling covers the nominal
measurement interval while throughput also includes final request draining.
These effects limit precise comparisons; retain raw reports and repeat on an
otherwise idle GPU for a controlled performance study.

Generated weights, dependencies, full CUDA profiles and local run artifacts
remain ignored under `.zig-cache/gpu-experiment`. A small sanitized
[evidence bundle](results/phase4-gpu/README.md) retains measured JSON reports,
correctness logs and the eight CUDA kernel records; its
[summary](results/phase4-gpu/summary.json) records the combined qualifications.
The excluded initial installed-build timing is not part of the table.
GPU experiments run on the local development host. GitHub CI is CPU-only; it
does not request a self-hosted runner or execute CUDA workloads.

For the cuBLAS workload, with CUDA development libraries already installed:

```sh
sh tools/experiments/gpu_batch/setup.sh
zig build gpu-verify -Doptimize=ReleaseFast
zig build scheduler-bench -Doptimize=ReleaseFast -- \
  --model-repository .zig-cache/gpu-experiment/cublas-models \
  --backend-directory .zig-cache/gpu-experiment/backends \
  --model gpu_dense_mean --workload .zig-cache/gpu-experiment/cublas-workload.json \
  --workers 1 --concurrency 16 --max-batch-size 8 \
  --warmup-seconds 3 --duration-seconds 15 --gpu-index 0
```

`GPU_EXPERIMENT_HIDDEN` and `GPU_EXPERIMENT_LAYERS` configure the synthetic
cuBLAS workload when the server starts; defaults are 4096 and 16. They change
the workload and must be kept identical within a comparison.

For ONNX, use a Python 3.12 environment with NumPy and `onnxruntime-gpu`. On the
tested machine this is the existing Miniforge environment. Generator-only ONNX
dependencies are installed into the ignored experiment directory:

```sh
"$HOME/miniforge3/bin/python3" tools/experiments/onnx_gpu/setup.py --install-deps
"$HOME/miniforge3/bin/python3" tools/experiments/onnx_gpu/setup.py --verify-only
zig build gpu-verify -Doptimize=ReleaseFast -- \
  --model-repository .zig-cache/gpu-experiment/models \
  --backend-directory core/build/install/backends --model gpu_mlp --width 8192
zig build scheduler-bench -Doptimize=ReleaseFast -- \
  --model-repository .zig-cache/gpu-experiment/models \
  --backend-directory core/build/install/backends \
  --model gpu_mlp --workload .zig-cache/gpu-experiment/workload.json \
  --workers 1 --concurrency 16 --max-batch-size 8 \
  --warmup-seconds 3 --duration-seconds 15 --gpu-index 0
```

This route also requires the compatible Triton Python backend and Python 3.12
stub already present in the tested installation. Its generated configuration
records package paths and runtime versions. The optional cuBLAS path avoids
Python/ONNX dependencies. Neither path adds direct device-buffer support to the
Zig API or enables native `KIND_GPU` support in the CPU-only Triton core.
