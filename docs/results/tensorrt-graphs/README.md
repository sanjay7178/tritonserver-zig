# RTX 4050 TensorRT validation

The repository's CPU-only Triton installation returned
`unable to find backend library for backend 'tensorrt'` when the Zig client loaded
the B8 plan. The GPU-enabled Triton 2.69.0 distribution loaded the same plan
and executed it through the Zig wrapper. The driver was 595.104.02, with CUDA
13.2 and TensorRT 10.16.1.11, on an NVIDIA GeForce RTX 4050 Laptop GPU with
6,141 MiB of memory.

An untimed probe observed 12 successful `cudaGraphLaunch` calls from the native
TensorRT backend for 12 requests with graphs on, at both batch sizes. It
observed zero calls for 12 requests at each batch size with graphs off. With
graphs on, the standalone TensorRT harness also recorded exactly 20,000
successful replays per batch size in the timed loop. The maximum absolute
error across the engine and embedded reference checks was below 1.2 × 10⁻⁶.

## Throughput and host latency

Each timed case used 1,000 warmup requests and 20,000 measured requests, with
one request in flight. Latencies are in microseconds.

| Path | Batch | Graphs | Requests/s | Samples/s | p50 | p95 | p99 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| TensorRT engine | 1 | Off | 31,726 | 31,726 | 30.27 | 34.25 | 35.39 |
| TensorRT engine | 1 | On | 36,397 | 36,397 | 26.15 | 29.77 | 30.45 |
| TensorRT engine | 8 | Off | 31,355 | 250,841 | 30.67 | 34.34 | 36.20 |
| TensorRT engine | 8 | On | 36,251 | 290,011 | 26.18 | 30.03 | 32.39 |
| Embedded Zig | 1 | Off | 9,365 | 9,365 | 98.61 | 163.06 | 248.39 |
| Embedded Zig | 1 | On | 10,920 | 10,920 | 85.66 | 129.20 | 178.09 |
| Embedded Zig | 8 | Off | 8,268 | 66,141 | 109.97 | 189.12 | 283.28 |
| Embedded Zig | 8 | On | 9,158 | 73,261 | 102.76 | 156.63 | 220.08 |

The [full JSON report](rtx4050.json) contains engine and library SHA-256
digests, software versions, host submission timings, CUDA event timings for
compute and pinned transfers, Triton phase means, and sampled device
utilization and memory. The [benchmark instructions](../../../tools/experiments/tensorrt_graphs/README.md)
describe how to reproduce the workload and interpret the measurements.

## Independent CUDA trace

Nsight Systems 2026.5.2 recorded the embedded Zig client executing four warmup
requests and eight measured requests per case without the launch probe. The
CUDA runtime returned 12 graph launches in each graphs-on trace, and CUPTI
recorded 12 graph executions on the GPU. Each trace contained 12 host-to-device
and 12 device-to-host copies of the expected tensor size.

| Batch | Graphs | GPU graph executions | Device compute, µs/request | H2D, µs | D2H, µs |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Off | 0 | 25.843 | 0.413 | 1.056 |
| 1 | On | 12 | 25.336 | 0.416 | 0.963 |
| 8 | Off | 0 | 25.875 | 2.040 | 2.621 |
| 8 | On | 12 | 25.357 | 2.405 | 2.720 |

Graphs-off compute is the sum of the three standalone kernel durations per
request. Graphs-on compute is the duration of the graph activity, which can
include gaps between nodes. The transfer columns average only request-sized
copies. The throughput and latency table above comes from the unprofiled runs.

With the setup environment loaded and a completed benchmark, set `result` to
its output directory. This example captures the two batch-8 cases.

```sh
for graphs in 0 1; do
    nsys profile --trace=cuda --sample=none --cpuctxsw=none \
        --cuda-graph-trace=graph --export=sqlite --output="$result/nsys-b8-g$graphs" \
        "$result/zig/bin/tensorrt-bench" "$result/b8-graphs-$graphs/models" \
        "$TRT_PREFIX/backends" bench "$result/batch-8.bin" 8 512 4 8 0
    nsys stats --report cuda_api_sum,cuda_gpu_trace "$result/nsys-b8-g$graphs.sqlite"
done
```

The SQLite export contains the graphs-on GPU records in
`CUPTI_ACTIVITY_KIND_GRAPH_TRACE`, individual kernels in
`CUPTI_ACTIVITY_KIND_KERNEL`, and the request-sized transfers in
`CUPTI_ACTIVITY_KIND_MEMCPY`. Activity timestamps are in nanoseconds.
