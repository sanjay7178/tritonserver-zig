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
