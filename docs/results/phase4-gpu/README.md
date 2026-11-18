# Phase 4 GPU evidence

These archived results support the synthetic-workload measurements in
[the scheduler documentation](../../scheduler.md#optional-gpu-experiments).
That section includes hardware, limitations and commands to regenerate the
models, verify numerical results and repeat the benchmarks.

- [summary.json](summary.json) records hardware, the three measured cases and
  qualifications. The full measurements are [ONNX batch 8](onnx-batch8.json),
  [ONNX batch 32](onnx-batch32.json) and [cuBLAS batch 8](cublas-batch8.json).
- [desktop-baseline.json](desktop-baseline.json) retains the device-wide busy
  samples before the experiments. Desktop activity contributes to GPU
  utilization; these percentages are neither process attribution nor occupancy.
- [cuda-verification.json](cuda-verification.json) records the ONNX initializer's
  exact-output check and disabled CPU fallback/TF32 settings.
  [gpu-kernel-evidence.json](gpu-kernel-evidence.json) extracts the eight actual
  ORT kernel records: four MatMul and four Relu, all on `CUDAExecutionProvider`.
  The excerpt retains operation names, providers, durations and tensor shapes.
  It comes from a batch-one initialization check; those diagnostic durations
  are separate from the warmed benchmark measurements.
- [ONNX correctness](onnx-scheduler-correctness.log) and
  [cuBLAS correctness](cublas-scheduler-correctness.log) record shape/value checks
  for one sample, eight distinct requests combined into batch eight, and mixed
  request sizes 1/3/4 combined into batch eight.
  [cuda-cublas-check.json](cuda-cublas-check.json) retains a separate small
  cuBLAS matrix-multiplication reference check.

Numerical measurement values are unchanged. Host-specific absolute paths were
replaced with repository-relative references, and the CUDA profile references
point to the retained kernel excerpt. PID, thread, absolute timestamp and large
thread-scheduling metadata are omitted; the source profile location uses
placeholders. Original full profiles, generated weights, Python dependencies
and local run artifacts remain ignored under `.zig-cache/gpu-experiment`.

The measured-report `phase4_performance_target_met` field remains `null`.
Device-level observed thresholds and separate CUDA/numerical evidence qualify
only these synthetic runs. Baseline-first ordering, shared desktop graphics and
uncontrolled thermal/clock state limit broader performance conclusions.
