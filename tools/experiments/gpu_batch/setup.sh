#!/bin/sh
set -eu
project_root=$(CDPATH= cd -- "$(dirname "$0")/../../.." && pwd)
sh "$project_root/tools/experiments/gpu_batch/build.sh"
experiment="$project_root/.zig-cache/gpu-experiment"
model="$experiment/cublas-models/gpu_dense_mean"
mkdir -p "$model/1"
cat > "$model/config.pbtxt" <<'CONFIG'
name: "gpu_dense_mean"
backend: "gpu_batch"
max_batch_size: 64
input [{ name: "IN0" data_type: TYPE_FP32 dims: [64] }]
output [{ name: "OUT0" data_type: TYPE_FP32 dims: [64] }]
# The CPU-only Triton core transports host tensors. The backend independently
# selects CUDA device zero and executes every dense layer using cuBLAS.
instance_group [{ kind: KIND_CPU count: 1 }]
CONFIG
cat > "$experiment/cublas-workload.json" <<'JSON'
{
  "model": "gpu_dense_mean",
  "backend": "gpu_batch",
  "inputs": [{
    "name": "IN0", "datatype": "FP32", "shape": [1, 64],
    "data": [
      -32, -31, -30, -29, -28, -27, -26, -25,
      -24, -23, -22, -21, -20, -19, -18, -17,
      -16, -15, -14, -13, -12, -11, -10, -9,
      -8, -7, -6, -5, -4, -3, -2, -1,
      0, 1, 2, 3, 4, 5, 6, 7,
      8, 9, 10, 11, 12, 13, 14, 15,
      16, 17, 18, 19, 20, 21, 22, 23,
      24, 25, 26, 27, 28, 29, 30, 31
    ]
  }],
  "outputs": ["OUT0"],
  "max_batch_size": 64,
  "hidden_default": 4096,
  "layers_default": 16,
  "hidden_environment": "GPU_EXPERIMENT_HIDDEN",
  "layers_environment": "GPU_EXPERIMENT_LAYERS",
  "reference": "OUT0[b,j] = sum(IN0[b,0..64]) / 64",
  "execution": "FP32 pedantic cuBLAS SGEMM; 64 -> hidden -> repeated hidden layers -> 64",
  "weights": "Dense uniform weights equal to one divided by each layer's input width",
  "transport": "Triton KIND_CPU; explicit CUDA device zero allocation, execution and transfers",
  "native_dynamic_batching": false,
  "synthetic_workload": true
}
JSON
printf 'Model repository: %s\nBackend directory: %s\n' \
    "$experiment/cublas-models" "$experiment/backends"
