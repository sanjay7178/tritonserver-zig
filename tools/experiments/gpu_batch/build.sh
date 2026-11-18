#!/bin/sh
set -eu
project_root=$(CDPATH= cd -- "$(dirname "$0")/../../.." && pwd)
source_dir="$project_root/tools/experiments/gpu_batch"
install_dir="$project_root/core/build/install"
output_dir="$project_root/.zig-cache/gpu-experiment/backends/gpu_batch"
mkdir -p "$output_dir"
temporary_library=$(mktemp "$output_dir/.gpu-batch-XXXXXX.so")
trap 'rm -f "$temporary_library"' EXIT HUP INT TERM
"${CXX:-c++}" -std=c++17 -O3 -fPIC -fvisibility=hidden -shared \
    -Wall -Wextra -Wno-unused-parameter "$source_dir/backend.cc" \
    -I "$install_dir/include" -L "$install_dir/lib" \
    -ltritonserver -lcublas -lcudart -pthread \
    -Wl,-z,defs -Wl,-rpath,"$install_dir/lib" \
    -o "$temporary_library"
mv "$temporary_library" "$output_dir/libtriton_gpu_batch.so"
printf '%s\n' "$output_dir/libtriton_gpu_batch.so"
