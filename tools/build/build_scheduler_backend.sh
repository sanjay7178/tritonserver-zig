#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
install_dir="${TRITON_INSTALL_DIR:-$PWD/core/build/install}"
output_dir="$PWD/.zig-cache/phase2/backends/phase4_batch"
mkdir -p "$output_dir"
temporary_library=$(mktemp "$output_dir/.phase4-batch-XXXXXX.so")
trap 'rm -f "$temporary_library"' EXIT HUP INT TERM
"${CXX:-c++}" -std=c++17 -pthread -fPIC -shared tests/fixtures/backends/phase4_batch/batch.cc \
    -I "$install_dir/include" -L "$install_dir/lib" -ltritonserver \
    -Wl,-z,defs -Wl,-rpath,"$install_dir/lib" -o "$temporary_library"
mv "$temporary_library" "$output_dir/libtriton_phase4_batch.so"
