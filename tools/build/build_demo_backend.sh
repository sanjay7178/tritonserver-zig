#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
install_dir="${TRITON_INSTALL_DIR:-$PWD/core/build/install}"
output_dir="$PWD/.zig-cache/phase2/backends/minimal"
rapidjson_include="${RAPIDJSON_INCLUDE_DIR:-/usr/include}"
if [ ! -f "$rapidjson_include/rapidjson/document.h" ]; then
    rapidjson_include="${HOMEBREW_PREFIX:-/home/linuxbrew/.linuxbrew}/include"
fi
mkdir -p "$output_dir"
temporary_library=$(mktemp "$output_dir/.minimal-XXXXXX.so")
trap 'rm -f "$temporary_library"' EXIT HUP INT TERM
"${CXX:-c++}" -std=c++17 -fPIC -shared src/backend/examples/minimal/minimal.cc \
    -I "$install_dir/include" -I "$rapidjson_include" \
    -L "$install_dir/lib" -ltritonbackendutils -ltritonasyncworkqueue \
    -ltritoncommonerror -ltritonserver -pthread \
    -Wl,-z,defs -Wl,--version-script=src/backend/examples/minimal/libtriton_minimal.ldscript \
    -Wl,-rpath,"$install_dir/lib" -o "$temporary_library"
mv "$temporary_library" "$output_dir/libtriton_minimal.so"

# A separate test-only backend produces an asynchronous execution error without
# violating assumptions of the unmodified upstream minimal reference backend.
output_dir="$PWD/.zig-cache/phase2/backends/phase3_error"
mkdir -p "$output_dir"
temporary_library=$(mktemp "$output_dir/.phase3-error-XXXXXX.so")
"${CXX:-c++}" -std=c++17 -fPIC -shared tests/fixtures/backends/phase3_error/error.cc \
    -I "$install_dir/include" -L "$install_dir/lib" -ltritonserver \
    -Wl,-z,defs -Wl,-rpath,"$install_dir/lib" -o "$temporary_library"
mv "$temporary_library" "$output_dir/libtriton_phase3_error.so"
