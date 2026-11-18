#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 || ( "$2" != cpu && "$2" != gpu ) ]]; then
    echo "Usage: $0 TRITON_PREFIX cpu|gpu" >&2
    exit 2
fi

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
prefix=$(cd -- "$1" && pwd)
if [[ ! -f "$prefix/lib/libtritonserver.so" || ! -f "$prefix/include/triton/core/tritonserver.h" ]]; then
    echo "Expected a Triton core installation with lib/libtritonserver.so and include/" >&2
    exit 2
fi

cache=${TRITON_GRPC_CACHE:-"$repo/.deps/native-grpc"}
mkdir -p -- "$cache/sources"
cache=$(cd -- "$cache" && pwd)
cmake_bin=${CMAKE:-cmake}
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-2}

fetch_source() (
    local name=$1 commit=$2 destination temporary url
    destination="$cache/sources/$name"
    if [[ -d "$destination" && -f "$destination/.triton-source-pin" ]] &&
        [[ "$(cat -- "$destination/.triton-source-pin")" == "$commit" ]]; then
        return
    fi
    if [[ -e "$destination" ]]; then
        echo "Source cache has the wrong pin for $name. Use a fresh TRITON_GRPC_CACHE." >&2
        return 1
    fi

    temporary=$(mktemp -d "$cache/sources/$name.XXXXXX")
    trap 'rm -r -- "$temporary"' EXIT
    url="https://codeload.github.com/triton-inference-server/$name/tar.gz/$commit"
    curl --fail --silent --show-error --location --retry 3 "$url" |
        tar -xz --strip-components=1 -C "$temporary"
    printf '%s\n' "$commit" > "$temporary/.triton-source-pin"
    mv -T -- "$temporary" "$destination"
    trap - EXIT
)

fetch_source server 34ca83e780bff7829b33d922fa9d16a0628e050c
fetch_source common ecb874c3126f9f07810f4900d4e7b088fee5eb8c
fetch_source third_party 4d8c8cec032bf696312e2c8dcc2ee077c05f0175

deps="$cache/third-party"
"$cmake_bin" -S "$cache/sources/third_party" -B "$cache/third-party-build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DTRITON_THIRD_PARTY_INSTALL_PREFIX="$deps"
"$cmake_bin" --build "$cache/third-party-build" --target grpc

package_prefixes="$deps/grpc;$deps/protobuf;$deps/absl;$deps/c-ares;$deps/re2"
build="$cache/bridge-$2"
"$cmake_bin" -S "$repo/tools/build/grpc" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DCMAKE_PREFIX_PATH="$package_prefixes" \
    -DTRITONZIG_TRITON_PREFIX="$prefix" \
    -DTRITONZIG_SERVER_SOURCE="$cache/sources/server" \
    -DTRITONZIG_COMMON_SOURCE="$cache/sources/common"
"$cmake_bin" --build "$build"
"$cmake_bin" --install "$build"
