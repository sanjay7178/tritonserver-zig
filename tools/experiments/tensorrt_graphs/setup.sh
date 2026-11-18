#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 || "$(uname -sm)" != "Linux x86_64" ]]; then
    echo "Usage on x86_64 Linux: $0 NEW_INSTALL_DIR" >&2
    exit 2
fi

mkdir -p -- "$1"
install_dir=$(cd -- "$1" && pwd)
if [[ -n "$(ls -A -- "$install_dir")" ]]; then
    echo "Install directory must be empty: $install_dir" >&2
    exit 2
fi
cache=${TRITON_DOWNLOAD_CACHE:-"$install_dir/downloads"}
mkdir -p -- "$cache"

download() {
    local name=$1 url=$2 checksum=$3 file
    file="$cache/$name"
    if [[ ! -f "$file" ]]; then
        curl --fail --location --retry 3 --output "$file.part" "$url"
        mv -- "$file.part" "$file"
    fi
    printf '%s  %s\n' "$checksum" "$file" | sha256sum --check --status
}

release=https://github.com/triton-inference-server/server/releases/download/v2.69.0
archive=tritonserver-2.69.0%2Bnv26.05-51748363-cu132-cp312-manylinux_2_28-x86_64.zip
download tritonserver.zip "$release/$archive" 6dbc21c837df452692eff8e595480218386cff3490567f1507a232a8f8a40294
unzip -q "$cache/tritonserver.zip" 'tritonserver/include/*' 'tritonserver/lib64/*' \
    'tritonserver/backends/tensorrt/*' 'tritonserver/TRITON_VERSION' 'tritonserver/README.md' \
    -d "$install_dir"
prefix="$install_dir/tritonserver"
ln -s lib64 "$prefix/lib"

nvidia=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64
cudart=cuda-cudart-13-2_13.2.86-1_amd64.deb
cupti=cuda-cupti-13-2_13.2.86-1_amd64.deb
dcgm=datacenter-gpu-manager-4-core_4.5.3-1_amd64.deb
download "$cudart" "$nvidia/$cudart" 6f5c230f993f4971eec9f91b756ba3827fa0ff1419eeba9862d02f88247c34bf
download "$cupti" "$nvidia/$cupti" 96d449894db64a6c575effbe2255c1a0249520436a321b8253a21a1cc00b4c04
download "$dcgm" "$nvidia/$dcgm" 75aa9a88085c7d2e985a9b0a442fb75ae8cc296cc84c644e604befdc092471d2
ssl=https://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.24_amd64.deb
download libssl1.1.deb "$ssl" 7cf39d70a639017d1dd7c8d36daa2258063608688e449fddf40ffdd46f992a78
for package in "$cudart" "$cupti" "$dcgm" libssl1.1.deb; do
    dpkg-deb -x "$cache/$package" "$install_dir/deps"
done

uv venv --python 3.12 "$install_dir/venv"
python="$install_dir/venv/bin/python"
uv pip install --python "$python" 'numpy==2.2.6' 'cuda-bindings==13.2.0' \
    'cuda-pathfinder==1.9.0' 'tensorrt-cu13==10.16.1.11'
site_packages=$("$python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
library_path="$prefix/lib:$install_dir/deps/usr/lib/x86_64-linux-gnu"
library_path="$library_path:$install_dir/deps/usr/local/cuda-13.2/targets/x86_64-linux/lib:$site_packages/tensorrt_libs"
{
    printf 'export TRT_PREFIX=%q\n' "$prefix"
    printf 'export TRT_PYTHON=%q\n' "$python"
    # Expand the prior library path when env.sh is sourced.
    # shellcheck disable=SC2016
    printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' "$library_path"
} > "$install_dir/env.sh"
printf 'Load this installation with source %q\n' "$install_dir/env.sh"
