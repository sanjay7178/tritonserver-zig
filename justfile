# https://just.systems

default:
    @just --help

# Build the project (debug)
build:
    zig build

# Build in release mode (safe|fast|small)
release mode="fast":
    zig build -Doptimize=Release{{ capitalize(mode) }}

# Build and run the starter using a writable cache and local dependencies
run *args:
    ZIG_GLOBAL_CACHE_DIR="${ZIG_GLOBAL_CACHE_DIR:-/tmp/tritonserver-zig-global-cache}" \
      ZIG_LOCAL_PKG_DIR="${ZIG_LOCAL_PKG_DIR:-$PWD/zig-pkg}" \
      zig build run -- {{ args }}

# build the linker
# Configure and build only libtritonserver.so; skip Python wheel.
linker:
    /opt/cmake/bin/cmake \
      -S core -B core/build \
      -DCMAKE_INSTALL_PREFIX="$PWD/core/build/install" \
      -DTRITON_CORE_HEADERS_ONLY=OFF \
      -DTRITON_ENABLE_GPU=OFF \
      -DTRITON_ENABLE_METRICS=OFF \
      -DTRITON_ENABLE_METRICS_GPU=OFF \
      -DTRITON_ENABLE_METRICS_CPU=OFF \
      -DTRITON_ENABLE_STATS=OFF \
      -DCMAKE_PREFIX_PATH="/home/linuxbrew/.linuxbrew/opt/boost;$PWD/.deps/utf8-range-install"
 
    /opt/cmake/bin/cmake \
      --build core/build --target protobuf -j10

    /opt/cmake/bin/cmake \
      --build core/build --target triton-core -j10

    /opt/cmake/bin/cmake \
      --install core/build/triton-core

# Run tests
test:
    zig build test

# Remove build artifacts
clean:
    rm -rf zig-out .zig-cache
