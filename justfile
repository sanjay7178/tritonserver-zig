# https://just.systems

default:
    @just --help

# Build the project (debug)
build:
    zig build

# Build in release mode (safe|fast|small)
release mode="fast":
    zig build -Doptimize=Release{{ capitalize(mode) }}

# Run the app
run *args:
    zig build run -- {{ args }}

# build the linker
# Configure and build only libtritonserver.so; skip Python wheel.
linker:
    /home/linuxbrew/.linuxbrew/bin/cmake \
      -S core -B core/build \
      -DCMAKE_INSTALL_PREFIX="$PWD/core/build/install" \
      -DTRITON_CORE_HEADERS_ONLY=OFF \
      -DTRITON_ENABLE_GPU=OFF \
      -DTRITON_ENABLE_METRICS=OFF \
      -DTRITON_ENABLE_METRICS_GPU=OFF \
      -DTRITON_ENABLE_METRICS_CPU=OFF \
      -DTRITON_ENABLE_STATS=OFF \
      -DCMAKE_PREFIX_PATH="/home/linuxbrew/.linuxbrew/opt/boost;$PWD/.deps/utf8-range-install"
 
    /home/linuxbrew/.linuxbrew/bin/cmake \
      --build core/build --target protobuf -j1

    /home/linuxbrew/.linuxbrew/bin/cmake \
      --build core/build --target triton-core -j1

    /home/linuxbrew/.linuxbrew/bin/cmake \
      --install core/build/triton-core

# Run tests
test:
    zig build test

# Remove build artifacts
clean:
    rm -rf zig-out .zig-cache
