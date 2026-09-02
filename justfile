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

# Run tests
test:
    zig build test

# Remove build artifacts
clean:
    rm -rf zig-out .zig-cache
