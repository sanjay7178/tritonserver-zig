# Triton Server Zig

Zig bindings and server/model repository wrappers for the Triton Inference
Server C API. Phases 1–3 of `triton_zig_roadmap.pdf` are implemented: manage
a server, load a real repository, inspect metadata, and run asynchronous inference
with owned, typed tensor results. Phase 4 adds bounded software batching and
worker scheduling, with CPU demonstrations and a comparison benchmark. Optional
GPU experiments verify CUDA execution and numerical results on this machine;
throughput and utilization remain workload-specific measurements.
Phase 5 adds a Zig-native pass-through plugin, backend-author helpers and
optional CUDA-resident tensor copies. See [backend plugins](docs/backend.md).

## Project layout

The reusable library lives in `src/core`, the reference backend example in
`src/backend`, and the standalone CLI, argument parsing, output and signal
handling in `src/server`. These boundaries follow NVIDIA Triton's core, backend
and server repositories. `src/root.zig` keeps the public API stable.
Unit/integration tests are grouped under
`src/tests`, their inputs under `tests/fixtures`, and build helpers under
`tools/build`. The existing upstream `core/` submodule is separate.

See [architecture and ownership](docs/architecture.md) for the source tree and
its relationship to NVIDIA Triton's backend/core/server separation.

## Requirements

- Zig `0.17.0-dev.1936+5a625d5f3` (the toolchain used for validation).
- Triton's headers in the `core` submodule and a compatible `libtritonserver.so`
  in `core/build/install/lib`. The existing CPU build can be rebuilt with
  `just linker`; that recipe requires the CMake and dependency paths it lists.
- `just` to use the commands below. The recipes set a writable Zig cache and
  use `zig-pkg` for package dependencies; both locations can be overridden with
  `ZIG_GLOBAL_CACHE_DIR` and `ZIG_LOCAL_PKG_DIR`. Zig fetches missing packages
  when network access is available.

## Run and verify

From the project directory:

```sh
just build
just test
just smoke
just demo
just infer-demo
just scheduler-demo
just backend-demo
just scheduler-bench --duration-seconds 1 --warmup-seconds 0
just run
```

`just smoke` (or `just run --once`) starts Triton with an empty repository,
requires `live=true`, explicitly stops the server, checks resource deletion,
and exits successfully. `just run` stays alive until Ctrl-C/SIGINT or SIGTERM;
the signal handler requests shutdown and the main thread performs cleanup.

The starter creates `.tritonserver-zig-sample-model-repository` if needed.
To choose an existing repository:

```sh
just run --model-repository /path/to/models --list-models \
  --backend-directory /path/to/backends
```

Add `--once` to list and exit after checked shutdown. Startup fails if models
cannot load. `--allow-model-errors --list-models --once` instead allows Triton
to start and prints model failure reasons; an unready server still exits with
an error after cleanup. A custom repository is never created automatically.

This is an embedded C API server with an optional software scheduler.
HTTP/gRPC listeners are outside the current milestone.

## Zig-native backend

`just backend` builds `libtriton_zig_passthrough.so` into
`zig-out/backends/zig_passthrough/`. `just backend-demo` loads it in real Triton
and checks pass-through inference. These commands require Triton's C headers
and shared library, but no C++ backend utilities or CUDA installation.

`just backend-cuda-test` explicitly builds a separate CUDA variant and checks
real CPU/pinned/device input and output buffers on local device 0. It requires
CUDA development headers, shared `cudart` and a working NVIDIA driver. GPU tests
remain local; CI runs the CPU plugin and regression suite. Existing typed
request/response and scheduler APIs retain their CPU buffer ownership rules.

## Phase 2 CPU demo

`just demo` compiles the existing upstream C++ minimal backend, loads
`tests/fixtures/models`, lists both models, and shuts down. No model download,
Python runtime, or GPU is required. The backend returns input values unchanged;
Phase 2 only loads models and inspects their metadata.

Expected model information:

| Model | Version | State | Batch limit | Input/output |
| --- | --- | --- | --- | --- |
| batching | 1 | READY | 8 | INT32, shape `[-1, 4]` |
| nonbatching | 1 | READY | 0 | INT32, shape `[4]` |

`IN0` and `OUT0` are the input and output names. Triton's protocol metadata
includes the leading dynamic batch dimension for the batching model.

Demo/integration-test prerequisites are a C++17 compiler, the bundled upstream source
in `src/backend/examples/minimal`, and the existing headers and static backend
utility libraries under `core/build/install`. RapidJSON headers are searched under
`/usr/include` and the Homebrew include directory, or set
`RAPIDJSON_INCLUDE_DIR`. Override `CXX` to select a compiler. The build script
retains upstream source/license notices and writes the shared library under
ignored `.zig-cache/phase2/backends/minimal`. Both `just test` and `just demo`
build this backend; `just build` and `just smoke` do not require it.

`just linker` builds Triton core; it does not install the separate upstream
backend utility headers/libraries used by this demo. For a clean Ubuntu 24.04
setup, [the CI workflow](.github/workflows/ci.yml) includes the full CPU build:
pinned Protobuf, RE2, Triton common and backend sources, plus the checked-out
core revision. Its native build does not require the local protobuf
compatibility changes.

## Continuous integration

[CI](.github/workflows/ci.yml) runs on pull requests, pushes to `main`, merge
queues and manual dispatch. One Linux job checks formatting and shell syntax,
builds and tests Debug and ReleaseSafe, then checks startup, model loading and
real CPU inference. Scheduler tests and a short batching benchmark also exercise
the Phase 4 CPU path. The pinned Zig toolchain and installed native dependencies
are cached. Dependency versions and upstream sources are part of the cache key;
the native recipe version is bumped when its build flags change.

CI runs only on GitHub-hosted CPU runners. GPU verification and benchmarks
are run locally with the optional tools documented below.

## Server and repository API

```zig
const std = @import("std");
const t = @import("tritonserver_zig");

var server = try t.Server.init(allocator, .{
    .model_repository = "/path/to/models",
    .backend_directory = "/path/to/backends",
});
defer server.deinit() catch |err| std.log.err("cleanup: {s}", .{@errorName(err)});

var snapshot = try server.listModels(allocator);
defer snapshot.deinit();
for (snapshot.models) |model| {
    // name, optional version, state, optional reason, optional metadata.
    // READY metadata contains platform, max_batch_size, inputs and outputs.
    // Each tensor has a name, DataType and signed protocol shape.
    _ = model;
}
try server.stop();
try server.deinit();
// snapshot remains valid here, independently of server and C-message lifetimes.
```

- `Server.init` accepts ordinary Zig strings, rejects embedded NULs, checks API
  compatibility, and releases temporary C options/strings after startup.
- `ServerOptions` also accepts `exit_on_error` (default true), `log_info`, and
  an optional synchronous `on_error` callback. The callback receives a borrowed
  message before the C error is consumed; copy it if it must outlive the call.
- `isLive()` and `isReady()` query Triton. `stop()` and `deinit()` are repeat-safe
  on the same owner. Failed deletion retains ownership for a retry; methods
  requiring a deleted server return `ServerClosed`. A stopped server cannot
  restart; health checks follow Triton's `Unavailable` contract.
- `pollRepository()` explicitly refreshes Triton's polling-mode repository.
  There is no automatic filesystem watcher or periodic polling loop.
- `Repository.scan(allocator, io, path)` produces an owned filesystem inventory
  with a resolved path and sorted model names/positive numeric versions. It
  ignores hidden model directories, files and symlinks. Use its `deinit()` to
  release the inventory. Discovery does not imply a model is loaded.
- `listModels(allocator)` uses Triton's runtime index for model state/version
  and parses its metadata/configuration JSON. Triton handles `config.pbtxt`;
  no separate protobuf-text parser is maintained. Unknown JSON fields are
  tolerated, unknown tensor data types fail explicitly, and dynamic dimensions
  remain signed. Unavailable models carry their reason and no metadata.
- `ModelList` owns a heap-stable arena containing all strings and arrays. Its
  data can outlive the server. `Server`, `Repository` and `ModelList` are owning
  values: move them and call `deinit`; do not copy and free both copies.
  Externally synchronize overlapping lifecycle calls from multiple threads.

## C bindings and ownership

```zig
const t = @import("tritonserver_zig");
// t.c exposes the declarations from the upstream tritonserver.h.
try t.check(t.c.TRITONSERVER_ServerIsLive(server, &live));
```

- The pinned Zig toolchain replaces `@cImport` with the `translate-c` package.
  `build.zig` translates the original header directly; handwritten glue or a
  manually maintained copy of the C declarations is unnecessary.
- Triton handles remain opaque. Their layout belongs to Triton; use the C API
  to access them and pair each successful `New` with its documented `Delete`.
- `check` consumes a returned C error: null means success, and each non-null
  error is mapped to a Zig error and deleted. Borrowed error messages must be
  read or copied before calling it. The starter prints C error details first.
- `DataType` and `MemoryType` use the imported C enum representation. Use
  `@backingInt` when passing them to the raw API with this Zig toolchain.
- C strings must be sentinel terminated (`[:0]const u8` for borrowed Zig
  strings). Callbacks must use `callconv(.c)` and preserve the lifetime and
  alignment of any context passed as `?*anyopaque`.
- The bindings module supplies the Triton library search path and runtime
  search path to both consumers and tests. The runtime API major version must
  match the header's major version, with a minor version at least as recent.

`just test` checks the linked API version, all eight error mappings, tensor
and memory enums through the real C functions, invalid-option errors, a server
lifecycle with an isolated empty repository, and C callback types with opaque
context. The callback test invokes a callback through its imported C function
pointer type; Phase 3 integration tests additionally exercise actual asynchronous
inference callback delivery and request/response ownership.


Phase 2 tests also cover allocator-failure cleanup, repository discovery and
version sorting, malformed JSON, dynamic shapes, model load failures, polling
new models with multiple versions, and metadata snapshots surviving server
shutdown. Expected negative tests may emit Triton's model-load error logs.

## Phase 3 inference

Run `just infer-demo` to send real INT32 tensors to both CPU demo models and
verify decoded `OUT0` values against the inputs. It exercises `inferSync` and
`inferAsync`/`PendingInference.wait`, then releases requests and shuts down.

The library exports `Request`, `PendingInference`, `InferenceResponse` and
`OutputTensor`. Inputs are copied; output snapshots are independently owned.
One-to-one models and fixed-width CPU tensors are supported in this phase.
See [inference API, ownership and errors](docs/inference.md) for examples,
concurrency/lifetime rules, test coverage and the pinned-runtime error contract.

## Phase 4 scheduling and benchmarking

`Scheduler` combines compatible queued requests into one native inference,
dispatches batches across worker threads, and returns an owned response to each
caller. Queue capacity, maximum batch samples and maximum batching delay are
configurable. `InputTensor.fromSlice` accepts borrowed fixed-width data;
submission copies it before returning. The server must outlive its scheduler.

```sh
just scheduler-demo
just scheduler-bench
```

The demo verifies distinct callers' values and observes merged batches through
a test backend. The release benchmark compares batch size 1/wait 0 against
batch size 8/wait 5 ms, using the same workload, worker count and concurrency.
Its JSON report includes samples per second, p50/p95/p99 end-to-end latency,
mean queue latency, mean batch samples and dispatch count. A CPU pass-through
model may not benefit from batching; a short CI smoke run makes no speed claim.

For a real model, specify its repository, backend directory, name, version and
a JSON tensor workload. Optional `--gpu-index 0` measures device utilization
with `nvidia-smi`; requested GPU query failures are errors. The report separates
observed device-level thresholds from verified GPU execution and correctness.
See [scheduler API, ownership and benchmark methodology](docs/scheduler.md)
for complete commands, input format, limitations and GPU validation guidance.
The [optional GPU experiments](docs/scheduler.md#optional-gpu-experiments) include
real CUDA/cuBLAS and ONNX Runtime workloads, numerical verification and recorded
hardware results. Their dependencies and generated weights stay outside CPU CI.
