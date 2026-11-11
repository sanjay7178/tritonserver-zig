# Source organization

The architectural references are NVIDIA Triton's three repositories:

| Upstream boundary | Responsibility | Location here |
| --- | --- | --- |
| [core](https://github.com/triton-inference-server/core#readme) | Reusable inference-server library, model management and public API | `src/core` and public `src/root.zig` |
| [backend](https://github.com/triton-inference-server/backend#readme) | Model execution plugins and backend-author examples/utilities | `src/backend` |
| [server](https://github.com/triton-inference-server/server/tree/main/src) | Standalone executable, command-line options, signal handling and frontends | `src/server` |

The same separation is kept within one Zig checkout. The backend example is
upstream C++, copied with its license; the Phase 2 core wrappers and standalone
process are Zig. No new Zig-native backend is introduced in this phase.

```text
src/
  root.zig                         Stable public tritonserver_zig exports
  core/                            Reusable inference-server library (Zig)
    bindings/                      Raw C declarations, errors and enum types
    server.zig                     Lifecycle, health and runtime model index
    model/
      repository.zig               Filesystem discovery and version inventory
      metadata.zig                 Typed metadata/config JSON and owned model lists
    inference/                     Request builder, async callbacks and owned results
    internal/owned_arena.zig        Heap-stable snapshot ownership
  backend/                         Model execution / backend-author boundary
    examples/minimal/              Licensed upstream C++ reference plugin
  server/                          Standalone process/CLI (Zig)
    main.zig                       Startup, repository selection and shutdown
    command_line.zig               CLI options, validation and argument tests
    output.zig                     Model/tensor presentation
    inference_demo.zig             CPU inference acceptance demo
    signals.zig                    Shutdown requests and handler restoration
  tests/
    unit/                          JSON/metadata and filesystem discovery tests
    integration/                   Real C bindings, lifecycle, models and inference

tests/fixtures/models/             Batched and non-batched CPU model repositories
tools/build/build_demo_backend.sh   Build helper; outputs into ignored .zig-cache
build.zig                          Module, executable, test, smoke and demo graph
core/                              Existing upstream C++ submodule/build dependency
backend/                           Pre-existing upstream C++ checkout/dependency
```

## Upstream provenance

The [symbol-level provenance index](upstream-provenance.md) records pinned
repositories, source paths, C API contracts, Zig adaptations, regression tests,
and native build flags. `src/core` wraps the native core; `Repository.scan` is
an independent filesystem inventory, and `src/server` is a project CLI rather
than an upstream HTTP/gRPC frontend port.

## Dependency and ownership boundaries

`src/server` consumes the public library through `@import("tritonserver_zig")`.
`src/core` calls the translated C API and does not import CLI parsing, formatting,
signals, or backend implementation source. Its metadata/repository code shares
an internal ownership helper. The backend example compiles independently into a
shared library that Triton loads at runtime through the C backend ABI.

`src/root.zig` remains the public entrypoint and retains existing public names.
Neither the root `core/` submodule nor the existing `backend/` checkout is used
as a destination for Zig implementation files. Tests live under `src/tests` so
library parser tests stay within Zig's module source root. Root's test block
imports them; CLI argument tests run separately without linking Triton.

## Runtime data flow

1. The CLI scans an existing repository into an owned filesystem inventory.
2. Server initialization copies paths to C strings and creates a Triton server;
   temporary options and strings are released before returning.
3. Triton parses configurations and loads backend plugins. Its runtime model
   index determines readiness and versions, independently of filesystem discovery.
4. Model listing parses C-owned metadata/configuration JSON into a separate
   snapshot arena, then deletes the original messages.
5. Callers deinitialize snapshots and inventories independently. A snapshot can
   outlive the server. Signal handlers set a flag; the main thread performs
   server stop/deletion.

Phase 3 request handling lives in `src/core/inference`: typed input copying,
C asynchronous submission, response allocation, dual-callback completion and
owned output decoding. The CLI demo consumes these library APIs; no model
execution code is moved into the CLI. See [inference ownership](inference.md).
Later network frontends belong in `src/server`, and Zig-native plugins/backend
helpers belong in `src/backend` when the roadmap reaches that phase.
