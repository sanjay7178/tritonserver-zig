# Upstream provenance

This package wraps the native Triton core through its C API. The Zig filesystem
inventory, JSON decoding, completion owner, and embedded CLI are project code.
They do not replace the native model manager, scheduler, or HTTP/gRPC frontends.
This index maps upstream fixes to the Zig boundary that needs review.

## Source pins

| Repository / tool | Full revision | Used by |
| --- | --- | --- |
| [triton-inference-server/core](https://github.com/triton-inference-server/core) | `eaff04d5d77eb1c32e98c573c90fb78b363c88c0` | `core` gitlink, translated header and linked runtime |
| [triton-inference-server/common](https://github.com/triton-inference-server/common) | `fd2237e516a7e863af65e327f5e4fbb2cba70f09` | Native runtime/backend utilities in CI |
| [triton-inference-server/backend](https://github.com/triton-inference-server/backend) | `efad1152ed836d336bb04cc4095ffe026e780bb7` | Native backend utilities and vendored minimal example |
| [protocolbuffers/protobuf](https://github.com/protocolbuffers/protobuf) | `f0dc78d7e6e331b8c6bb2d5283e06aa26883ca7c` (v21.12) | Native build in CI |
| [google/re2](https://github.com/google/re2) | `5723bb8950318135ed9cf4fc76bed988a087f536` (2022-06-01) | Native build in CI |
| [ziglang/translate-c](https://codeberg.org/ziglang/translate-c) | `0944784e197e419433a21d4b28bfc65e48e7d514` | Header translation via `build.zig.zon` |
| Zig | `0.17.0-dev.1936+5a625d5f3` | Package minimum and CI compiler |

The gitlink, [package manifest](../build.zig.zon), and
[CI workflow](../.github/workflows/ci.yml) are the executable sources of these
pins. The upstream `server` repository is an architectural reference only;
no server/frontend source is vendored or ported by this PR.

## Symbol map

All local paths below are relative to `src/` unless stated otherwise. Each
upstream link includes its repository, full commit, and file path. The test
groups below name existing regressions; project extensions have no direct
upstream implementation.

| Local file / symbols | Pinned upstream path / symbols | Relation and contract | Regression group |
| --- | --- | --- | --- |
| `core/bindings/c.zig: raw`; root `build.zig: translated` | [core/include/triton/core/tritonserver.h][core-header] :: public C declarations | Generated ABI; declarations come from the pinned header, implementation from the linked runtime. | [Bindings](#bindings) |
| `core/bindings/error.zig: check` | [core/src/tritonserver.cc][core-api] :: `TRITONSERVER_ErrorCode`, `TRITONSERVER_ErrorDelete` | Zig error/ownership adapter; consumes each non-null owned error exactly once. | [Bindings](#bindings) |
| `core/bindings/datatype.zig: DataType, MemoryType` | [core/include/triton/core/tritonserver.h][core-header] :: `TRITONSERVER_DataType`, `TRITONSERVER_MemoryType` | Typed enum wrappers around ABI values. | [Bindings](#bindings) |
| `core/server.zig: apiVersion, Server.init` | [core/src/tritonserver.cc][core-api] :: `TRITONSERVER_ApiVersion`, `TRITONSERVER_ServerOptions*`, `TRITONSERVER_ServerNew` | C API wrapper; temporary options/strings are released after initialization. Version compatibility does not validate implementation-specific ownership. | [Server and models](#server-and-models) |
| `core/server.zig: Server.isLive, isReady, stop, deinit, pollRepository` | [core/src/tritonserver.cc][core-api] :: `TRITONSERVER_ServerIsLive`, `ServerIsReady`, `ServerStop`, `ServerDelete`, `ServerPollModelRepository` (all with `TRITONSERVER_` prefix) | C API wrappers; callers synchronize lifecycle and drain requests before server deletion. | [Server and models](#server-and-models) |
| `core/server.zig: Server.listModels, Server.messageBytes` | [core/src/tritonserver.cc][core-api] :: `TRITONSERVER_ServerModelIndex`, `ServerModelMetadata`, `ServerModelConfig`, `MessageSerializeToJson`, `MessageDelete` (same prefix) | C API wrappers plus independent snapshots; borrowed JSON is copied before its message is deleted. | [Server and models](#server-and-models) |
| `core/model/metadata.zig: parseIndex, parseMetadata, convertTensors` | [core/src/tritonserver.cc][core-api] :: JSON from `TRITONSERVER_ServerModelIndex`, `TRITONSERVER_ServerModelMetadata`, `TRITONSERVER_ServerModelConfig`; `TRITONSERVER_StringToDataType` | Project extension: JSON decoding, no direct upstream implementation; not a protobuf/model-manager port. | [Metadata and inventory](#metadata-and-inventory) |
| `core/model/repository.zig: Repository.scan` | Compare [core/src/model_repository_manager/model_repository_manager.cc][core-repository] :: `ModelRepositoryManager::Poll` | Project extension: filesystem inventory, no direct upstream implementation. Native loading, version policy, dependencies and readiness stay upstream; scan results are not the runtime model index. | [Metadata and inventory](#metadata-and-inventory) |
| `core/inference/request.zig: Request.init, addInput, requestOutput, deinit` | [core/include/triton/core/tritonserver.h][core-header] and [core/src/infer_request.cc][core-request] :: `TRITONSERVER_InferenceRequestNew`, `InferenceRequestAddInput`, `InferenceRequestAppendInputData`, `InferenceRequestAddRequestedOutput`, `InferenceRequestDelete` (same prefix); native `InferenceRequest` | C API wrapper; owns CPU input copies and rejects decoupled models. | [Inference ownership](#inference-ownership) |
| `core/inference/operation.zig: submit, allocate, release, onResponse, onRelease; PendingInference.wait, deinit` | [core/include/triton/core/tritonserver.h][core-header] :: `TRITONSERVER_ServerInferAsync`, `TRITONSERVER_ResponseAllocatorNew`, `TRITONSERVER_InferenceResponseCompleteFn_t`, `TRITONSERVER_InferenceRequestReleaseFn_t` | Project extension: new Zig completion owner, no direct upstream implementation. Single waiter drains both FINAL and RELEASE_ALL in either order, including null FINAL. Callbacks allocate only through libc; caller allocation stays on the submitting/waiting thread. Task cancellation must not free C-owned callback state. | [Completion](#completion), [Inference ownership](#inference-ownership) |
| `core/inference/response.zig: decode` | [core/src/tritonserver.cc][core-api] and [core/src/infer_response.cc][core-response] :: `TRITONSERVER_InferenceResponseError`, `InferenceResponseModel`, `InferenceResponseOutputCount`, `InferenceResponseOutput` (same prefix); native `InferenceResponse` | Wrapper plus owned CPU snapshot. At this core pin the error accessor allocates an owned error despite the header saying borrowed; consume once and re-audit on every core update. | [Inference ownership](#inference-ownership) |
| `core/inference/tensor.zig: dataType, elementCount, OutputTensor.asSlice`; `core/internal/owned_arena.zig: OwnedArena.init, allocator, deinit` | [core/include/triton/core/tritonserver.h][core-header] :: fixed-width C tensor representation and `TRITONSERVER_DataTypeByteSize` | Project extensions: Zig validation/aligned views and heap-stable snapshot lifetimes, no direct upstream implementation. Serialized BYTES and direct GPU/pinned-memory output views are unsupported. | [Metadata and inventory](#metadata-and-inventory), [Inference ownership](#inference-ownership) |
| `server/main.zig: main`; `server/command_line.zig`, `signals.zig`, `output.zig`, `inference_demo.zig` | This package's public API in `root.zig` | Project extensions, no direct upstream implementation. Embedded CLI, signal handling and CPU demo; not an HTTP/gRPC frontend port. | [CLI and fixtures](#cli-and-fixtures) |
| `backend/examples/minimal/minimal.cc`, `libtriton_minimal.ldscript` | [backend/examples/backends/minimal/src][backend-minimal] :: `TRITONBACKEND_ModelInitialize`, `ModelFinalize`, `ModelInstanceInitialize`, `ModelInstanceFinalize`, `ModelInstanceExecute` (all with `TRITONBACKEND_` prefix); exported `TRITONBACKEND_*` symbols | Vendored C++ and linker script, unchanged from the backend pin; not a Zig rewrite. See the [fixture provenance and hashes](../src/backend/examples/minimal/README.md). | [CLI and fixtures](#cli-and-fixtures) |
| Root `tests/fixtures`, `tools/build/build_demo_backend.sh`, `justfile`, `.github/workflows/ci.yml` | The contracts and pinned native sources above | Project-specific test/build infrastructure, no direct upstream implementation. | [CLI and fixtures](#cli-and-fixtures) |

## Existing regression coverage

### Bindings

[`bindings_test.zig`](../src/tests/integration/bindings_test.zig):

- `linked Triton API is compatible with the imported header`
- `check consumes each C error and maps null to success`
- `Zig tensor and memory enums round trip through the C API`

### Server and models

[`server_test.zig`](../src/tests/integration/server_test.zig):

- `server validates paths, cleans up failed initialization and rejects closed operations`
- `real CPU models expose typed metadata and snapshots outlive the server`
- `unavailable models retain failure reasons when fail-fast is disabled`
- `repository and server initialization release each partial allocation`
- `repository polling loads real models with multiple versions`

### Metadata and inventory

- [`model_test.zig`](../src/tests/unit/model_test.zig): `model parsing releases partial snapshots at every allocation failure`; `metadata snapshot owns strings, protocol shapes and batch limits`; `index ordering includes unavailable and name-only models`; `malformed index, versions, dtypes, shapes and batch limits fail explicitly`.
- [`repository_test.zig`](../src/tests/unit/repository_test.zig): `repository scan sorts models and positive numeric versions and ignores files`; `repository scan handles empty, missing and invalid paths`.
- [`tensor_test.zig`](../src/tests/unit/tensor_test.zig): `fixed-width tensor type mapping and concrete shape sizes`.

### Completion

[`operation.zig`](../src/core/inference/operation.zig):

- `completion context handles FINAL and RELEASE_ALL in either order` also covers null FINAL and ignoring RELEASE_RESCHEDULE.
- `CPU allocator handles zero-length outputs and overrides memory preference`

### Inference ownership

[`inference_test.zig`](../src/tests/integration/inference_test.zig):

- `real inference copies caller inputs and returns owned typed outputs`
- `real batched inference decodes the full batch with its protocol shape`
- `independent callers can infer concurrently through the same live server`
- `released requests can be reused and each response remains independent`
- `request and response allocation failures release partial ownership`
- `asynchronous backend errors map and release all callback ownership` protects the response-error path at the pinned runtime.
- `pending inference guards request ownership until wait and supports safe draining`
- `request initialization rejects decoupled models before submission`

### CLI and fixtures

[`command_line.zig`](../src/server/command_line.zig) covers CLI arguments,
including `starter accepts the finite inference demo`. The `smoke`, `demo`,
and `infer-demo` build steps check server startup, fixture metadata, and typed
CPU outputs; the inference integration tests exercise the vendored backend.

## Native build configuration

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) contains the full
bootstrap commands; these are the flags that define this PR's CPU boundary:

- Protobuf and RE2: Release, position-independent static libraries, native tests off; Protobuf zlib support off.
- Core: Release, `TRITON_VERSION=0.0.0`, logging on; GPU, Mali GPU, metrics (including CPU/GPU metrics), stats, tracing, NVTX, ensemble, GCS, S3 and Azure storage off. The pinned common checkout supplies dependencies; Python protobuf generation is off. CI installs the `triton-core` target's `libtritonserver.so`.
- Backend utilities: Release, position-independent code, `TRITON_CORE_HEADERS_ONLY=ON`, GPU/stats off, common protobuf/Python generation off; core headers and common sources come from the pins above.
- Zig: translates `core/include/triton/core/tritonserver.h`, links and sets an rpath to `core/build/install/lib`; CI builds/tests Debug and ReleaseSafe and runs the three demos.

## Updating NVIDIA dependencies

1. Diff the pinned files and symbols above against the proposed core/common/backend revisions. Classify each change as inherited through native linking, requiring a Zig adaptation, or intentionally inapplicable to this CPU scope.
2. Re-audit `TRITONSERVER_InferenceResponseError` ownership against the implementation and header on every core update. API major/minor checks do not establish this ownership contract; keep the consume-once call site and asynchronous backend-error regression aligned with the linked library.
3. Preserve both completion events, null FINAL handling, libc-only callback allocation, and drain-before-free behavior through any `std.Io`/async refactor. Zig task cancellation does not cancel native callbacks.
4. Compare both vendored minimal-backend files against the same backend revision, retaining their licenses. Update their hashes, source pins, this map, native flags and affected regressions together.

[core-header]: https://github.com/triton-inference-server/core/blob/eaff04d5d77eb1c32e98c573c90fb78b363c88c0/include/triton/core/tritonserver.h
[core-api]: https://github.com/triton-inference-server/core/blob/eaff04d5d77eb1c32e98c573c90fb78b363c88c0/src/tritonserver.cc
[core-repository]: https://github.com/triton-inference-server/core/blob/eaff04d5d77eb1c32e98c573c90fb78b363c88c0/src/model_repository_manager/model_repository_manager.cc
[core-request]: https://github.com/triton-inference-server/core/blob/eaff04d5d77eb1c32e98c573c90fb78b363c88c0/src/infer_request.cc
[core-response]: https://github.com/triton-inference-server/core/blob/eaff04d5d77eb1c32e98c573c90fb78b363c88c0/src/infer_response.cc
[backend-minimal]: https://github.com/triton-inference-server/backend/tree/efad1152ed836d336bb04cc4095ffe026e780bb7/examples/backends/minimal/src
