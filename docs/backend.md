# Zig-native backend plugins

Phase 5 produces a real Triton backend shared library written in Zig. The CPU
plugin uses libc and the public Triton C API; it does not use the C++ backend
utilities. The licensed C++ minimal backend remains an independent reference.

## Build and run

```sh
just backend
just backend-demo
just test
```

The installed CPU plugin is
`zig-out/backends/zig_passthrough/libtriton_zig_passthrough.so`. Integration
commands also stage it under ignored `.zig-cache/phase5/backends`. Select the
parent backend directory in Triton; each model uses `backend: "zig_passthrough"`.
The valid fixture repository is `tests/fixtures/backend_models/valid`.

The plugin copies one `IN0` tensor to `OUT0`. Configuration must declare matching
fixed-width datatypes and shapes. Nonbatched and batched tensors, dynamic
dimensions and native Triton dynamic batching are supported. Each native request
receives its own result, preserving runtime shape and requested-output selection.
Split input buffers are concatenated by byte offset. BYTES, reshape, sequence or
decoupled execution, optional inputs, shape tensors and ragged batching are
rejected during model loading.

## Authoring and ownership

`@import("tritonserver_zig").backend` exposes the translated backend C API,
borrowed request/input views, response-output allocation, tagged tensor buffers,
CPU copying, plugin inspection and `exportBackend(Implementation)`.

The implementation supplies `execute(instance, request, response)` and may add
backend/model/instance initialization and finalization hooks. Hooks return an
owned `?*c.TRITONSERVER_Error`, with null indicating success. The export helper
checks signatures at compile time and generates the C ABI entrypoints. Plugin
initialization checks the runtime backend API major/minor version, and model
initialization validates configuration before allocating instance state.

Execution is synchronous. The SDK creates every response before committing to
the batch. A failure during preparation deletes earlier responses and returns
an error to Triton, which still owns all requests. After commitment, failures
become individual final error responses. The SDK attempts to release each
request and returns null from the batch callback. It logs release failures,
which leave request ownership with the backend. It also deletes each error
passed to response sending. Send consumes a valid response even if send fails.

Input metadata, pointers and request handles are borrowed until request release.
Output buffers belong to the response allocator. Never free these buffers in the
backend, retain views after execution, or reinterpret GPU pointers as host
slices. Custom allocators must provide output storage separate from the input
buffers. Validate total input bytes against datatype and runtime shape before
copying. State shared across instances must be immutable or synchronized;
per-instance state can rely on Triton's serialized execution for that instance.

`PluginLibrary.load(path)` uses `std.DynLib` to inspect a shared library and
resolve a typed vtable. Execute is required; lifecycle callbacks are optional.
The loader does not fabricate backend handles or call lifecycle callbacks.
Triton performs the authoritative runtime load. Treat the wrapper as move-only
and keep it alive while using any vtable pointer; `deinit` closes its handle.

## Upstream and toolchain references

| Zig entry point | Pinned upstream contract and adaptation |
| --- | --- |
| `TRITONZIG_*Fn`, `exportBackend`, `PluginLibrary.load` | [Backend C ABI][core-header]. C type checks keep the translation aliases aligned with the real declarations. The export helper emits the implemented C hooks, and the loader resolves them by name. |
| `InputView`, `RequestView`, `Response` | [Backend C ABI][core-header]. These wrappers expose borrowed views and report the memory kind and device the response allocator actually returned. |
| `executeRequests` | The [minimal backend][minimal] prepares responses before taking responsibility for the requests. The Zig SDK adds rollback for responses created before a later preparation failure. [Core's send and release implementation][ownership] defines the ownership of the handles and of the error passed to `ResponseSend`. |
| `zig_passthrough` configuration and lifecycle | The [minimal backend][minimal] is the reference for model and instance state. The Zig plugin adapts the example for fixed-width tensors, additional ranks, and segmented input. |
| `copyCpu`, `cuda.Context`, `executeRequest` | Compare [CopyBuffer][copy-buffer]. Host copies use libc `memmove`. The optional CUDA path uses a private stream on device 0. The example copies byte segments in order. The upstream [input collector][input-collector] and [output responder][output-responder] also provide staging and gather/scatter machinery. |

Keep vendored license notices with the C++ reference. Record the source and
license for any upstream code copied into the Zig example.

The literal callback aliases work around a debug-info issue in Zig
`0.17.0-dev.1936+5a625d5f3` with translate-c
`0944784e197e419433a21d4b28bfc65e48e7d514`. On a toolchain upgrade, verify a
Debug loader client links and has no plugin callback relocations before
removing the aliases. The C enum-folded type checks guard signature drift.
The plugin build selects LLVM and LLD for its link and export policy. On the
pinned Linux x86_64 Debug build, disabling both exposed UBSan helper symbols
alongside the plugin hooks despite the export map. Check the exported hooks,
strict shared-library linking, and the resolve-only loader tests before
changing that selection.

The SDK and example omit native request and batch statistics. The CI runtime
has statistics and metrics disabled, so the CPU tests validate outputs and
ownership. For stats-enabled deployments, a future integration needs distinct
input, compute, and output timing boundaries and must report request statistics
before `RELEASE_ALL`. Client-side benchmark latency is a separate measurement.
See the reporting API in the [pinned backend header][core-header].

[core-header]: https://github.com/triton-inference-server/core/blob/eaff04d5d77eb1c32e98c573c90fb78b363c88c0/include/triton/core/tritonbackend.h
[minimal]: https://github.com/triton-inference-server/backend/blob/efad1152ed836d336bb04cc4095ffe026e780bb7/examples/backends/minimal/src/minimal.cc
[ownership]: https://github.com/triton-inference-server/core/blob/eaff04d5d77eb1c32e98c573c90fb78b363c88c0/src/backend_model.cc
[copy-buffer]: https://github.com/triton-inference-server/backend/blob/efad1152ed836d336bb04cc4095ffe026e780bb7/src/backend_common.cc#L726
[input-collector]: https://github.com/triton-inference-server/backend/blob/efad1152ed836d336bb04cc4095ffe026e780bb7/src/backend_input_collector.cc
[output-responder]: https://github.com/triton-inference-server/backend/blob/efad1152ed836d336bb04cc4095ffe026e780bb7/src/backend_output_responder.cc

## Local CUDA verification

```sh
just backend-cuda-test
# For a toolkit outside the system prefix:
zig build backend-cuda-test -Dbackend-cuda=true -Dcuda-prefix=/usr/local/cuda
```

CUDA headers and shared `cudart` are used only when explicitly enabled. No NVCC
or custom kernel is needed. The separate artifact is
`zig-out/backends/zig_passthrough_cuda/libtriton_zig_passthrough_cuda.so`.
With `backend-cuda` enabled, the package also exposes the reusable
`triton_backend_cuda` build module (`Context.init/copy/finish/deinit`).
GPU tests use a dedicated raw C request/response allocator harness, checking
device memory tags and IDs as well as downloaded output bytes.
The plugin reports `input_memory_type`, `input_memory_type_id` and
`input_buffer_count` as integer response parameters for this verification;
the memory fields describe the first input segment.

The copy context supports CPU, pinned-host and device buffers on device 0,
including CPU→GPU, GPU→CPU and GPU→GPU. Other device IDs and peer copies are
rejected, as is managed memory. CUDA pointer attributes check the device
tag. Callers supply valid allocation extents and establish source readiness
before the private nonblocking stream reads from it. Treat device transfers
as asynchronous. Keep both allocations alive and finish queued copies before
reading the output, sending the response, or releasing the request. A partial
copy failure still requires draining earlier copies. The CUDA harness explicitly
synchronizes its upload because even a pageable host-to-device `cudaMemcpy`
can return before the device transfer completes. See the [CUDA transfer rules][cuda-sync].
The allocator's actual returned memory type controls the copy operation.

[cuda-sync]: https://docs.nvidia.com/cuda/archive/12.0.1/cuda-runtime-api/api-sync-behavior.html

The current Triton build has native GPU support disabled. CUDA fixtures use
`KIND_CPU` with application-owned CUDA allocations and explicit backend copies;
this does not enable `KIND_GPU` instances or Triton's GPU memory pools. Existing
typed `Request`, `InferenceResponse` and `Scheduler` APIs continue to own CPU
buffers. CI validates the CPU plugin only; GPU verification runs locally.
