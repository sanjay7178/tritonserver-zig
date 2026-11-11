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
become individual final error responses; every request is released and the
batch callback returns null. Errors passed to response sending are deleted by
the SDK. Sending transfers the response ownership, including send failures.

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
rejected, as is managed memory. Device tags are checked against CUDA pointer
attributes before copying; callers remain responsible for valid allocation
extents and input contents that are ready before submission. It uses a private
stream per instance and completes queued copies
before returning a response or releasing a request, including error paths.
The allocator's actual returned memory type controls the copy operation.

The current Triton build has native GPU support disabled. CUDA fixtures use
`KIND_CPU` with application-owned CUDA allocations and explicit backend copies;
this does not enable `KIND_GPU` instances or Triton's GPU memory pools. Existing
typed `Request`, `InferenceResponse` and `Scheduler` APIs continue to own CPU
buffers. CI validates the CPU plugin only; GPU verification runs locally.
