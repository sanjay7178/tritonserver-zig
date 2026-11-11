# Phase 3: request pipeline

The milestone is a real request sent to a loaded model, completed through
Triton's asynchronous C callbacks, and decoded into owned typed output tensors.
`just infer-demo` verifies this against both CPU models, including a full batch
of eight samples with shape `[8, 4]`.

## Library API

The following is a function-body example with a live `server: *const t.Server`
and a caller-provided `allocator`:

```zig
const t = @import("tritonserver_zig");
var request = try t.Request.init(allocator, server, "nonbatching", -1);
defer request.deinit() catch @panic("request cleanup failed");
try request.addInput(i32, "IN0", &.{4}, &.{ 1, 2, 3, 4 });
try request.requestOutput("OUT0"); // Omit to request all outputs.

var response = try server.inferSync(allocator, &request);
defer response.deinit();
const out = try response.output("OUT0");
const values = try out.asSlice(i32); // Borrowed until response.deinit().
// values == {1, 2, 3, 4}; out.shape == {4}; out.datatype == .int32
_ = values;
```

`addInput` copies the name, shape and data; caller buffers may be changed or
freed after it returns. Dimensions must be concrete and non-negative; their
product must equal the typed element count, without overflow. The empty shape
is a scalar and zero-sized dimensions represent empty tensors. Supported input
and typed-view elements are bool, u8/u16/u32/u64, i8/i16/i32/i64 and f16/f32/f64.
Duplicate inputs/outputs and names containing NUL bytes are rejected.

`InferenceResponse` owns the model name/version, output names, shapes and aligned
CPU buffers. Its contents survive deletion of the raw C response, Request and
Server. `output(name)` fails with `OutputNotFound`; `asSlice(T)` checks the
returned data type and buffer size rather than silently reinterpreting data.
Response deinitialization is repeat-safe on the same owner.

Requests retain copied inputs for reuse after a completed operation. To change
the set of inputs, construct a new Request. Move owning values instead of copying
them and attempting to deinitialize multiple copies.

## Asynchronous ownership

```zig
var pending = try server.inferAsync(allocator, &request);
defer pending.deinit();
// Independent requests can be submitted here before waiting.
var response = try pending.wait();
defer response.deinit();
```

- An accepted submit transfers the C request to Triton. A rejected submit keeps
  ownership with Request and returns immediately without waiting for callbacks.
- The response callback owns each non-null C response. FINAL may also carry a
  null response. Request ownership and copied inputs return only on RELEASE_ALL.
- Each pending operation has a stable heap context, protected by a pthread mutex
  and condition variable. This adapts the roadmap's older semaphore example to
  the installed Zig toolchain. The waiter requires **both** FINAL and RELEASE_ALL;
  their arrival order does not matter. Signaling alone never frees the context.
- C output allocation callbacks use thread-safe libc allocation and return CPU
  memory regardless of the preferred location. The waiting thread copies results
  using the caller allocator. Raw responses are deleted before their allocator.
- `PendingInference.wait` consumes the operation even if decoding fails. A second
  wait returns `InferenceAlreadyCompleted`. `deinit` instead drains and discards
  the result, without allocating an output snapshot. It can block.
- Pending operations have one owner/waiter. Keep Request at a stable address
  until wait/deinit finishes. Mutation or deletion through the same Request while
  pending returns `RequestInFlight`; this flag is not a shared-request lock.
- Separate requests may run on separate threads through a live Server. Use
  separate or thread-safe allocators for concurrent callers. Callbacks never use
  the caller allocator. Externally synchronize server lifecycle operations;
  drain pending work and delete requests **before** stopping/deleting Server.

## Supported boundary and errors

This phase supports one-to-one models with fixed-width CPU tensor results.
Decoupled models are rejected during request creation with
`DecoupledModelUnsupported`. Serialized BYTES tensors and direct GPU/pinned-memory
result views are not implemented. Raw C bindings remain available for extensions.
There is no timeout that could return while C still owns callback state; drain
accepted work even when an application error occurs.

Input/model validation errors, immediate submission errors, backend error
responses and output-copy allocation failures return Zig errors. The existing
ServerOptions.on_error callback receives C diagnostics synchronously on the
calling/waiting thread before the C error is released.

One source/header inconsistency matters for this pinned local runtime:
`core/include/triton/core/tritonserver.h` describes the error returned by
InferenceResponseError as borrowed, while `core/src/tritonserver.cc` creates a
new error using RETURN_IF_STATUS_ERROR. Core's own query tests consume that
error independently of the response. This implementation calls the accessor
once and consumes its error, matching the checked-out implementation and linked
library; review this contract when changing the upstream Triton runtime.

## Verification

`just test` exercises input-copy isolation, typed non-batched/batched results,
request reuse, result lifetimes, four concurrent callers, invalid inputs and
outputs, allocator failure at every caller-allocation site, pending draining,
callback completion in both orders, null FINAL responses and zero-byte response
allocation. A tiny test-only C++ backend under
`tests/fixtures/backends/phase3_error` returns an execution error to exercise the
accepted asynchronous error-response path. The licensed pass-through backend
is unchanged. `just infer-demo` uses only the valid pass-through models.
