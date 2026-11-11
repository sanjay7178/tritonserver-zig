const std = @import("std");
const c = @import("../core/bindings/c.zig").raw;
const check = @import("../core/bindings/error.zig").check;
const tensor = @import("tensor.zig");

pub const InputProperties = struct {
    name: [:0]const u8,
    datatype: c.TRITONSERVER_DataType,
    shape: []const i64,
    byte_size: usize,
    buffer_count: u32,
};

/// Borrowed until the parent request is released. Neither shapes nor buffers
/// may be freed, mutated, or retained past that point.
pub const InputView = struct {
    handle: *c.TRITONBACKEND_Input,

    pub fn properties(self: InputView) !InputProperties {
        var name: [*c]const u8 = null;
        var datatype: c.TRITONSERVER_DataType = 0;
        var shape: [*c]const i64 = null;
        var rank: u32 = 0;
        var byte_size: u64 = 0;
        var buffer_count: u32 = 0;
        try check(c.TRITONBACKEND_InputProperties(self.handle, &name, &datatype, &shape, &rank, &byte_size, &buffer_count));
        if (name == null or (rank != 0 and shape == null)) return error.InvalidInputProperties;
        return .{
            .name = std.mem.span(name),
            .datatype = datatype,
            .shape = if (rank == 0) &.{} else shape[0..rank],
            .byte_size = std.math.cast(usize, byte_size) orelse return error.BufferSizeOverflow,
            .buffer_count = buffer_count,
        };
    }

    pub fn buffer(self: InputView, index: u32) !tensor.ConstBuffer {
        var ptr: ?*const anyopaque = null;
        var size: u64 = 0;
        var memory: c.TRITONSERVER_MemoryType = c.TRITONSERVER_MEMORY_CPU;
        var id: i64 = 0;
        try check(c.TRITONBACKEND_InputBuffer(self.handle, index, &ptr, &size, &memory, &id));
        const result: tensor.ConstBuffer = .{ .ptr = ptr, .byte_size = std.math.cast(usize, size) orelse return error.BufferSizeOverflow, .memory_type = memory, .memory_type_id = id };
        try result.validate();
        return result;
    }
};

pub const RequestView = struct {
    handle: *c.TRITONBACKEND_Request,

    pub fn inputCount(self: RequestView) !u32 {
        var count: u32 = 0;
        try check(c.TRITONBACKEND_RequestInputCount(self.handle, &count));
        return count;
    }

    pub fn input(self: RequestView, name: [:0]const u8) !InputView {
        var handle: ?*c.TRITONBACKEND_Input = null;
        try check(c.TRITONBACKEND_RequestInput(self.handle, name, &handle));
        return .{ .handle = handle orelse return error.InvalidInputHandle };
    }

    pub fn inputByIndex(self: RequestView, index: u32) !InputView {
        var handle: ?*c.TRITONBACKEND_Input = null;
        try check(c.TRITONBACKEND_RequestInputByIndex(self.handle, index, &handle));
        return .{ .handle = handle orelse return error.InvalidInputHandle };
    }

    pub fn requestedOutputCount(self: RequestView) !u32 {
        var count: u32 = 0;
        try check(c.TRITONBACKEND_RequestOutputCount(self.handle, &count));
        return count;
    }

    pub fn requestedOutputName(self: RequestView, index: u32) ![:0]const u8 {
        var name: [*c]const u8 = null;
        try check(c.TRITONBACKEND_RequestOutputName(self.handle, index, &name));
        if (name == null) return error.InvalidOutputName;
        return std.mem.span(name);
    }
};

/// Response ownership belongs to the SDK's execute transaction. Helpers expose
/// borrowed allocator-owned buffers; the implementation must finish all writes
/// and device operations before its execute hook returns.
pub const Response = struct {
    handle: *c.TRITONBACKEND_Response,

    pub fn setIntParameter(self: *Response, name: [:0]const u8, value: i64) !void {
        try check(c.TRITONBACKEND_ResponseSetIntParameter(self.handle, name, value));
    }

    pub fn output(self: *Response, name: [:0]const u8, datatype: c.TRITONSERVER_DataType, shape: []const i64, byte_size: usize, preferred_memory: c.TRITONSERVER_MemoryType, preferred_id: i64) !tensor.Buffer {
        if (try tensor.byteCount(datatype, shape) != byte_size) return error.ElementCountMismatch;
        const rank = std.math.cast(u32, shape.len) orelse return error.ShapeOverflow;
        var output_handle: ?*c.TRITONBACKEND_Output = null;
        try check(c.TRITONBACKEND_ResponseOutput(self.handle, &output_handle, name, datatype, if (shape.len == 0) null else shape.ptr, rank));
        const output_value = output_handle orelse return error.InvalidOutputHandle;
        var ptr: ?*anyopaque = null;
        var memory = preferred_memory;
        var id = preferred_id;
        try check(c.TRITONBACKEND_OutputBuffer(output_value, &ptr, @intCast(byte_size), &memory, &id));
        const result: tensor.Buffer = .{ .ptr = ptr, .byte_size = byte_size, .memory_type = memory, .memory_type_id = id };
        try result.validate();
        return result;
    }
};

/// Convert a Zig failure into an owned C error for an exported hook. Native C
/// errors returned directly by implementations already have this ownership.
pub fn errorToC(err: anyerror) *c.TRITONSERVER_Error {
    const code: c.TRITONSERVER_Error_Code = switch (err) {
        error.Unsupported,
        error.UnsupportedDataType,
        error.UnsupportedMemoryType,
        error.UnsupportedDevice,
        error.UnsupportedModelConfiguration,
        error.IncompatibleApiVersion,
        => c.TRITONSERVER_ERROR_UNSUPPORTED,
        error.InvalidArg,
        error.InvalidShape,
        error.InvalidBuffer,
        error.InvalidBufferRange,
        error.InvalidMemoryTypeId,
        error.InvalidDevicePointer,
        error.InvalidDeviceBuffer,
        error.InvalidMemoryType,
        error.OverlappingDeviceBuffers,
        error.BufferAddressOverflow,
        error.BufferSizeOverflow,
        error.ShapeOverflow,
        error.BufferSizeMismatch,
        error.ElementCountMismatch,
        error.InvalidInputBuffer,
        error.InvalidInputProperties,
        error.InvalidInputCount,
        error.InvalidTensorName,
        error.InvalidModelConfiguration,
        error.DataTypeMismatch,
        error.UnknownOutput,
        error.InvalidOutputName,
        => c.TRITONSERVER_ERROR_INVALID_ARG,
        error.NotFound => c.TRITONSERVER_ERROR_NOT_FOUND,
        error.Unavailable => c.TRITONSERVER_ERROR_UNAVAILABLE,
        error.AlreadyExists => c.TRITONSERVER_ERROR_ALREADY_EXISTS,
        error.Cancelled => c.TRITONSERVER_ERROR_CANCELLED,
        else => c.TRITONSERVER_ERROR_INTERNAL,
    };
    return c.TRITONSERVER_ErrorNew(code, @errorName(err)).?;
}

const RealApi = struct {
    fn responseNew(out: *?*c.TRITONBACKEND_Response, request: *c.TRITONBACKEND_Request) ?*c.TRITONSERVER_Error {
        return c.TRITONBACKEND_ResponseNew(out, request);
    }
    fn responseDelete(response: *c.TRITONBACKEND_Response) ?*c.TRITONSERVER_Error {
        return c.TRITONBACKEND_ResponseDelete(response);
    }
    fn responseSend(response: *c.TRITONBACKEND_Response, failure: ?*c.TRITONSERVER_Error) ?*c.TRITONSERVER_Error {
        return c.TRITONBACKEND_ResponseSend(response, c.TRITONSERVER_RESPONSE_COMPLETE_FINAL, failure);
    }
    fn requestRelease(request: *c.TRITONBACKEND_Request) ?*c.TRITONSERVER_Error {
        return c.TRITONBACKEND_RequestRelease(request, c.TRITONSERVER_REQUEST_RELEASE_ALL);
    }
    fn deleteError(failure: *c.TRITONSERVER_Error) void {
        c.TRITONSERVER_ErrorDelete(failure);
    }
    fn logAndDelete(failure: ?*c.TRITONSERVER_Error) void {
        if (failure) |owned| {
            std.debug.print("Triton backend cleanup: {s}\n", .{std.mem.span(c.TRITONSERVER_ErrorMessage(owned))});
            c.TRITONSERVER_ErrorDelete(owned);
        }
    }
};

fn executeRequests(comptime Impl: type, comptime Api: type, alloc: std.mem.Allocator, instance: *c.TRITONBACKEND_ModelInstance, requests: []const ?*c.TRITONBACKEND_Request) ?*c.TRITONSERVER_Error {
    for (requests) |request| if (request == null) return errorToC(error.InvalidArg);
    const responses = alloc.alloc(?*c.TRITONBACKEND_Response, requests.len) catch |err| return errorToC(err);
    defer alloc.free(responses);
    @memset(responses, null);
    for (requests, responses, 0..) |request, *response, index| {
        if (Api.responseNew(response, request.?)) |failure| {
            for (responses[0..index]) |created| if (created) |value| Api.logAndDelete(Api.responseDelete(value));
            return failure;
        }
        if (response.* == null) {
            for (responses[0..index]) |created| if (created) |value| Api.logAndDelete(Api.responseDelete(value));
            return errorToC(error.InvalidResponseHandle);
        }
    }
    // Commitment: every request now has a response. Return success regardless
    // of per-request processing failures, otherwise Triton handles them twice.
    for (requests, responses) |request, raw_response| {
        var response: Response = .{ .handle = raw_response.? };
        const failure = Impl.execute(instance, .{ .handle = request.? }, &response);
        Api.logAndDelete(Api.responseSend(response.handle, failure));
        if (failure) |owned| Api.deleteError(owned);
        // A release error is a host failure that retains backend ownership.
        // Log it; never turn an already committed batch into a returned error.
        Api.logAndDelete(Api.requestRelease(request.?));
    }
    return null;
}

fn requireSignature(comptime Impl: type, comptime name: []const u8, comptime Expected: type) void {
    if (!@hasDecl(Impl, name)) @compileError("backend implementation requires " ++ name);
    if (@TypeOf(@field(Impl, name)) != Expected)
        @compileError("invalid backend hook signature: " ++ name ++ "; expected " ++ @typeName(Expected));
}

fn exportOptional(comptime Impl: type, comptime Thunks: type, comptime field: []const u8, comptime symbol: []const u8, comptime FunctionPointer: type) void {
    if (@hasDecl(Impl, field)) {
        requireSignature(Impl, field, @typeInfo(FunctionPointer).pointer.child);
        const thunk = &@field(Thunks, field);
        if (@TypeOf(thunk) != FunctionPointer) @compileError("backend C thunk does not match " ++ symbol);
        @export(thunk, .{ .name = symbol });
    }
}

/// Export ABI-checked C entrypoints. Lifecycle hooks are raw C ABI functions;
/// execute is a Zig hook for one request and one SDK-owned response. Only hooks
/// declared by Impl are exported, except the required execute transaction.
pub fn exportBackend(comptime Impl: type) void {
    requireSignature(Impl, "execute", fn (*c.TRITONBACKEND_ModelInstance, RequestView, *Response) ?*c.TRITONSERVER_Error);
    const Thunks = struct {
        fn execute(instance: ?*c.TRITONBACKEND_ModelInstance, requests: [*c]?*c.TRITONBACKEND_Request, request_count: u32) callconv(.c) ?*c.TRITONSERVER_Error {
            const value = instance orelse return errorToC(error.InvalidArg);
            if (request_count == 0) return null;
            if (requests == null) return errorToC(error.InvalidArg);
            return executeRequests(Impl, RealApi, std.heap.c_allocator, value, requests[0..request_count]);
        }
        fn initialize(value: ?*c.TRITONBACKEND_Backend) callconv(.c) ?*c.TRITONSERVER_Error {
            return Impl.initialize(value);
        }
        fn finalize(value: ?*c.TRITONBACKEND_Backend) callconv(.c) ?*c.TRITONSERVER_Error {
            return Impl.finalize(value);
        }
        fn modelInitialize(value: ?*c.TRITONBACKEND_Model) callconv(.c) ?*c.TRITONSERVER_Error {
            return Impl.modelInitialize(value);
        }
        fn modelFinalize(value: ?*c.TRITONBACKEND_Model) callconv(.c) ?*c.TRITONSERVER_Error {
            return Impl.modelFinalize(value);
        }
        fn modelInstanceInitialize(value: ?*c.TRITONBACKEND_ModelInstance) callconv(.c) ?*c.TRITONSERVER_Error {
            return Impl.modelInstanceInitialize(value);
        }
        fn modelInstanceFinalize(value: ?*c.TRITONBACKEND_ModelInstance) callconv(.c) ?*c.TRITONSERVER_Error {
            return Impl.modelInstanceFinalize(value);
        }
        fn modelInstanceReady(value: ?*c.TRITONBACKEND_ModelInstance) callconv(.c) ?*c.TRITONSERVER_Error {
            return Impl.modelInstanceReady(value);
        }
        fn getBackendAttribute(value: ?*c.TRITONBACKEND_Backend, attributes: ?*c.TRITONBACKEND_BackendAttribute) callconv(.c) ?*c.TRITONSERVER_Error {
            return Impl.getBackendAttribute(value, attributes);
        }
    };
    if (@TypeOf(&Thunks.execute) != *const c.TRITONZIG_ModelInstanceExecuteFn)
        @compileError("backend execute thunk does not match pinned C ABI");
    @export(&Thunks.execute, .{ .name = "TRITONBACKEND_ModelInstanceExecute" });
    exportOptional(Impl, Thunks, "initialize", "TRITONBACKEND_Initialize", *const c.TRITONZIG_BackendInitializeFn);
    exportOptional(Impl, Thunks, "finalize", "TRITONBACKEND_Finalize", *const c.TRITONZIG_BackendFinalizeFn);
    exportOptional(Impl, Thunks, "modelInitialize", "TRITONBACKEND_ModelInitialize", *const c.TRITONZIG_ModelInitializeFn);
    exportOptional(Impl, Thunks, "modelFinalize", "TRITONBACKEND_ModelFinalize", *const c.TRITONZIG_ModelFinalizeFn);
    exportOptional(Impl, Thunks, "modelInstanceInitialize", "TRITONBACKEND_ModelInstanceInitialize", *const c.TRITONZIG_ModelInstanceInitializeFn);
    exportOptional(Impl, Thunks, "modelInstanceFinalize", "TRITONBACKEND_ModelInstanceFinalize", *const c.TRITONZIG_ModelInstanceFinalizeFn);
    exportOptional(Impl, Thunks, "modelInstanceReady", "TRITONBACKEND_ModelInstanceReady", *const c.TRITONZIG_ModelInstanceReadyFn);
    exportOptional(Impl, Thunks, "getBackendAttribute", "TRITONBACKEND_GetBackendAttribute", *const c.TRITONZIG_GetBackendAttributeFn);
}

const Mock = struct {
    var created: usize = 0;
    var deleted: usize = 0;
    var sent: usize = 0;
    var released: usize = 0;
    var executed: usize = 0;
    var deleted_errors: usize = 0;
    var fail_creation: ?usize = null;
    var fail_execution: ?usize = null;
    var fail_send = false;
    var fail_release = false;
    fn reset() void {
        created = 0;
        deleted = 0;
        sent = 0;
        released = 0;
        executed = 0;
        deleted_errors = 0;
        fail_creation = null;
        fail_execution = null;
        fail_send = false;
        fail_release = false;
    }
    fn responseNew(out: *?*c.TRITONBACKEND_Response, _: *c.TRITONBACKEND_Request) ?*c.TRITONSERVER_Error {
        if (fail_creation == created) return errorToC(error.OutOfMemory);
        out.* = @ptrFromInt(0x1000 + created * 16);
        created += 1;
        return null;
    }
    fn responseDelete(_: *c.TRITONBACKEND_Response) ?*c.TRITONSERVER_Error {
        deleted += 1;
        return null;
    }
    fn responseSend(_: *c.TRITONBACKEND_Response, _: ?*c.TRITONSERVER_Error) ?*c.TRITONSERVER_Error {
        sent += 1;
        return if (fail_send) errorToC(error.Internal) else null;
    }
    fn requestRelease(_: *c.TRITONBACKEND_Request) ?*c.TRITONSERVER_Error {
        released += 1;
        return if (fail_release) errorToC(error.Internal) else null;
    }
    fn deleteError(failure: *c.TRITONSERVER_Error) void {
        deleted_errors += 1;
        c.TRITONSERVER_ErrorDelete(failure);
    }
    fn logAndDelete(failure: ?*c.TRITONSERVER_Error) void {
        if (failure) |owned| deleteError(owned);
    }
    fn execute(_: *c.TRITONBACKEND_ModelInstance, _: RequestView, _: *Response) ?*c.TRITONSERVER_Error {
        const index = executed;
        executed += 1;
        return if (fail_execution == index) errorToC(error.InvalidArg) else null;
    }
};

test "SDK rolls back all created responses before transferring request ownership" {
    Mock.reset();
    Mock.fail_creation = 2;
    const requests = [_]?*c.TRITONBACKEND_Request{ @ptrFromInt(0x100), @ptrFromInt(0x200), @ptrFromInt(0x300) };
    const failure = executeRequests(Mock, Mock, std.testing.allocator, @ptrFromInt(0x400), &requests);
    try std.testing.expectError(error.Internal, check(failure));
    try std.testing.expectEqual(@as(usize, 2), Mock.created);
    try std.testing.expectEqual(@as(usize, 2), Mock.deleted);
    try std.testing.expectEqual(@as(usize, 0), Mock.sent);
    try std.testing.expectEqual(@as(usize, 0), Mock.released);
    try std.testing.expectEqual(@as(usize, 0), Mock.executed);
}

test "SDK completes every committed request and deletes all hook/cleanup errors" {
    Mock.reset();
    Mock.fail_execution = 1;
    Mock.fail_send = true;
    Mock.fail_release = true;
    const requests = [_]?*c.TRITONBACKEND_Request{ @ptrFromInt(0x100), @ptrFromInt(0x200), @ptrFromInt(0x300) };
    try check(executeRequests(Mock, Mock, std.testing.allocator, @ptrFromInt(0x400), &requests));
    try std.testing.expectEqual(@as(usize, 3), Mock.executed);
    try std.testing.expectEqual(@as(usize, 3), Mock.sent);
    try std.testing.expectEqual(@as(usize, 3), Mock.released);
    try std.testing.expectEqual(@as(usize, 7), Mock.deleted_errors);
    try std.testing.expectEqual(@as(usize, 0), Mock.deleted);
}

test "SDK allocation failure occurs before any native ownership change" {
    Mock.reset();
    var failing: std.testing.FailingAllocator = .init(std.testing.allocator, .{ .fail_index = 0 });
    const requests = [_]?*c.TRITONBACKEND_Request{@ptrFromInt(0x100)};
    try std.testing.expectError(error.Internal, check(executeRequests(Mock, Mock, failing.allocator(), @ptrFromInt(0x400), &requests)));
    try std.testing.expectEqual(@as(usize, 0), Mock.created);
    try std.testing.expectEqual(@as(usize, 0), Mock.released);
    try std.testing.expectError(error.InvalidArg, check(executeRequests(Mock, Mock, std.testing.allocator, @ptrFromInt(0x400), &.{null})));
}

test "SDK error conversion retains explicit unsupported and validation categories" {
    try std.testing.expectError(error.Unsupported, check(errorToC(error.UnsupportedDevice)));
    try std.testing.expectError(error.InvalidArg, check(errorToC(error.ElementCountMismatch)));
    try std.testing.expectError(error.Internal, check(errorToC(error.CudaFailure)));
}
