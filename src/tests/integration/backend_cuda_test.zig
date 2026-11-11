//! Explicitly opted-in, real-device tests. These are never imported by CPU CI.
const std = @import("std");
const t = @import("tritonserver_zig");
const engine = @import("backend_cuda");
const cuda = @import("cuda_c");
const c = t.c;
const b = t.backend;
const alloc = std.heap.c_allocator;
const testing = std.testing;

fn pthreadCheck(status: std.c.E) void {
    if (status != .SUCCESS) @panic("raw CUDA test synchronization failed");
}

const Allocation = struct {
    ptr: ?*anyopaque,
    byte_size: usize,
    memory_type: c.TRITONSERVER_MemoryType,

    fn init(memory: c.TRITONSERVER_MemoryType, bytes: usize) !Allocation {
        var result: Allocation = .{ .ptr = null, .byte_size = bytes, .memory_type = memory };
        if (bytes == 0) return result;
        switch (memory) {
            c.TRITONSERVER_MEMORY_GPU => {
                try engine.check(cuda.cudaSetDevice(0));
                try engine.check(cuda.cudaMalloc(&result.ptr, bytes));
            },
            c.TRITONSERVER_MEMORY_CPU_PINNED => try engine.check(cuda.cudaHostAlloc(&result.ptr, bytes, cuda.cudaHostAllocDefault)),
            c.TRITONSERVER_MEMORY_CPU => result.ptr = (try alloc.alignedAlloc(u8, .@"16", bytes)).ptr,
            else => return error.UnsupportedMemoryType,
        }
        return result;
    }

    fn deinit(self: *Allocation) void {
        const ptr = self.ptr orelse return;
        switch (self.memory_type) {
            c.TRITONSERVER_MEMORY_GPU => {
                engine.check(cuda.cudaSetDevice(0)) catch @panic("select device for free");
                engine.check(cuda.cudaFree(ptr)) catch @panic("free CUDA allocation");
            },
            c.TRITONSERVER_MEMORY_CPU_PINNED => engine.check(cuda.cudaFreeHost(ptr)) catch @panic("free pinned allocation"),
            c.TRITONSERVER_MEMORY_CPU => alloc.free(@as([*]align(16) u8, @ptrCast(@alignCast(ptr)))[0..self.byte_size]),
            else => unreachable,
        }
        self.ptr = null;
    }

    fn write(self: *Allocation, values: []const i32) !void {
        const bytes = std.mem.sliceAsBytes(values);
        if (bytes.len != self.byte_size) return error.BufferSizeMismatch;
        if (self.memory_type == c.TRITONSERVER_MEMORY_GPU) {
            try engine.check(cuda.cudaMemcpy(self.ptr, bytes.ptr, bytes.len, cuda.cudaMemcpyHostToDevice));
            // Pageable H2D cudaMemcpy may return once staging is complete. The
            // backend uses an independent nonblocking stream, so the caller
            // must explicitly finish the upload before publishing its input.
            try engine.check(cuda.cudaStreamSynchronize(null));
        } else if (bytes.len != 0) {
            @memcpy(@as([*]u8, @ptrCast(self.ptr.?))[0..bytes.len], bytes);
        }
    }

    fn view(self: Allocation) b.ConstBuffer {
        return .{ .ptr = self.ptr, .byte_size = self.byte_size, .memory_type = self.memory_type, .memory_type_id = 0 };
    }
};

const RawContext = struct {
    mutex: std.c.pthread_mutex_t = .{},
    condition: std.c.pthread_cond_t = .{},
    request: ?*c.TRITONSERVER_InferenceRequest = null,
    allocator: ?*c.TRITONSERVER_ResponseAllocator = null,
    response: ?*c.TRITONSERVER_InferenceResponse = null,
    accepted: bool = false,
    final: bool = false,
    released: bool = false,
    final_count: usize = 0,
    release_count: usize = 0,
    output_memory: c.TRITONSERVER_MemoryType,
    output_device: i64 = 0,
    allocations: std.atomic.Value(usize) = .init(0),
    frees: std.atomic.Value(usize) = .init(0),

    fn wait(self: *RawContext) void {
        if (!self.accepted) return;
        pthreadCheck(std.c.pthread_mutex_lock(&self.mutex));
        while (!self.final or !self.released)
            pthreadCheck(std.c.pthread_cond_wait(&self.condition, &self.mutex));
        pthreadCheck(std.c.pthread_mutex_unlock(&self.mutex));
    }
};

fn allocate(
    _: ?*c.TRITONSERVER_ResponseAllocator,
    _: [*c]const u8,
    byte_size: usize,
    _: c.TRITONSERVER_MemoryType,
    _: i64,
    userp: ?*anyopaque,
    buffer: [*c]?*anyopaque,
    buffer_userp: [*c]?*anyopaque,
    memory: [*c]c.TRITONSERVER_MemoryType,
    memory_id: [*c]i64,
) callconv(.c) ?*c.TRITONSERVER_Error {
    const ctx: *RawContext = @ptrCast(@alignCast(userp.?));
    buffer.* = null;
    buffer_userp.* = ctx;
    memory.* = ctx.output_memory;
    memory_id.* = ctx.output_device;
    const allocation = Allocation.init(ctx.output_memory, byte_size) catch
        return c.TRITONSERVER_ErrorNew(c.TRITONSERVER_ERROR_INTERNAL, "test output allocation failed");
    buffer.* = allocation.ptr;
    if (allocation.ptr != null) _ = ctx.allocations.fetchAdd(1, .monotonic);
    return null;
}

fn release(
    _: ?*c.TRITONSERVER_ResponseAllocator,
    buffer: ?*anyopaque,
    userp: ?*anyopaque,
    bytes: usize,
    _: c.TRITONSERVER_MemoryType,
    _: i64,
) callconv(.c) ?*c.TRITONSERVER_Error {
    const ctx: *RawContext = @ptrCast(@alignCast(userp.?));
    var allocation: Allocation = .{ .ptr = buffer, .byte_size = bytes, .memory_type = ctx.output_memory };
    allocation.deinit();
    if (buffer != null) _ = ctx.frees.fetchAdd(1, .monotonic);
    return null;
}

fn onResponse(raw: ?*c.TRITONSERVER_InferenceResponse, flags: u32, userp: ?*anyopaque) callconv(.c) void {
    const ctx: *RawContext = @ptrCast(@alignCast(userp.?));
    pthreadCheck(std.c.pthread_mutex_lock(&ctx.mutex));
    defer pthreadCheck(std.c.pthread_mutex_unlock(&ctx.mutex));
    if (raw) |response| {
        if (ctx.response != null) @panic("unexpected second coupled response");
        ctx.response = response;
    }
    if ((flags & c.TRITONSERVER_RESPONSE_COMPLETE_FINAL) != 0) {
        ctx.final = true;
        ctx.final_count += 1;
    }
    pthreadCheck(std.c.pthread_cond_signal(&ctx.condition));
}

fn onRelease(_: ?*c.TRITONSERVER_InferenceRequest, flags: u32, userp: ?*anyopaque) callconv(.c) void {
    if ((flags & c.TRITONSERVER_REQUEST_RELEASE_ALL) == 0) return;
    const ctx: *RawContext = @ptrCast(@alignCast(userp.?));
    pthreadCheck(std.c.pthread_mutex_lock(&ctx.mutex));
    defer pthreadCheck(std.c.pthread_mutex_unlock(&ctx.mutex));
    ctx.released = true;
    ctx.release_count += 1;
    pthreadCheck(std.c.pthread_cond_signal(&ctx.condition));
}

const RawResult = struct {
    context: *RawContext,

    fn submit(server: *const t.Server, inputs: []const b.ConstBuffer, memory: c.TRITONSERVER_MemoryType, device: i64) !RawResult {
        const ctx = try alloc.create(RawContext);
        ctx.* = .{ .output_memory = memory, .output_device = device };
        var result: RawResult = .{ .context = ctx };
        errdefer result.deinit();
        try t.check(c.TRITONSERVER_InferenceRequestNew(&ctx.request, server.handle, "phase5_cuda", 1));
        const shape = [_]i64{ 1, 4 };
        try t.check(c.TRITONSERVER_InferenceRequestAddInput(ctx.request, "IN0", c.TRITONSERVER_TYPE_INT32, &shape, shape.len));
        for (inputs) |input| {
            try t.check(c.TRITONSERVER_InferenceRequestAppendInputData(ctx.request, "IN0", input.ptr, input.byte_size, input.memory_type, input.memory_type_id));
        }
        try t.check(c.TRITONSERVER_InferenceRequestAddRequestedOutput(ctx.request, "OUT0"));
        try t.check(c.TRITONSERVER_ResponseAllocatorNew(&ctx.allocator, allocate, release, null));
        try t.check(c.TRITONSERVER_InferenceRequestSetReleaseCallback(ctx.request, onRelease, ctx));
        try t.check(c.TRITONSERVER_InferenceRequestSetResponseCallback(ctx.request, ctx.allocator, ctx, onResponse, ctx));
        try t.check(c.TRITONSERVER_ServerInferAsync(server.handle, ctx.request, null));
        ctx.accepted = true;
        ctx.wait();
        return result;
    }

    fn closeResponse(self: *RawResult) !void {
        if (self.context.response) |response| {
            try t.check(c.TRITONSERVER_InferenceResponseDelete(response));
            self.context.response = null;
        }
    }

    fn deinit(self: *RawResult) void {
        const ctx = self.context;
        ctx.wait();
        self.closeResponse() catch @panic("raw response cleanup failed");
        if (ctx.request) |request| t.check(c.TRITONSERVER_InferenceRequestDelete(request)) catch @panic("raw request cleanup failed");
        if (ctx.allocator) |allocator| t.check(c.TRITONSERVER_ResponseAllocatorDelete(allocator)) catch @panic("raw allocator cleanup failed");
        pthreadCheck(std.c.pthread_cond_destroy(&ctx.condition));
        pthreadCheck(std.c.pthread_mutex_destroy(&ctx.mutex));
        alloc.destroy(ctx);
    }

    fn expectOwnership(self: *const RawResult) !void {
        try testing.expectEqual(@as(usize, 1), self.context.final_count);
        try testing.expectEqual(@as(usize, 1), self.context.release_count);
    }

    fn expectInputTag(self: *const RawResult, memory: c.TRITONSERVER_MemoryType, id: i64, buffers: i64) !void {
        try testing.expectEqual(@as(i64, @intCast(memory)), try self.parameter("input_memory_type"));
        try testing.expectEqual(id, try self.parameter("input_memory_type_id"));
        try testing.expectEqual(buffers, try self.parameter("input_buffer_count"));
    }

    fn parameter(self: *const RawResult, expected: []const u8) !i64 {
        const response = self.context.response orelse return error.MissingResponse;
        var count: u32 = 0;
        try t.check(c.TRITONSERVER_InferenceResponseParameterCount(response, &count));
        for (0..count) |index| {
            var name: [*c]const u8 = null;
            var kind: c.TRITONSERVER_ParameterType = 0;
            var value: ?*const anyopaque = null;
            try t.check(c.TRITONSERVER_InferenceResponseParameter(response, @intCast(index), &name, &kind, &value));
            if (name != null and std.mem.eql(u8, expected, std.mem.span(name))) {
                if (kind != c.TRITONSERVER_PARAMETER_INT or value == null) return error.InvalidParameter;
                return @as(*const i64, @ptrCast(@alignCast(value.?))).*;
            }
        }
        return error.MissingParameter;
    }

    fn expectOutput(self: *const RawResult, memory: c.TRITONSERVER_MemoryType, expected: []const i32) !void {
        const response = self.context.response orelse return error.MissingResponse;
        try t.check(c.TRITONSERVER_InferenceResponseError(response));
        var count: u32 = 0;
        try t.check(c.TRITONSERVER_InferenceResponseOutputCount(response, &count));
        try testing.expectEqual(@as(u32, 1), count);
        var name: [*c]const u8 = null;
        var datatype: c.TRITONSERVER_DataType = 0;
        var shape: [*c]const i64 = null;
        var dimensions: u64 = 0;
        var buffer: ?*const anyopaque = null;
        var bytes: usize = 0;
        var actual_memory: c.TRITONSERVER_MemoryType = 0;
        var device: i64 = -1;
        var userp: ?*anyopaque = null;
        try t.check(c.TRITONSERVER_InferenceResponseOutput(response, 0, &name, &datatype, &shape, &dimensions, &buffer, &bytes, &actual_memory, &device, &userp));
        try testing.expectEqualStrings("OUT0", std.mem.span(name));
        try testing.expectEqual(@as(c.TRITONSERVER_DataType, c.TRITONSERVER_TYPE_INT32), datatype);
        try testing.expectEqual(@as(u64, 2), dimensions);
        try testing.expectEqualSlices(i64, &.{ 1, 4 }, shape[0..2]);
        try testing.expectEqual(memory, actual_memory);
        try testing.expectEqual(@as(i64, 0), device);
        try testing.expectEqual(expected.len * @sizeOf(i32), bytes);
        try testing.expect(buffer != null);
        var actual: [4]i32 = undefined;
        if (memory == c.TRITONSERVER_MEMORY_GPU) {
            try engine.check(cuda.cudaSetDevice(0));
            try engine.check(cuda.cudaMemcpy(&actual, buffer, bytes, cuda.cudaMemcpyDeviceToHost));
        } else {
            @memcpy(std.mem.sliceAsBytes(&actual), @as([*]const u8, @ptrCast(buffer.?))[0..bytes]);
        }
        try testing.expectEqualSlices(i32, expected, &actual);
    }

    fn expectFailure(self: *const RawResult) !void {
        const response = self.context.response orelse return error.MissingResponse;
        const failure = c.TRITONSERVER_InferenceResponseError(response) orelse return error.ExpectedInferenceFailure;
        defer c.TRITONSERVER_ErrorDelete(failure);
        try testing.expect(c.TRITONSERVER_ErrorCode(failure) != c.TRITONSERVER_ERROR_UNKNOWN);
        try self.expectOwnership();
    }

    fn expectFreed(self: *RawResult) !void {
        try self.expectOwnership();
        try self.closeResponse();
        try testing.expectEqual(self.context.allocations.load(.acquire), self.context.frees.load(.acquire));
    }
};

fn startServer() !t.Server {
    return t.Server.init(alloc, .{
        .model_repository = "tests/fixtures/backend_cuda_models",
        .backend_directory = ".zig-cache/phase5/backends",
        .log_info = false,
    });
}

test "raw GPU input and output stay device resident across fragmented buffers" {
    var server = try startServer();
    defer server.deinit() catch @panic("server cleanup");
    const expected = [_]i32{ -7, 19, std.math.minInt(i32), std.math.maxInt(i32) };
    var input = try Allocation.init(c.TRITONSERVER_MEMORY_GPU, @sizeOf(@TypeOf(expected)));
    defer input.deinit();
    try input.write(&expected);
    const parts = [_]b.ConstBuffer{ try input.view().slice(0, 8), try input.view().slice(8, 8) };
    var result = try RawResult.submit(&server, &parts, c.TRITONSERVER_MEMORY_GPU, 0);
    defer result.deinit();
    try result.expectInputTag(c.TRITONSERVER_MEMORY_GPU, 0, 2);
    input.deinit(); // The independent GPU output must outlive the input allocation.
    try result.expectOutput(c.TRITONSERVER_MEMORY_GPU, &expected);
    try result.expectFreed();
}

test "raw CPU and pinned inputs copy into GPU allocator outputs" {
    var server = try startServer();
    defer server.deinit() catch @panic("server cleanup");
    const expected = [_]i32{ 3, -4, 9, 123456 };
    for ([_]c.TRITONSERVER_MemoryType{ c.TRITONSERVER_MEMORY_CPU, c.TRITONSERVER_MEMORY_CPU_PINNED }) |memory| {
        var input = try Allocation.init(memory, @sizeOf(@TypeOf(expected)));
        defer input.deinit();
        try input.write(&expected);
        var result = try RawResult.submit(&server, &.{input.view()}, c.TRITONSERVER_MEMORY_GPU, 0);
        defer result.deinit();
        try result.expectInputTag(memory, 0, 1);
        try result.expectOutput(c.TRITONSERVER_MEMORY_GPU, &expected);
        try result.expectFreed();
    }
}

test "raw GPU inputs copy into CPU and pinned allocator outputs" {
    var server = try startServer();
    defer server.deinit() catch @panic("server cleanup");
    const expected = [_]i32{ 8, 4, 2, 1 };
    var input = try Allocation.init(c.TRITONSERVER_MEMORY_GPU, @sizeOf(@TypeOf(expected)));
    defer input.deinit();
    try input.write(&expected);
    for ([_]c.TRITONSERVER_MemoryType{ c.TRITONSERVER_MEMORY_CPU, c.TRITONSERVER_MEMORY_CPU_PINNED }) |memory| {
        var result = try RawResult.submit(&server, &.{input.view()}, memory, 0);
        defer result.deinit();
        try result.expectInputTag(c.TRITONSERVER_MEMORY_GPU, 0, 1);
        try result.expectOutput(memory, &expected);
        try result.expectFreed();
    }
}

test "raw foreign GPU memory IDs fail with complete response and allocation cleanup" {
    var server = try startServer();
    defer server.deinit() catch @panic("server cleanup");
    const expected = [_]i32{ 1, 2, 3, 4 };
    var input = try Allocation.init(c.TRITONSERVER_MEMORY_GPU, @sizeOf(@TypeOf(expected)));
    defer input.deinit();
    try input.write(&expected);
    var foreign = input.view();
    foreign.memory_type_id = 1;
    var bad_input = try RawResult.submit(&server, &.{foreign}, c.TRITONSERVER_MEMORY_GPU, 0);
    defer bad_input.deinit();
    try bad_input.expectFailure();
    try bad_input.expectFreed();
    var bad_output = try RawResult.submit(&server, &.{input.view()}, c.TRITONSERVER_MEMORY_GPU, 1);
    defer bad_output.deinit();
    try bad_output.expectFailure();
    try bad_output.expectFreed();
}

test "raw CUDA copy errors drain earlier fragments before response and request cleanup" {
    var server = try startServer();
    defer server.deinit() catch @panic("server cleanup");
    const expected = [_]i32{ 17, 18, 19, 20 };
    var input = try Allocation.init(c.TRITONSERVER_MEMORY_GPU, @sizeOf(@TypeOf(expected)));
    defer input.deinit();
    try input.write(&expected);
    const mislabeled_host = [_]i32{ 19, 20 };
    const fragments = [_]b.ConstBuffer{
        try input.view().slice(0, 8),
        .{ .ptr = &mislabeled_host, .byte_size = 8, .memory_type = c.TRITONSERVER_MEMORY_GPU, .memory_type_id = 0 },
    };
    var result = try RawResult.submit(&server, &fragments, c.TRITONSERVER_MEMORY_GPU, 0);
    defer result.deinit();
    try result.expectFailure();
    try testing.expectEqual(@as(usize, 1), result.context.allocations.load(.acquire));
    try result.expectFreed();
    // The failed copy must not poison the instance's stream or retain ownership.
    var retry = try RawResult.submit(&server, &.{input.view()}, c.TRITONSERVER_MEMORY_GPU, 0);
    defer retry.deinit();
    try retry.expectOutput(c.TRITONSERVER_MEMORY_GPU, &expected);
    try retry.expectFreed();
}
