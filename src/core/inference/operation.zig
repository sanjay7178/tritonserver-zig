const std = @import("std");
const c = @import("../bindings/c.zig").raw;
const errors = @import("../bindings/error.zig");
const Request = @import("request.zig").Request;
const check = @import("request.zig").check;
const response = @import("response.zig");
const Server = @import("../server.zig").Server;

fn pthreadCheck(status: std.c.E) void {
    if (status != .SUCCESS) @panic("inference completion synchronization failed");
}

// Stable per-submission state. Callbacks use only libc allocation; caller
// allocators are accessed exclusively by the submitting/waiting thread.
const Context = struct {
    parent: std.mem.Allocator,
    request: *Request,
    on_error: ?*const fn ([]const u8) void,
    allocator: ?*c.TRITONSERVER_ResponseAllocator = null,
    mutex: std.c.pthread_mutex_t = .{},
    condition: std.c.pthread_cond_t = .{},
    released: bool = false,
    final: bool = false,
    raw_response: ?*c.TRITONSERVER_InferenceResponse = null,
    multiple_responses: bool = false,

    fn wait(self: *Context) void {
        pthreadCheck(std.c.pthread_mutex_lock(&self.mutex));
        while (!self.final or !self.released)
            pthreadCheck(std.c.pthread_cond_wait(&self.condition, &self.mutex));
        pthreadCheck(std.c.pthread_mutex_unlock(&self.mutex));
    }

    fn destroy(self: *Context) void {
        if (self.raw_response) |raw| errors.check(c.TRITONSERVER_InferenceResponseDelete(raw)) catch {};
        errors.check(c.TRITONSERVER_ResponseAllocatorDelete(self.allocator)) catch {};
        pthreadCheck(std.c.pthread_cond_destroy(&self.condition));
        pthreadCheck(std.c.pthread_mutex_destroy(&self.mutex));
        self.parent.destroy(self);
    }
};

fn allocate(
    _: ?*c.TRITONSERVER_ResponseAllocator,
    _: [*c]const u8,
    byte_size: usize,
    _: c.TRITONSERVER_MemoryType,
    _: i64,
    _: ?*anyopaque,
    buffer: [*c]?*anyopaque,
    buffer_userp: [*c]?*anyopaque,
    memory: [*c]c.TRITONSERVER_MemoryType,
    memory_id: [*c]i64,
) callconv(.c) ?*c.TRITONSERVER_Error {
    buffer.* = null;
    buffer_userp.* = null;
    memory.* = c.TRITONSERVER_MEMORY_CPU;
    memory_id.* = 0;
    if (byte_size != 0) {
        const bytes = std.heap.c_allocator.alignedAlloc(u8, .@"16", byte_size) catch
            return c.TRITONSERVER_ErrorNew(c.TRITONSERVER_ERROR_INTERNAL, "CPU response allocation failed");
        buffer.* = bytes.ptr;
    }
    return null;
}

fn release(_: ?*c.TRITONSERVER_ResponseAllocator, buffer: ?*anyopaque, _: ?*anyopaque, byte_size: usize, _: c.TRITONSERVER_MemoryType, _: i64) callconv(.c) ?*c.TRITONSERVER_Error {
    if (buffer) |ptr| std.heap.c_allocator.free(@as([*]align(16) u8, @ptrCast(@alignCast(ptr)))[0..byte_size]);
    return null;
}

fn onResponse(raw: ?*c.TRITONSERVER_InferenceResponse, flags: u32, userp: ?*anyopaque) callconv(.c) void {
    const ctx: *Context = @ptrCast(@alignCast(userp.?));
    pthreadCheck(std.c.pthread_mutex_lock(&ctx.mutex));
    defer pthreadCheck(std.c.pthread_mutex_unlock(&ctx.mutex));
    if (raw) |item| {
        if (ctx.raw_response == null) ctx.raw_response = item else {
            ctx.multiple_responses = true;
            errors.check(c.TRITONSERVER_InferenceResponseDelete(item)) catch {};
        }
    }
    if ((flags & c.TRITONSERVER_RESPONSE_COMPLETE_FINAL) != 0) ctx.final = true;
    pthreadCheck(std.c.pthread_cond_signal(&ctx.condition));
}

fn onRelease(_: ?*c.TRITONSERVER_InferenceRequest, flags: u32, userp: ?*anyopaque) callconv(.c) void {
    if ((flags & c.TRITONSERVER_REQUEST_RELEASE_ALL) == 0) return;
    const ctx: *Context = @ptrCast(@alignCast(userp.?));
    pthreadCheck(std.c.pthread_mutex_lock(&ctx.mutex));
    defer pthreadCheck(std.c.pthread_mutex_unlock(&ctx.mutex));
    ctx.released = true;
    pthreadCheck(std.c.pthread_cond_signal(&ctx.condition));
}

/// A single-waiter operation. Keep the request at a stable address and server
/// alive until wait/deinit completes. Deinit drains accepted work; it is not
/// cancellation. Never abandon an operation, even on an application error.
pub const PendingInference = struct {
    context: ?*Context,

    pub fn wait(self: *PendingInference) !response.InferenceResponse {
        const ctx = self.context orelse return error.InferenceAlreadyCompleted;
        ctx.wait();
        self.context = null;
        defer ctx.destroy();
        defer ctx.request.in_flight = false;
        if (ctx.multiple_responses) return error.UnexpectedResponseCount;
        const raw = ctx.raw_response orelse return error.NoInferenceResponse;
        return response.decode(ctx.parent, raw, ctx.on_error);
    }

    pub fn deinit(self: *PendingInference) void {
        if (self.context) |ctx| {
            ctx.wait();
            self.context = null;
            ctx.request.in_flight = false;
            ctx.destroy();
        }
    }
};

pub fn submit(server: *const Server, parent: std.mem.Allocator, request: *Request) !PendingInference {
    const handle = server.handle orelse return error.ServerClosed;
    if (server.stopped) return error.ServerStopped;
    const raw_request = try request.ensureIdle();
    if (request.server_handle != handle) return error.RequestServerMismatch;
    const ctx = try parent.create(Context);
    ctx.* = .{ .parent = parent, .request = request, .on_error = server.on_error };
    errdefer ctx.destroy();
    try check(c.TRITONSERVER_ResponseAllocatorNew(&ctx.allocator, allocate, release, null), server.on_error);
    try check(c.TRITONSERVER_InferenceRequestSetReleaseCallback(raw_request, onRelease, ctx), server.on_error);
    try check(c.TRITONSERVER_InferenceRequestSetResponseCallback(raw_request, ctx.allocator, ctx, onResponse, ctx), server.on_error);
    request.in_flight = true;
    // Rejected submissions retain ownership and must not wait for callbacks.
    check(c.TRITONSERVER_ServerInferAsync(handle, raw_request, null), server.on_error) catch |err| {
        request.in_flight = false;
        return err;
    };
    return .{ .context = ctx };
}

test "completion context handles FINAL and RELEASE_ALL in either order" {
    const OwnedArena = @import("../internal/owned_arena.zig").OwnedArena;
    for ([_]bool{ true, false }) |final_first| {
        var request: Request = .{
            .handle = null,
            .server_handle = undefined,
            .storage = try OwnedArena.init(std.testing.allocator),
            .in_flight = true,
        };
        defer request.deinit() catch @panic("request cleanup failed");
        const ctx = try std.testing.allocator.create(Context);
        ctx.* = .{ .parent = std.testing.allocator, .request = &request, .on_error = null };
        var pending: PendingInference = .{ .context = ctx };
        defer pending.deinit();
        try errors.check(c.TRITONSERVER_ResponseAllocatorNew(&ctx.allocator, allocate, release, null));
        // RESCHEDULE does not return complete ownership.
        onRelease(null, c.TRITONSERVER_REQUEST_RELEASE_RESCHEDULE, ctx);
        try std.testing.expect(!ctx.released);
        if (final_first) {
            onResponse(null, c.TRITONSERVER_RESPONSE_COMPLETE_FINAL, ctx);
            try std.testing.expect(ctx.final and !ctx.released);
            onRelease(null, c.TRITONSERVER_REQUEST_RELEASE_ALL, ctx);
        } else {
            onRelease(null, c.TRITONSERVER_REQUEST_RELEASE_ALL, ctx);
            try std.testing.expect(!ctx.final and ctx.released);
            onResponse(null, c.TRITONSERVER_RESPONSE_COMPLETE_FINAL, ctx);
        }
        try std.testing.expectError(error.NoInferenceResponse, pending.wait());
        try std.testing.expect(!request.in_flight);
        try std.testing.expectError(error.InferenceAlreadyCompleted, pending.wait());
    }
}

test "CPU allocator handles zero-length outputs and overrides memory preference" {
    var buffer: ?*anyopaque = undefined;
    var userp: ?*anyopaque = undefined;
    var memory: c.TRITONSERVER_MemoryType = undefined;
    var memory_id: i64 = undefined;
    try errors.check(allocate(null, "OUT0", 0, c.TRITONSERVER_MEMORY_GPU, 7, null, &buffer, &userp, &memory, &memory_id));
    try std.testing.expect(buffer == null and userp == null);
    try std.testing.expectEqual(@as(c.TRITONSERVER_MemoryType, c.TRITONSERVER_MEMORY_CPU), memory);
    try std.testing.expectEqual(@as(i64, 0), memory_id);
    try errors.check(release(null, buffer, userp, 0, memory, memory_id));
    try errors.check(allocate(null, "OUT0", 17, c.TRITONSERVER_MEMORY_GPU, 7, null, &buffer, &userp, &memory, &memory_id));
    try std.testing.expect(buffer != null);
    try errors.check(release(null, buffer, userp, 17, memory, memory_id));
}
