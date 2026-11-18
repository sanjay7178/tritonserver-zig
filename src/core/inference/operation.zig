const std = @import("std");
const c = @import("../bindings/c.zig").raw;
const errors = @import("../bindings/error.zig");
const Request = @import("request.zig").Request;
const check = @import("request.zig").check;
const response = @import("response.zig");
const Server = @import("../server.zig").Server;

// Stable per-submission state. Callbacks use only libc allocation; caller
// allocators are accessed exclusively by the submitting/waiting thread.
const Context = struct {
    parent: std.mem.Allocator,
    io: std.Io,
    request: *Request,
    on_error: ?*const fn ([]const u8) void,
    allocator: ?*c.TRITONSERVER_ResponseAllocator = null,
    mutex: std.Io.Mutex = .init,
    condition: std.Io.Condition = .init,
    released: bool = false,
    final: bool = false,
    raw_response: ?*c.TRITONSERVER_InferenceResponse = null,
    multiple_responses: bool = false,

    fn wait(self: *Context) std.Io.Cancelable!void {
        // Upstream: core@eaff04d5d77eb1c32e98c573c90fb78b363c88c0
        // include/triton/core/tritonserver.h::
        // TRITONSERVER_InferenceResponseCompleteFn_t + RESPONSE_COMPLETE_FINAL
        // TRITONSERVER_InferenceRequestReleaseFn_t + REQUEST_RELEASE_ALL
        // This Zig adapter drains both contracts in either order, even for a
        // null FINAL. Callbacks use libc allocation, never the caller allocator.
        // Cancellation returns through PendingInference.wait, which drains these
        // same contracts before freeing Context. It does not cancel C callbacks.
        // Test: "completion context handles FINAL and RELEASE_ALL in either order".
        try self.mutex.lock(self.io);
        defer self.mutex.unlock(self.io);
        while (!self.final or !self.released)
            try self.condition.wait(self.io, &self.mutex);
    }

    fn drain(self: *Context) void {
        // core@eaff04d5d77eb1c32e98c573c90fb78b363c88c0,
        // include/triton/core/tritonserver.h:
        // TRITONSERVER_InferenceResponseCompleteFn_t (FINAL) and
        // TRITONSERVER_InferenceRequestReleaseFn_t (RELEASE_ALL) are independent
        // ownership contracts. Both events are required, including null FINAL.
        // Canceling a Zig waiter does not cancel these native callbacks.
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        while (!self.final or !self.released)
            self.condition.waitUncancelable(self.io, &self.mutex);
    }

    fn destroy(self: *Context) void {
        if (self.raw_response) |raw| errors.check(c.TRITONSERVER_InferenceResponseDelete(raw)) catch {};
        errors.check(c.TRITONSERVER_ResponseAllocatorDelete(self.allocator)) catch {};
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
    ctx.mutex.lockUncancelable(ctx.io);
    defer ctx.mutex.unlock(ctx.io);
    if (raw) |item| {
        if (ctx.raw_response == null) ctx.raw_response = item else {
            ctx.multiple_responses = true;
            errors.check(c.TRITONSERVER_InferenceResponseDelete(item)) catch {};
        }
    }
    if ((flags & c.TRITONSERVER_RESPONSE_COMPLETE_FINAL) != 0) ctx.final = true;
    ctx.condition.signal(ctx.io);
}

fn onRelease(_: ?*c.TRITONSERVER_InferenceRequest, flags: u32, userp: ?*anyopaque) callconv(.c) void {
    if ((flags & c.TRITONSERVER_REQUEST_RELEASE_ALL) == 0) return;
    const ctx: *Context = @ptrCast(@alignCast(userp.?));
    ctx.mutex.lockUncancelable(ctx.io);
    defer ctx.mutex.unlock(ctx.io);
    ctx.released = true;
    ctx.condition.signal(ctx.io);
}

/// A single-waiter operation. Keep the request at a stable address and server
/// alive until wait/deinit completes. Deinit drains accepted work; it is not
/// cancellation. Never abandon an operation, even on an application error.
/// The supplied Io backend must support synchronization from native threads;
/// std.Io.Threaded is supported. Keep it alive until wait/deinit returns.
pub const PendingInference = struct {
    context: ?*Context,

    /// Consumes the operation. Cancellation drains native work before returning
    /// Canceled; the request remains in flight throughout that drain.
    pub fn wait(self: *PendingInference) !response.InferenceResponse {
        const ctx = self.context orelse return error.InferenceAlreadyCompleted;
        self.context = null;
        defer ctx.destroy();
        defer ctx.request.in_flight = false;
        ctx.wait() catch |err| {
            // Cancellation consumes the operation, but native work must finish
            // before freeing its context, allocator, response or copied inputs.
            ctx.drain();
            return err;
        };
        if (ctx.multiple_responses) return error.UnexpectedResponseCount;
        const raw = ctx.raw_response orelse return error.NoInferenceResponse;
        return response.decode(ctx.parent, raw, ctx.on_error);
    }

    pub fn deinit(self: *PendingInference) void {
        if (self.context) |ctx| {
            ctx.drain();
            self.context = null;
            ctx.request.in_flight = false;
            ctx.destroy();
        }
    }
};

pub fn submit(server: *const Server, parent: std.mem.Allocator, io: std.Io, request: *Request) !PendingInference {
    const handle = server.handle orelse return error.ServerClosed;
    if (server.stopped) return error.ServerStopped;
    const raw_request = try request.ensureIdle();
    if (request.server_handle != handle) return error.RequestServerMismatch;
    const ctx = try parent.create(Context);
    ctx.* = .{ .parent = parent, .io = io, .request = request, .on_error = server.on_error };
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
        ctx.* = .{ .parent = std.testing.allocator, .io = std.testing.io, .request = &request, .on_error = null };
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

// Observe the actual condition wait without sleeps or assumptions about task
// scheduling. Only Context's futex operations are forwarded to the real backend.
const TestIo = struct {
    base: std.Io = std.testing.io,
    vtable: std.Io.VTable = std.Io.failing.vtable.*,
    target: *const u32,
    waiting: std.Io.Event = .unset,
    draining: std.Io.Event = .unset,

    fn io(self: *TestIo) std.Io {
        self.vtable.futexWait = wait;
        self.vtable.futexWaitUncancelable = drain;
        self.vtable.futexWake = wake;
        return .{ .userdata = self, .vtable = &self.vtable };
    }

    fn wait(userdata: ?*anyopaque, ptr: *const u32, expected: u32, timeout: std.Io.Timeout) std.Io.Cancelable!void {
        const self: *TestIo = @ptrCast(@alignCast(userdata.?));
        if (ptr == self.target) self.waiting.set(self.base);
        return self.base.vtable.futexWait(self.base.userdata, ptr, expected, timeout);
    }

    fn drain(userdata: ?*anyopaque, ptr: *const u32, expected: u32) void {
        const self: *TestIo = @ptrCast(@alignCast(userdata.?));
        if (ptr == self.target) self.draining.set(self.base);
        self.base.vtable.futexWaitUncancelable(self.base.userdata, ptr, expected);
    }

    fn wake(userdata: ?*anyopaque, ptr: *const u32, count: u32) void {
        const self: *TestIo = @ptrCast(@alignCast(userdata.?));
        self.base.vtable.futexWake(self.base.userdata, ptr, count);
    }
};

const TestCallbacks = struct {
    context: *Context,
    final_first: bool,
    first_sent: std.Io.Event = .unset,
    finish: std.Io.Event = .unset,

    fn run(self: *TestCallbacks) void {
        // A plain OS thread models callbacks owned by Triton, outside Io tasks.
        if (self.final_first)
            onResponse(null, c.TRITONSERVER_RESPONSE_COMPLETE_FINAL, self.context)
        else
            onRelease(null, c.TRITONSERVER_REQUEST_RELEASE_ALL, self.context);
        self.first_sent.set(std.testing.io);
        self.finish.waitUncancelable(std.testing.io);
        if (self.final_first)
            onRelease(null, c.TRITONSERVER_REQUEST_RELEASE_ALL, self.context)
        else
            onResponse(null, c.TRITONSERVER_RESPONSE_COMPLETE_FINAL, self.context);
    }
};

const TestWaiter = struct {
    pending: *PendingInference,
    discard: bool,
    finished: std.atomic.Value(bool) = .init(false),

    fn run(self: *TestWaiter) anyerror!void {
        defer self.finished.store(true, .release);
        if (self.discard) return self.pending.deinit();
        var result = try self.pending.wait();
        result.deinit();
    }

    fn cancel(future: *std.Io.Future(anyerror!void)) anyerror!void {
        return future.cancel(std.testing.io);
    }
};

fn testWaitFor(event: *std.Io.Event) !void {
    // Event.waitTimeout permits spurious wakes; keep one absolute deadline.
    const io = std.testing.io;
    const deadline: std.Io.Timeout = .{ .deadline = .{
        .raw = std.Io.Timestamp.now(io, .awake).addDuration(.fromSeconds(5)),
        .clock = .awake,
    } };
    while (!event.isSet()) {
        event.waitTimeout(io, deadline) catch |err| switch (err) {
            error.Timeout => if (deadline.deadline.untilNow(io).raw.nanoseconds >= 0) return err,
            error.Canceled => return err,
        };
    }
}

const TestCompletion = enum { wait, cancel, discard };

fn testNativeCompletion(mode: TestCompletion, final_first: bool) !void {
    const testing = std.testing;
    const io = testing.io;
    const OwnedArena = @import("../internal/owned_arena.zig").OwnedArena;
    var request: Request = .{
        .handle = null,
        .server_handle = undefined,
        .storage = try OwnedArena.init(testing.allocator),
        .in_flight = true,
    };
    defer request.deinit() catch @panic("request cleanup failed");
    const ctx = try testing.allocator.create(Context);
    ctx.* = .{ .parent = testing.allocator, .io = io, .request = &request, .on_error = null };
    var pending: PendingInference = .{ .context = ctx };
    defer {
        // No waiter took ownership if fixture/thread setup failed. Finish the
        // synthetic callbacks after joining any thread before draining here.
        if (pending.context) |remaining| {
            onResponse(null, c.TRITONSERVER_RESPONSE_COMPLETE_FINAL, remaining);
            onRelease(null, c.TRITONSERVER_REQUEST_RELEASE_ALL, remaining);
        }
        pending.deinit();
    }
    var probe: TestIo = .{ .target = &ctx.condition.epoch.raw };
    ctx.io = probe.io();
    try errors.check(c.TRITONSERVER_ResponseAllocatorNew(&ctx.allocator, allocate, release, null));
    var callbacks: TestCallbacks = .{ .context = ctx, .final_first = final_first };
    const callback_thread = try std.Thread.spawn(.{}, TestCallbacks.run, .{&callbacks});
    defer callback_thread.join();
    defer callbacks.finish.set(io);
    try testWaitFor(&callbacks.first_sent);

    var waiter: TestWaiter = .{ .pending = &pending, .discard = mode == .discard };
    var future = try io.concurrent(TestWaiter.run, .{&waiter});
    defer {
        callbacks.finish.set(io);
        _ = future.cancel(io) catch {};
    }
    try testWaitFor(if (mode == .discard) &probe.draining else &probe.waiting);
    try testing.expect(!waiter.finished.load(.acquire));
    try testing.expectError(error.RequestInFlight, request.deinit());

    if (mode == .cancel) {
        var cancel_task = try io.concurrent(TestWaiter.cancel, .{&future});
        defer {
            callbacks.finish.set(io);
            _ = cancel_task.cancel(io) catch {};
        }
        try testWaitFor(&probe.draining);
        // Cancellation reached the wait, but cannot release native ownership.
        try testing.expect(!waiter.finished.load(.acquire));
        try testing.expectError(error.RequestInFlight, request.deinit());
        callbacks.finish.set(io);
        try testing.expectError(error.Canceled, cancel_task.await(io));
    } else {
        callbacks.finish.set(io);
        if (mode == .discard) try future.await(io) else try testing.expectError(error.NoInferenceResponse, future.await(io));
    }
    try testing.expect(waiter.finished.load(.acquire));
    try testing.expect(!request.in_flight);
    try testing.expectError(error.InferenceAlreadyCompleted, pending.wait());
    pending.deinit();
}

test "Io waiter blocks until both native callbacks in either order" {
    for ([_]bool{ true, false }) |final_first| try testNativeCompletion(.wait, final_first);
}

test "Io cancellation drains both native callbacks before releasing ownership" {
    for ([_]bool{ true, false }) |final_first| try testNativeCompletion(.cancel, final_first);
}

test "pending deinit drains native callbacks without a cancelable wait" {
    for ([_]bool{ true, false }) |final_first| try testNativeCompletion(.discard, final_first);
}
