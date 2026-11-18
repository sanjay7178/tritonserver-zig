const std = @import("std");
const t = @import("../../root.zig");
const c = t.c;
const testing = std.testing;

test "linked Triton API is compatible with the imported header" {
    var major: u32 = 0;
    var minor: u32 = 0;
    try t.check(c.TRITONSERVER_ApiVersion(&major, &minor));
    try testing.expectEqual(@as(u32, c.TRITONSERVER_API_VERSION_MAJOR), major);
    try testing.expect(minor >= c.TRITONSERVER_API_VERSION_MINOR);
}

test "linked backend API is compatible with the canonical translated header" {
    var major: u32 = 0;
    var minor: u32 = 0;
    try t.check(c.TRITONBACKEND_ApiVersion(&major, &minor));
    try testing.expectEqual(@as(u32, c.TRITONBACKEND_API_VERSION_MAJOR), major);
    try testing.expect(minor >= c.TRITONBACKEND_API_VERSION_MINOR);
}

test "check consumes each C error and maps null to success" {
    try t.check(null);
    const cases = .{
        .{ c.TRITONSERVER_ERROR_UNKNOWN, error.Unknown },
        .{ c.TRITONSERVER_ERROR_INTERNAL, error.Internal },
        .{ c.TRITONSERVER_ERROR_NOT_FOUND, error.NotFound },
        .{ c.TRITONSERVER_ERROR_INVALID_ARG, error.InvalidArg },
        .{ c.TRITONSERVER_ERROR_UNAVAILABLE, error.Unavailable },
        .{ c.TRITONSERVER_ERROR_UNSUPPORTED, error.Unsupported },
        .{ c.TRITONSERVER_ERROR_ALREADY_EXISTS, error.AlreadyExists },
        .{ c.TRITONSERVER_ERROR_CANCELLED, error.Cancelled },
    };
    inline for (cases) |case| {
        const err = c.TRITONSERVER_ErrorNew(case[0], "binding error test") orelse
            return error.ErrorAllocationFailed;
        // The message is borrowed; check frees it along with the C error.
        try testing.expectEqualStrings("binding error test", std.mem.span(c.TRITONSERVER_ErrorMessage(err)));
        try testing.expectError(case[1], t.check(err));
    }
}

test "Zig tensor and memory enums round trip through the C API" {
    try testing.expectEqual(@sizeOf(c.TRITONSERVER_DataType), @sizeOf(t.DataType));
    try testing.expectEqual(@alignOf(c.TRITONSERVER_DataType), @alignOf(t.DataType));
    const types = [_]struct { dtype: t.DataType, name: [:0]const u8, bytes: u32 }{
        .{ .dtype = .boolean, .name = "BOOL", .bytes = 1 },
        .{ .dtype = .uint8, .name = "UINT8", .bytes = 1 },
        .{ .dtype = .uint16, .name = "UINT16", .bytes = 2 },
        .{ .dtype = .uint32, .name = "UINT32", .bytes = 4 },
        .{ .dtype = .uint64, .name = "UINT64", .bytes = 8 },
        .{ .dtype = .int8, .name = "INT8", .bytes = 1 },
        .{ .dtype = .int16, .name = "INT16", .bytes = 2 },
        .{ .dtype = .int32, .name = "INT32", .bytes = 4 },
        .{ .dtype = .int64, .name = "INT64", .bytes = 8 },
        .{ .dtype = .fp16, .name = "FP16", .bytes = 2 },
        .{ .dtype = .fp32, .name = "FP32", .bytes = 4 },
        .{ .dtype = .fp64, .name = "FP64", .bytes = 8 },
        .{ .dtype = .bytes, .name = "BYTES", .bytes = 0 },
        .{ .dtype = .bf16, .name = "BF16", .bytes = 2 },
    };
    for (types) |case| {
        const raw = @backingInt(case.dtype);
        try testing.expectEqualStrings(case.name, std.mem.span(c.TRITONSERVER_DataTypeString(raw)));
        try testing.expectEqual(raw, c.TRITONSERVER_StringToDataType(case.name));
        try testing.expectEqual(case.bytes, c.TRITONSERVER_DataTypeByteSize(raw));
    }
    try testing.expectEqual(@backingInt(t.DataType.invalid), c.TRITONSERVER_StringToDataType("not-a-type"));
    try testing.expectEqual(@as(u32, 0), c.TRITONSERVER_DataTypeByteSize(@backingInt(t.DataType.invalid)));

    const memories = [_]struct { kind: t.MemoryType, name: []const u8 }{
        .{ .kind = .cpu, .name = "CPU" },
        .{ .kind = .cpu_pinned, .name = "CPU_PINNED" },
        .{ .kind = .gpu, .name = "GPU" },
    };
    for (memories) |case|
        try testing.expectEqualStrings(case.name, std.mem.span(c.TRITONSERVER_MemoryTypeString(@backingInt(case.kind))));
}

test "invalid C server options produce an owned Zig error" {
    var opts: ?*c.TRITONSERVER_ServerOptions = null;
    try t.check(c.TRITONSERVER_ServerOptionsNew(&opts));
    defer t.check(c.TRITONSERVER_ServerOptionsDelete(opts)) catch @panic("options cleanup failed");
    try testing.expectError(error.InvalidArg, t.check(c.TRITONSERVER_ServerOptionsSetModelControlMode(opts, 999)));
}

test "C server starts live and stops with an empty temporary repository" {
    var repo = testing.tmpDir(.{});
    defer repo.cleanup();
    const path = try repo.dir.realPathFileAlloc(testing.io, ".", testing.allocator);
    defer testing.allocator.free(path);

    var opts: ?*c.TRITONSERVER_ServerOptions = null;
    try t.check(c.TRITONSERVER_ServerOptionsNew(&opts));
    defer t.check(c.TRITONSERVER_ServerOptionsDelete(opts)) catch @panic("options cleanup failed");
    try t.check(c.TRITONSERVER_ServerOptionsSetModelRepositoryPath(opts, path));
    try t.check(c.TRITONSERVER_ServerOptionsSetLogInfo(opts, false));

    var server: ?*c.TRITONSERVER_Server = null;
    try t.check(c.TRITONSERVER_ServerNew(&server, opts));
    defer t.check(c.TRITONSERVER_ServerDelete(server)) catch @panic("server cleanup failed");
    var live = false;
    try t.check(c.TRITONSERVER_ServerIsLive(server, &live));
    try testing.expect(live);
    try t.check(c.TRITONSERVER_ServerStop(server));
    try testing.expectError(error.Unavailable, t.check(c.TRITONSERVER_ServerIsLive(server, &live)));
    try testing.expect(!live);
}

test "C allocator accepts Zig callbacks and opaque user context" {
    const Context = extern struct { starts: usize = 0 };
    const Callbacks = struct {
        fn allocate(
            _: ?*c.TRITONSERVER_ResponseAllocator,
            _: [*c]const u8,
            _: usize,
            _: c.TRITONSERVER_MemoryType,
            _: i64,
            _: ?*anyopaque,
            _: [*c]?*anyopaque,
            _: [*c]?*anyopaque,
            _: [*c]c.TRITONSERVER_MemoryType,
            _: [*c]i64,
        ) callconv(.c) ?*c.TRITONSERVER_Error {
            return c.TRITONSERVER_ErrorNew(c.TRITONSERVER_ERROR_UNSUPPORTED, "ABI test does not allocate tensor buffers");
        }

        fn release(_: ?*c.TRITONSERVER_ResponseAllocator, _: ?*anyopaque, _: ?*anyopaque, _: usize, _: c.TRITONSERVER_MemoryType, _: i64) callconv(.c) ?*c.TRITONSERVER_Error {
            return null;
        }

        fn start(_: ?*c.TRITONSERVER_ResponseAllocator, userp: ?*anyopaque) callconv(.c) ?*c.TRITONSERVER_Error {
            const context: *Context = @ptrCast(@alignCast(userp.?));
            context.starts += 1;
            return null;
        }
    };
    const start: c.TRITONSERVER_ResponseAllocatorStartFn_t = &Callbacks.start;
    var allocator: ?*c.TRITONSERVER_ResponseAllocator = null;
    try t.check(c.TRITONSERVER_ResponseAllocatorNew(&allocator, &Callbacks.allocate, &Callbacks.release, start));
    defer t.check(c.TRITONSERVER_ResponseAllocatorDelete(allocator)) catch @panic("allocator cleanup failed");
    // Exercise the imported C callback type without needing a loaded model.
    var context: Context = .{};
    try t.check(start.?(allocator, &context));
    try testing.expectEqual(@as(usize, 1), context.starts);
}
