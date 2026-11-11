const std = @import("std");
const t = @import("../../root.zig");
const testing = std.testing;
const c = t.c;
const repository = "tests/fixtures/backend_models/valid";
const backend_directory = ".zig-cache/phase5/backends";

fn startServer() !t.Server {
    return t.Server.init(testing.allocator, .{
        .model_repository = repository,
        .backend_directory = backend_directory,
        .log_info = false,
    });
}

fn destroyServer(server: *t.Server) void {
    server.deinit() catch @panic("backend test server cleanup failed");
}

fn destroyRequest(request: *t.Request) void {
    request.deinit() catch @panic("backend test request cleanup failed");
}

fn expectBytes(response: *const t.InferenceResponse, datatype: t.DataType, shape: []const i64, bytes: []const u8) !void {
    try testing.expectEqual(@as(usize, 1), response.outputs.len);
    const output = try response.output("OUT0");
    try testing.expectEqual(datatype, output.datatype);
    try testing.expectEqualSlices(i64, shape, output.shape);
    try testing.expectEqualSlices(u8, bytes, output.data);
}

test "Zig backend library exports ABI checked execution and lifecycle hooks" {
    var library = try t.backend.PluginLibrary.load(backend_directory ++ "/zig_passthrough/libtriton_zig_passthrough.so");
    defer library.deinit();
    try testing.expect(library.vtable.initialize != null);
    try testing.expect(library.vtable.model_initialize != null);
    try testing.expect(library.vtable.model_finalize != null);
    try testing.expect(library.vtable.model_instance_initialize != null);
    try testing.expect(library.vtable.model_instance_finalize != null);
    try testing.expect(library.vtable.finalize == null);
}

test "Zig backend preserves bytes for every supported fixed width datatype" {
    var server = try startServer();
    defer destroyServer(&server);
    const Case = struct { model: []const u8, datatype: t.DataType, width: usize };
    const cases = [_]Case{
        .{ .model = "zig_bool", .datatype = .boolean, .width = 1 },
        .{ .model = "zig_u8", .datatype = .uint8, .width = 1 },
        .{ .model = "zig_u16", .datatype = .uint16, .width = 2 },
        .{ .model = "zig_u32", .datatype = .uint32, .width = 4 },
        .{ .model = "zig_u64", .datatype = .uint64, .width = 8 },
        .{ .model = "zig_i8", .datatype = .int8, .width = 1 },
        .{ .model = "zig_i16", .datatype = .int16, .width = 2 },
        .{ .model = "zig_i32", .datatype = .int32, .width = 4 },
        .{ .model = "zig_i64", .datatype = .int64, .width = 8 },
        .{ .model = "zig_fp16", .datatype = .fp16, .width = 2 },
        .{ .model = "zig_fp32", .datatype = .fp32, .width = 4 },
        .{ .model = "zig_fp64", .datatype = .fp64, .width = 8 },
        .{ .model = "zig_bf16", .datatype = .bf16, .width = 2 },
    };
    for (cases) |case| {
        var request = try t.Request.init(testing.allocator, &server, case.model, 1);
        defer destroyRequest(&request);
        var buffer: [32]u8 = undefined;
        const bytes = buffer[0 .. 4 * case.width];
        for (bytes, 0..) |*byte, index| byte.* = @truncate(index * 37 + 129);
        if (case.datatype == .boolean) @memcpy(bytes, &[_]u8{ 0, 1, 1, 0 });
        // Raw comparisons also cover signed bit patterns and floating payloads;
        // the backend must not reinterpret, round, or normalize these bytes.
        try request.addInputBytes(case.datatype, "IN0", &.{4}, bytes);
        try request.requestOutput("OUT0");
        var response = try server.inferSync(testing.allocator, &request);
        defer response.deinit();
        try expectBytes(&response, case.datatype, &.{4}, bytes);
    }
}

test "Zig backend handles rank two inputs and output ownership survives server deletion" {
    var server = try startServer();
    defer destroyServer(&server);
    var request = try t.Request.init(testing.allocator, &server, "zig_nonbatch", 1);
    defer destroyRequest(&request);
    var values = [_]i32{ 31, -17, std.math.minInt(i32), std.math.maxInt(i32), 0, 8 };
    const expected = values;
    try request.addInput(i32, "IN0", &.{ 2, 3 }, &values);
    @memset(&values, 0);
    var response = try server.inferSync(testing.allocator, &request);
    defer response.deinit();
    try request.deinit();
    try server.stop();
    try server.deinit();
    try expectBytes(&response, .int32, &.{ 2, 3 }, std.mem.sliceAsBytes(&expected));
}

test "Zig backend concatenates raw CPU and pinned input segments without assuming alignment" {
    var server = try startServer();
    defer destroyServer(&server);
    var request = try t.Request.init(testing.allocator, &server, "zig_nonbatch", 1);
    defer destroyRequest(&request);
    const values = [_]i32{ 1, -2, 303, -404, 50505, -60606 };
    const bytes = std.mem.sliceAsBytes(&values);
    try request.addInput(i32, "IN0", &.{ 2, 3 }, &values);
    const handle = try request.ensureIdle();
    try t.check(c.TRITONSERVER_InferenceRequestRemoveAllInputData(handle, "IN0"));
    // A split inside an INT32 element rules out accidentally treating each
    // segment as an aligned typed tensor. These buffers stay alive until wait.
    try t.check(c.TRITONSERVER_InferenceRequestAppendInputData(handle, "IN0", bytes.ptr, 3, c.TRITONSERVER_MEMORY_CPU, 0));
    try t.check(c.TRITONSERVER_InferenceRequestAppendInputData(handle, "IN0", bytes[3..].ptr, bytes.len - 3, c.TRITONSERVER_MEMORY_CPU_PINNED, 0));
    var response = try server.inferSync(testing.allocator, &request);
    defer response.deinit();
    try expectBytes(&response, .int32, &.{ 2, 3 }, bytes);
}

test "Zig backend processes independent native dynamically batched requests" {
    var server = try startServer();
    defer destroyServer(&server);
    var requests: [8]t.Request = undefined;
    var pending: [8]t.PendingInference = undefined;
    var created: usize = 0;
    var submitted: usize = 0;
    defer {
        for (pending[0..submitted]) |*operation| operation.deinit();
        for (requests[0..created]) |*request| destroyRequest(request);
    }
    var values: [8][4]i32 = undefined;
    for (&requests, &pending, &values, 0..) |*request, *operation, *row, index| {
        const base: i32 = @intCast(index * 100);
        row.* = .{ base + 1, base + 2, -base - 3, base + 4 };
        request.* = try t.Request.init(testing.allocator, &server, "zig_native_dynamic", 1);
        created += 1;
        try request.addInput(i32, "IN0", &.{ 1, 4 }, row);
        operation.* = try server.inferAsync(testing.allocator, request);
        submitted += 1;
    }
    for (&pending, values) |*operation, row| {
        var response = try operation.wait();
        defer response.deinit();
        try expectBytes(&response, .int32, &.{ 1, 4 }, std.mem.sliceAsBytes(&row));
    }
}

test "Zig scheduler batches B1 B3 B4 through the Zig backend and restores caller shapes" {
    var server = try startServer();
    defer destroyServer(&server);
    var scheduler = try t.Scheduler.init(testing.allocator, &server, "zig_fixed_batch", 1, .{
        .max_batch_size = 8,
        .max_wait_us = 1_000_000,
        .worker_count = 1,
    });
    defer scheduler.deinit();
    var inputs: [32]i32 = undefined;
    for (&inputs, 0..) |*value, index| value.* = @as(i32, @intCast(index)) * 7 - 91;
    const counts = [_]usize{ 1, 3, 4 };
    var futures: [counts.len]t.ScheduledInference = undefined;
    var submitted: usize = 0;
    defer for (futures[0..submitted]) |*future| future.deinit();
    var offset: usize = 0;
    for (&futures, counts) |*future, count| {
        const shape = [_]i64{ @intCast(count), 4 };
        const input = try t.InputTensor.fromSlice(i32, "IN0", &shape, inputs[offset * 4 ..][0 .. count * 4]);
        future.* = try scheduler.submit(testing.allocator, &.{input}, .{});
        submitted += 1;
        offset += count;
    }
    scheduler.shutdown();
    const stats = scheduler.stats();
    try testing.expectEqual(@as(u64, 1), stats.dispatched_batches);
    try testing.expectEqual(@as(u64, 8), stats.dispatched_samples);
    try testing.expectEqual(@as(u64, 0), stats.failed_requests);
    offset = 0;
    for (&futures, counts) |*future, count| {
        var response = try future.wait();
        defer response.deinit();
        const shape = [_]i64{ @intCast(count), 4 };
        try expectBytes(&response, .int32, &shape, std.mem.sliceAsBytes(inputs[offset * 4 ..][0 .. count * 4]));
        offset += count;
    }
}

const ConcurrentCaller = struct {
    server: *const t.Server,
    index: usize,
    failure: ?anyerror = null,

    fn run(self: *ConcurrentCaller) void {
        self.perform() catch |err| {
            self.failure = err;
        };
    }

    fn perform(self: *ConcurrentCaller) !void {
        const alloc = std.heap.c_allocator;
        for (0..12) |iteration| {
            const inner = 1 + (self.index + iteration) % 3;
            const shape = [_]i64{ 1, @intCast(inner), 2 };
            var values: [6]f32 = undefined;
            for (values[0 .. inner * 2], 0..) |*value, column|
                value.* = @floatFromInt(self.index * 1000 + iteration * 10 + column);
            var request = try t.Request.init(alloc, self.server, "zig_multi", 1);
            defer destroyRequest(&request);
            try request.addInput(f32, "IN0", &shape, values[0 .. inner * 2]);
            var response = try self.server.inferSync(alloc, &request);
            defer response.deinit();
            const output = try response.output("OUT0");
            if (!std.mem.eql(i64, &shape, output.shape) or
                !std.mem.eql(f32, values[0 .. inner * 2], try output.asSlice(f32)))
                return error.ConcurrentOutputMismatch;
        }
    }
};

test "Zig backend keeps instance copy states independent across concurrent variable shapes" {
    var server = try startServer();
    defer destroyServer(&server);
    var callers: [8]ConcurrentCaller = undefined;
    var threads: [callers.len]std.Thread = undefined;
    var started: usize = 0;
    defer for (threads[0..started]) |thread| thread.join();
    for (&callers, 0..) |*caller, index| {
        caller.* = .{ .server = &server, .index = index };
        threads[started] = try std.Thread.spawn(.{}, ConcurrentCaller.run, .{caller});
        started += 1;
    }
    for (threads[0..started]) |thread| thread.join();
    started = 0;
    for (callers) |caller| if (caller.failure) |err| return err;
}

test "Zig backend reports invalid request errors and its instance remains usable" {
    var server = try startServer();
    defer destroyServer(&server);
    {
        var request = try t.Request.init(testing.allocator, &server, "zig_nonbatch", 1);
        defer destroyRequest(&request);
        try request.addInput(f32, "IN0", &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
        try testing.expectError(error.InvalidArg, server.inferSync(testing.allocator, &request));
    }
    {
        var request = try t.Request.init(testing.allocator, &server, "zig_nonbatch", 1);
        defer destroyRequest(&request);
        try request.addInput(i32, "IN0", &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
        try request.requestOutput("NO_SUCH_OUTPUT");
        try testing.expectError(error.InvalidArg, server.inferSync(testing.allocator, &request));
    }
    var good = try t.Request.init(testing.allocator, &server, "zig_nonbatch", 1);
    defer destroyRequest(&good);
    const values = [_]i32{ 1, 2, 3, 4, 5, 6 };
    try good.addInput(i32, "IN0", &.{ 2, 3 }, &values);
    var response = try server.inferSync(testing.allocator, &good);
    defer response.deinit();
    try expectBytes(&response, .int32, &.{ 2, 3 }, std.mem.sliceAsBytes(&values));
}

test "CPU Zig backend rejects a device tag without reading its address and completes the request" {
    var server = try startServer();
    defer destroyServer(&server);
    var request = try t.Request.init(testing.allocator, &server, "zig_nonbatch", 1);
    defer destroyRequest(&request);
    const values = [_]i32{ 1, -2, 3, -4, 5, -6 };
    try request.addInput(i32, "IN0", &.{ 2, 3 }, &values);
    const handle = try request.ensureIdle();
    try t.check(c.TRITONSERVER_InferenceRequestRemoveAllInputData(handle, "IN0"));
    // Fixed-width validation needs only shape and byte count. The CPU engine
    // must reject this device tag before interpreting the address as host RAM.
    try t.check(c.TRITONSERVER_InferenceRequestAppendInputData(handle, "IN0", @ptrFromInt(0x1000), @sizeOf(@TypeOf(values)), c.TRITONSERVER_MEMORY_GPU, 0));
    try testing.expectError(error.Unsupported, server.inferSync(testing.allocator, &request));
    try testing.expect(!request.in_flight);
    // Reuse the released request after an actual backend execution failure.
    try t.check(c.TRITONSERVER_InferenceRequestRemoveAllInputData(handle, "IN0"));
    try t.check(c.TRITONSERVER_InferenceRequestAppendInputData(handle, "IN0", &values, @sizeOf(@TypeOf(values)), c.TRITONSERVER_MEMORY_CPU, 0));
    var response = try server.inferSync(testing.allocator, &request);
    defer response.deinit();
    try expectBytes(&response, .int32, &.{ 2, 3 }, std.mem.sliceAsBytes(&values));
}

test "unsupported contracts are rejected during configuration or Zig backend initialization" {
    var server = try t.Server.init(testing.allocator, .{
        .model_repository = "tests/fixtures/backend_models/invalid",
        .backend_directory = backend_directory,
        .exit_on_error = false,
        .log_info = false,
    });
    defer destroyServer(&server);
    try testing.expect(try server.isLive());
    try testing.expect(!try server.isReady());
    var models = try server.listModels(testing.allocator);
    defer models.deinit();
    const names = [_][]const u8{
        "zig_invalid_bytes",    "zig_invalid_dtype",     "zig_invalid_dims",  "zig_invalid_names",
        "zig_invalid_reshape",  "zig_invalid_optional",  "zig_invalid_shape", "zig_invalid_ragged",
        "zig_invalid_sequence", "zig_invalid_decoupled",
    };
    for (names) |name| {
        var found = false;
        for (models.models) |model| {
            if (!std.mem.eql(u8, name, model.name)) continue;
            found = true;
            try testing.expect(!std.mem.eql(u8, model.state, "READY"));
            if (std.mem.eql(u8, name, "zig_invalid_shape")) {
                // Core rejects non-TensorRT shape tensors before registering a
                // model version. Its runtime index has no backend error reason.
                try testing.expectEqualStrings("UNKNOWN", model.state);
                try testing.expect(model.version == null);
                try testing.expect(model.reason == null);
            } else {
                try testing.expect(model.reason != null and model.reason.?.len != 0);
            }
        }
        try testing.expect(found);
    }
}
