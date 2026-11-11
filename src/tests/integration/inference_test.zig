const std = @import("std");
const t = @import("../../root.zig");
const testing = std.testing;

fn startServer() !t.Server {
    return t.Server.init(testing.allocator, .{
        .model_repository = "tests/fixtures/models",
        .backend_directory = ".zig-cache/phase2/backends",
        .log_info = false,
    });
}

fn destroyRequest(request: *t.Request) void {
    request.deinit() catch @panic("request cleanup failed");
}

fn destroyServer(server: *t.Server) void {
    server.deinit() catch @panic("server cleanup failed");
}

test "real inference copies caller inputs and returns owned typed outputs" {
    var server = try startServer();
    defer destroyServer(&server);
    var request = try t.Request.init(testing.allocator, &server, "nonbatching", -1);
    defer destroyRequest(&request);
    var input = [_]i32{ 42, -7, std.math.minInt(i32), std.math.maxInt(i32) };
    const expected = input;
    try request.addInput(i32, "IN0", &.{4}, &input);
    try request.requestOutput("OUT0");
    @memset(&input, 0);
    var response = try server.inferSync(testing.allocator, &request);
    defer response.deinit();
    const output = try response.output("OUT0");
    try testing.expectEqual(t.DataType.int32, output.datatype);
    try testing.expectEqualSlices(i64, &.{4}, output.shape);
    try testing.expectEqualSlices(i32, &expected, try output.asSlice(i32));
    try request.deinit();
    try server.stop();
    try server.deinit();
    // No result string, shape, or tensor storage may borrow the C response,
    // released request, caller input, or deleted server.
    try testing.expectEqualStrings("OUT0", output.name);
    try testing.expectEqualSlices(i32, &expected, try output.asSlice(i32));
    response.deinit();
}

test "real batched inference decodes the full batch with its protocol shape" {
    var server = try startServer();
    defer destroyServer(&server);
    var request = try t.Request.init(testing.allocator, &server, "batching", 1);
    defer destroyRequest(&request);
    var input: [32]i32 = undefined;
    for (&input, 0..) |*value, i| value.* = @as(i32, @intCast(i)) - 16;
    // A full batch avoids waiting for this fixture's five-second batch window.
    try request.addInput(i32, "IN0", &.{ 8, 4 }, &input);
    var response = try server.inferSync(testing.allocator, &request);
    defer response.deinit();
    const output = try response.output("OUT0");
    try testing.expectEqualSlices(i64, &.{ 8, 4 }, output.shape);
    try testing.expectEqualSlices(i32, &input, try output.asSlice(i32));
}

const Worker = struct {
    server: *t.Server,
    input: [4]i32,
    failure: ?anyerror = null,

    fn run(self: *Worker) void {
        self.perform() catch |err| {
            self.failure = err;
        };
    }

    fn perform(self: *Worker) !void {
        // Each worker has its own request/response; libc allocation is safe
        // across callback threads and independent caller threads.
        var request = try t.Request.init(std.heap.c_allocator, self.server, "nonbatching", -1);
        defer destroyRequest(&request);
        try request.addInput(i32, "IN0", &.{4}, &self.input);
        var response = try self.server.inferSync(std.heap.c_allocator, &request);
        defer response.deinit();
        const output = try response.output("OUT0");
        try testing.expectEqualSlices(i32, &self.input, try output.asSlice(i32));
    }
};

test "independent callers can infer concurrently through the same live server" {
    var server = try startServer();
    defer destroyServer(&server);
    var workers = [_]Worker{
        .{ .server = &server, .input = .{ 1, 2, 3, 4 } },
        .{ .server = &server, .input = .{ -1, -2, -3, -4 } },
        .{ .server = &server, .input = .{ 101, 102, 103, 104 } },
        .{ .server = &server, .input = .{ 0, 0, 0, 0 } },
    };
    var threads: [workers.len]std.Thread = undefined;
    var started: usize = 0;
    defer for (threads[0..started]) |thread| thread.join();
    for (&workers) |*worker| {
        threads[started] = try std.Thread.spawn(.{}, Worker.run, .{worker});
        started += 1;
    }
    for (threads[0..started]) |thread| thread.join();
    started = 0;
    for (workers) |worker| if (worker.failure) |err| return err;
}

test "released requests can be reused and each response remains independent" {
    var server = try startServer();
    defer destroyServer(&server);
    var request = try t.Request.init(testing.allocator, &server, "nonbatching", -1);
    defer destroyRequest(&request);
    const input = [_]i32{ 7, -2, 0, 91 };
    try request.addInput(i32, "IN0", &.{4}, &input);
    try request.requestOutput("OUT0");
    var first = try server.inferSync(testing.allocator, &request);
    defer first.deinit();
    var second = try server.inferSync(testing.allocator, &request);
    defer second.deinit();
    try testing.expectEqualSlices(i32, &input, try (try first.output("OUT0")).asSlice(i32));
    try testing.expectEqualSlices(i32, &input, try (try second.output("OUT0")).asSlice(i32));
    second.deinit();
    try testing.expectEqualSlices(i32, &input, try (try first.output("OUT0")).asSlice(i32));
    try testing.expectError(error.OutputNotFound, first.output("unknown_output"));
    try testing.expectError(error.DataTypeMismatch, (try first.output("OUT0")).asSlice(f32));
}

test "request validation rejects invalid dimensions, element counts and duplicate inputs" {
    var server = try startServer();
    defer destroyServer(&server);
    var request = try t.Request.init(testing.allocator, &server, "nonbatching", -1);
    defer destroyRequest(&request);
    const input = [_]i32{ 1, 2, 3, 4 };
    try testing.expectError(error.InvalidShape, request.addInput(i32, "IN0", &.{-1}, &input));
    try testing.expectError(error.ElementCountMismatch, request.addInput(i32, "IN0", &.{3}, &input));
    try testing.expectError(error.UnsupportedDataType, request.addInput(u128, "IN0", &.{1}, &.{1}));
    try request.addInput(i32, "IN0", &.{4}, &input);
    try testing.expectError(error.DuplicateInput, request.addInput(i32, "IN0", &.{4}, &input));
    var response = try server.inferSync(testing.allocator, &request);
    defer response.deinit();
    try testing.expectEqualSlices(i32, &input, try (try response.output("OUT0")).asSlice(i32));
    try request.deinit();
    try request.deinit();
    try testing.expectError(error.RequestClosed, request.addInput(i32, "IN0", &.{4}, &input));
    try testing.expectError(error.RequestClosed, request.requestOutput("OUT0"));
    try testing.expectError(error.RequestClosed, server.inferSync(testing.allocator, &request));
}

test "unknown models and invalid requested outputs produce inference errors" {
    var server = try startServer();
    defer destroyServer(&server);
    const missing = t.Request.init(testing.allocator, &server, "phase3_missing_model", -1);
    if (missing) |created| {
        var unexpected = created;
        try unexpected.deinit();
        return error.ExpectedMissingModelFailure;
    } else |err| {
        try testing.expect(err == error.NotFound or err == error.Unavailable or err == error.InvalidArg);
    }
    var request = try t.Request.init(testing.allocator, &server, "nonbatching", -1);
    defer destroyRequest(&request);
    try request.addInput(i32, "IN0", &.{4}, &.{ 1, 2, 3, 4 });
    try request.requestOutput("phase3_missing_output");
    try testing.expectError(error.InvalidArg, server.inferSync(testing.allocator, &request));
    // Immediate submission failure must leave ownership with the request.
    try request.deinit();
}

test "Triton rejects missing inputs and mismatched model input datatypes" {
    var server = try startServer();
    defer destroyServer(&server);
    var empty = try t.Request.init(testing.allocator, &server, "nonbatching", -1);
    defer destroyRequest(&empty);
    try testing.expectError(error.InvalidArg, server.inferSync(testing.allocator, &empty));
    var wrong_type = try t.Request.init(testing.allocator, &server, "nonbatching", -1);
    defer destroyRequest(&wrong_type);
    try wrong_type.addInput(f32, "IN0", &.{4}, &.{ 1, 2, 3, 4 });
    try testing.expectError(error.InvalidArg, server.inferSync(testing.allocator, &wrong_type));
}

fn inferAllocationCase(alloc: std.mem.Allocator, server: *t.Server) !void {
    var request = try t.Request.init(alloc, server, "nonbatching", -1);
    defer destroyRequest(&request);
    try request.addInput(i32, "IN0", &.{4}, &.{ 4, 3, 2, 1 });
    try request.requestOutput("OUT0");
    var response = try server.inferSync(alloc, &request);
    defer response.deinit();
    try testing.expectEqualSlices(i32, &.{ 4, 3, 2, 1 }, try (try response.output("OUT0")).asSlice(i32));
}

test "request and response allocation failures release partial ownership" {
    var server = try startServer();
    defer destroyServer(&server);
    try testing.checkAllAllocationFailures(testing.allocator, inferAllocationCase, .{&server});
}

test "asynchronous backend errors map and release all callback ownership" {
    var repo = testing.tmpDir(.{});
    defer repo.cleanup();
    try repo.dir.createDirPath(testing.io, "backend_error/1");
    try repo.dir.writeFile(testing.io, .{ .sub_path = "backend_error/config.pbtxt", .data =
        \\backend: "phase3_error"
        \\max_batch_size: 0
        \\input [{ name: "IN0" data_type: TYPE_INT32 dims: [4] }]
        \\output [{ name: "OUT0" data_type: TYPE_INT32 dims: [4] }]
        \\instance_group [{ kind: KIND_CPU }]
    });
    const path = try repo.dir.realPathFileAlloc(testing.io, ".", testing.allocator);
    defer testing.allocator.free(path);
    var server = try t.Server.init(testing.allocator, .{
        .model_repository = path,
        .backend_directory = ".zig-cache/phase2/backends",
        .log_info = false,
    });
    defer destroyServer(&server);
    var request = try t.Request.init(testing.allocator, &server, "backend_error", -1);
    defer destroyRequest(&request);
    // The request passes model validation. The test-only backend then returns
    // an execute error, which Triton delivers through the response callback.
    try request.addInput(i32, "IN0", &.{4}, &.{ 1, 2, 3, 4 });
    try testing.expectError(error.InvalidArg, server.inferSync(testing.allocator, &request));
    // Both callback paths must have completed, allowing safe immediate reuse
    // and eventual cleanup even when a response carries a backend error.
    try testing.expectError(error.InvalidArg, server.inferSync(testing.allocator, &request));
}

test "pending inference guards request ownership until wait and supports safe draining" {
    var server = try startServer();
    defer destroyServer(&server);
    var request = try t.Request.init(testing.allocator, &server, "nonbatching", -1);
    defer destroyRequest(&request);
    try request.addInput(i32, "IN0", &.{4}, &.{ 10, 20, 30, 40 });
    var pending = try server.inferAsync(testing.allocator, &request);
    defer pending.deinit();
    try testing.expectError(error.RequestInFlight, request.deinit());
    try testing.expectError(error.RequestInFlight, request.requestOutput("OUT0"));
    try testing.expectError(error.RequestInFlight, server.inferSync(testing.allocator, &request));
    var response = try pending.wait();
    defer response.deinit();
    try testing.expectEqualSlices(i32, &.{ 10, 20, 30, 40 }, try (try response.output("OUT0")).asSlice(i32));
    // Dropping a pending owner must drain both response and release callbacks.
    var discarded = try server.inferAsync(testing.allocator, &request);
    discarded.deinit();
    discarded.deinit();
    var after_discard = try server.inferSync(testing.allocator, &request);
    defer after_discard.deinit();
    try testing.expectEqualSlices(i32, &.{ 10, 20, 30, 40 }, try (try after_discard.output("OUT0")).asSlice(i32));
}

test "request initialization rejects decoupled models before submission" {
    var repo = testing.tmpDir(.{});
    defer repo.cleanup();
    try repo.dir.createDirPath(testing.io, "decoupled/1");
    try repo.dir.writeFile(testing.io, .{ .sub_path = "decoupled/config.pbtxt", .data =
        \\backend: "phase3_error"
        \\max_batch_size: 0
        \\model_transaction_policy { decoupled: true }
        \\input [{ name: "IN0" data_type: TYPE_INT32 dims: [4] }]
        \\output [{ name: "OUT0" data_type: TYPE_INT32 dims: [4] }]
        \\instance_group [{ kind: KIND_CPU }]
    });
    const path = try repo.dir.realPathFileAlloc(testing.io, ".", testing.allocator);
    defer testing.allocator.free(path);
    var server = try t.Server.init(testing.allocator, .{
        .model_repository = path,
        .backend_directory = ".zig-cache/phase2/backends",
        .log_info = false,
    });
    defer destroyServer(&server);
    try testing.expect(try server.isReady());
    try testing.expectError(error.DecoupledModelUnsupported, t.Request.init(testing.allocator, &server, "decoupled", -1));
}

test "requests cannot be submitted through a different live server" {
    var first = try startServer();
    defer destroyServer(&first);
    var second = try startServer();
    defer destroyServer(&second);
    var request = try t.Request.init(testing.allocator, &first, "nonbatching", -1);
    defer destroyRequest(&request);
    try request.addInput(i32, "IN0", &.{4}, &.{ 9, 8, 7, 6 });
    try testing.expectError(error.RequestServerMismatch, second.inferSync(testing.allocator, &request));
    // Rejected submission leaves the original request idle and usable by its
    // owning server, rather than taking ownership or installing live callbacks.
    var response = try first.inferSync(testing.allocator, &request);
    defer response.deinit();
    try testing.expectEqualSlices(i32, &.{ 9, 8, 7, 6 }, try (try response.output("OUT0")).asSlice(i32));
}
