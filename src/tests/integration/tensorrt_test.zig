const std = @import("std");
const t = @import("tritonserver_zig");
const options = @import("tensorrt_test_options");
const testing = std.testing;
const c = t.c;

const ModelStats = struct {
    const Count = struct { count: usize };

    execution_count: usize,
    inference_count: usize,
    inference_stats: struct { success: Count, fail: Count },
};

const DlInfo = extern struct {
    filename: ?[*:0]const u8,
    base: ?*anyopaque,
    symbol_name: ?[*:0]const u8,
    symbol: ?*anyopaque,
};
extern "c" fn dladdr(address: *const anyopaque, info: *DlInfo) c_int;

var graph_launches: std.atomic.Value(usize) = .init(0);

// Linux CUDA passes opaque graph and stream pointers and returns a C enum.
// Forward through RTLD_NEXT before counting a successful backend replay.
export fn cudaGraphLaunch(graph: ?*anyopaque, stream: ?*anyopaque) callconv(.c) c_int {
    const Launch = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) c_int;
    const next: *anyopaque = @ptrFromInt(std.math.maxInt(usize));
    const launch: Launch = @ptrCast(std.c.dlsym(next, "cudaGraphLaunch") orelse
        @panic("could not resolve the real CUDA graph launcher"));
    var info: DlInfo = undefined;
    const from_backend = dladdr(@ptrFromInt(@returnAddress()), &info) != 0 and
        if (info.filename) |filename|
            std.mem.eql(u8, std.fs.path.basename(std.mem.span(filename)), "libtriton_tensorrt.so")
        else
            false;
    const status = launch(graph, stream);
    if (status == 0 and from_backend) _ = graph_launches.fetchAdd(1, .monotonic);
    return status;
}

fn logError(message: []const u8) void {
    std.debug.print("Triton: {s}\n", .{message});
}

fn modelStats(server: *const t.Server) !ModelStats {
    var raw: ?*c.TRITONSERVER_Message = null;
    try t.check(c.TRITONSERVER_ServerModelStatistics(server.handle, "bench", 1, &raw));
    const message = raw orelse return error.MissingModelStatistics;
    defer t.check(c.TRITONSERVER_MessageDelete(message)) catch @panic("delete model statistics");
    var bytes: [*c]const u8 = null;
    var size: usize = 0;
    try t.check(c.TRITONSERVER_MessageSerializeToJson(message, &bytes, &size));
    if (bytes == null) return error.MissingModelStatistics;
    const parsed = try std.json.parseFromSlice(
        struct { model_stats: [1]ModelStats },
        testing.allocator,
        bytes[0..size],
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();
    return parsed.value.model_stats[0];
}

fn verifyCase(comptime batch: usize, comptime graphs: bool) !void {
    const alloc = testing.allocator;
    const io = testing.io;
    const width = 512;
    const elements = batch * width;
    const fixture_size = 4 * elements * @sizeOf(f32);
    const case_directory = std.fmt.comptimePrint("b{d}-graphs-{d}/models", .{ batch, @intFromBool(graphs) });
    const repository = try std.fs.path.join(alloc, &.{ options.artifacts, case_directory });
    defer alloc.free(repository);
    const fixture = try std.fs.path.join(alloc, &.{ options.artifacts, std.fmt.comptimePrint("batch-{d}.bin", .{batch}) });
    defer alloc.free(fixture);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, fixture, alloc, .limited(fixture_size + 1));
    defer alloc.free(bytes);
    try testing.expectEqual(fixture_size, bytes.len);
    var tensors: [4][elements]f32 = undefined;
    for (0..4 * elements) |index| {
        const offset = index * @sizeOf(f32);
        tensors[index / elements][index % elements] = @bitCast(std.mem.readInt(u32, bytes[offset..][0..4], .little));
    }

    var server = try t.Server.init(alloc, .{
        .model_repository = repository,
        .backend_directory = options.backends,
        .log_info = false,
        .on_error = logError,
    });
    defer server.deinit() catch @panic("delete Triton server");
    try testing.expect(try server.isReady());

    const shape = [_]i64{ batch, width };
    const case_requests = [_]usize{ 0, 1, 1, 0 };
    const launches_before = graph_launches.load(.monotonic);
    for (case_requests) |variant| {
        var request = try t.Request.init(alloc, &server, "bench", 1);
        defer request.deinit() catch @panic("delete inference request");
        try request.addInput(f32, "IN0", &shape, &tensors[variant]);
        try request.requestOutput("OUT0");
        var response = try server.inferSync(alloc, io, &request);
        defer response.deinit();
        const output = try response.output("OUT0");
        try testing.expectEqualSlices(i64, &shape, output.shape);
        const prediction = try output.asSlice(f32);
        const expected = &tensors[variant + 2];
        try testing.expectEqual(expected.len, prediction.len);
        for (prediction, expected) |actual, reference| {
            try testing.expect(std.math.isFinite(actual) and std.math.isFinite(reference));
            try testing.expectApproxEqAbs(reference, actual, 0.0005 + 0.002 * @abs(reference));
        }
    }

    const stats = try modelStats(&server);
    try testing.expectEqual(@as(usize, case_requests.len), stats.execution_count);
    try testing.expectEqual(@as(usize, case_requests.len * batch), stats.inference_count);
    try testing.expectEqual(@as(usize, case_requests.len), stats.inference_stats.success.count);
    try testing.expectEqual(@as(usize, 0), stats.inference_stats.fail.count);
    const expected_launches: usize = if (graphs) case_requests.len else 0;
    try testing.expectEqual(expected_launches, graph_launches.load(.monotonic) - launches_before);
}

test "TensorRT batch one without CUDA Graphs" {
    try verifyCase(1, false);
}

test "TensorRT batch one replays CUDA Graphs" {
    try verifyCase(1, true);
}

test "TensorRT batch eight without CUDA Graphs" {
    try verifyCase(8, false);
}

test "TensorRT batch eight replays CUDA Graphs" {
    try verifyCase(8, true);
}
