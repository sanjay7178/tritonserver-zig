const std = @import("std");
const t = @import("tritonserver_zig");
const c = t.c;
const alloc = std.heap.c_allocator;

fn logError(message: []const u8) void {
    std.log.err("Triton: {s}", .{message});
}

fn modelStats(server: *const t.Server, model: [:0]const u8) !std.json.Parsed(std.json.Value) {
    var message: ?*c.TRITONSERVER_Message = null;
    try t.check(c.TRITONSERVER_ServerModelStatistics(server.handle, model, 1, &message));
    const value = message orelse return error.MissingModelStatistics;
    defer t.check(c.TRITONSERVER_MessageDelete(value)) catch unreachable;
    var bytes: [*c]const u8 = null;
    var size: usize = 0;
    try t.check(c.TRITONSERVER_MessageSerializeToJson(value, &bytes, &size));
    if (bytes == null) return error.MissingModelStatistics;
    return std.json.parseFromSlice(std.json.Value, alloc, bytes[0..size], .{ .allocate = .alloc_always });
}

const Timing = struct {
    latency_ns: u64,
    submit_ns: u64,
    max_error: f32,
};

fn infer(io: std.Io, server: *const t.Server, model: []const u8, shape: []const i64, input: []const f32, expected: []const f32, verify: bool) !Timing {
    const started = std.Io.Clock.awake.now(io);
    var request = try t.Request.init(alloc, server, model, 1);
    errdefer request.deinit() catch @panic("request cleanup failed");
    try request.addInput(f32, "IN0", shape, input);
    try request.requestOutput("OUT0");
    const submitting = std.Io.Clock.awake.now(io);
    var pending = try server.inferAsync(alloc, io, &request);
    defer pending.deinit();
    const submitted = std.Io.Clock.awake.now(io);
    var response = try pending.wait();
    defer response.deinit();
    try request.deinit();
    const finished = std.Io.Clock.awake.now(io);
    var max_error: f32 = 0;
    if (verify) {
        const output = try response.output("OUT0");
        if (!std.mem.eql(i64, output.shape, shape)) return error.UnexpectedOutputShape;
        const values = try output.asSlice(f32);
        if (values.len != expected.len) return error.UnexpectedOutputLength;
        for (values, expected) |actual, reference| {
            const abs_error = @abs(actual - reference);
            if (!std.math.isFinite(actual) or abs_error > 0.0005 + 0.002 * @abs(reference)) {
                std.log.err("expected {d}, got {d}", .{ reference, actual });
                return error.IncorrectInference;
            }
            max_error = @max(max_error, abs_error);
        }
    }
    return .{
        .latency_ns = @intCast(durationNs(started, finished)),
        .submit_ns = @intCast(durationNs(submitting, submitted)),
        .max_error = max_error,
    };
}

fn durationNs(start: std.Io.Timestamp, finish: std.Io.Timestamp) i128 {
    return finish.nanoseconds - start.nanoseconds;
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 10) {
        std.log.err("usage: tensorrt-bench REPO BACKENDS MODEL FIXTURE BATCH WIDTH WARMUP ITERATIONS INFO", .{});
        return error.InvalidArguments;
    }
    const batch = try std.fmt.parseInt(usize, args[5], 10);
    const width = try std.fmt.parseInt(usize, args[6], 10);
    const warmup = try std.fmt.parseInt(usize, args[7], 10);
    const iterations = try std.fmt.parseInt(usize, args[8], 10);
    if (batch == 0 or batch > 1024 or width == 0 or width > 8192 or iterations < 2 or warmup < 2)
        return error.InvalidArguments;
    const show_info = std.mem.eql(u8, args[9], "1");
    const elements = batch * width;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[4], alloc, .limited(4 * elements * @sizeOf(f32) + 1));
    defer alloc.free(bytes);
    const values = try alloc.alloc(f32, 4 * elements);
    defer alloc.free(values);
    if (bytes.len != values.len * @sizeOf(f32)) return error.IncorrectFixtureSize;
    @memcpy(std.mem.sliceAsBytes(values), bytes);

    var server = try t.Server.init(alloc, .{
        .model_repository = args[1],
        .backend_directory = args[2],
        .log_info = show_info,
        .on_error = logError,
    });
    defer server.deinit() catch @panic("server cleanup failed");
    if (!try server.isReady()) return error.ServerNotReady;
    const api = try t.apiVersion();
    const shape = [_]i64{ @intCast(batch), @intCast(width) };
    var max_error: f32 = 0;
    for (0..warmup) |i| {
        const variant = i % 2;
        const timing = try infer(io, &server, args[3], &shape, values[(variant * elements)..][0..elements], values[((2 + variant) * elements)..][0..elements], i < 2);
        max_error = @max(max_error, timing.max_error);
    }
    const before = try modelStats(&server, args[3]);
    defer before.deinit();
    const latencies = try alloc.alloc(u64, iterations);
    defer alloc.free(latencies);
    const submits = try alloc.alloc(u64, iterations);
    defer alloc.free(submits);
    const start = std.Io.Clock.awake.now(io);
    const start_real = std.Io.Clock.real.now(io);
    for (latencies, submits, 0..) |*latency, *submit, i| {
        const variant = i % 2;
        const timing = try infer(io, &server, args[3], &shape, values[(variant * elements)..][0..elements], values[((2 + variant) * elements)..][0..elements], i >= iterations - 2);
        latency.* = timing.latency_ns;
        submit.* = timing.submit_ns;
        max_error = @max(max_error, timing.max_error);
    }
    const finish = std.Io.Clock.awake.now(io);
    const finish_real = std.Io.Clock.real.now(io);
    const after = try modelStats(&server, args[3]);
    defer after.deinit();
    var buffer: [8192]u8 = undefined;
    var writer = std.Io.File.stdout().writer(io, &buffer);
    try writer.interface.writeAll("TENSORRT_BENCH_RESULT ");
    try std.json.Stringify.value(.{
        .triton_c_api = api,
        .elapsed_ns = finish.nanoseconds - start.nanoseconds,
        .start_unix_ns = start_real.nanoseconds,
        .end_unix_ns = finish_real.nanoseconds,
        .latency_ns = latencies,
        .submit_ns = submits,
        .max_abs_error = max_error,
        .statistics_before = before.value,
        .statistics_after = after.value,
    }, .{}, &writer.interface);
    try writer.interface.writeByte('\n');
    try writer.interface.flush();
}
