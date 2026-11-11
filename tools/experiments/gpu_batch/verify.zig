//! Optional numerical check for the dense-mean cuBLAS and ONNX GPU experiments.
//! Device execution is established separately by backend logs/ORT profiling.
const std = @import("std");
const t = @import("tritonserver_zig");
const alloc = std.heap.c_allocator;

const Options = struct {
    repository: []const u8 = ".zig-cache/gpu-experiment/cublas-models",
    backend: []const u8 = ".zig-cache/gpu-experiment/backends",
    model: []const u8 = "gpu_dense_mean",
    width: usize = 64,
    help: bool = false,
};

fn parse(args: []const []const u8) !Options {
    var options: Options = .{};
    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        const name = args[index];
        if (std.mem.eql(u8, name, "--help")) {
            options.help = true;
            continue;
        }
        index += 1;
        if (index == args.len) return error.MissingArgumentValue;
        const value = args[index];
        if (std.mem.eql(u8, name, "--model-repository")) {
            options.repository = value;
        } else if (std.mem.eql(u8, name, "--backend-directory")) {
            options.backend = value;
        } else if (std.mem.eql(u8, name, "--model")) {
            options.model = value;
        } else if (std.mem.eql(u8, name, "--width")) {
            options.width = try std.fmt.parseInt(usize, value, 10);
        } else return error.UnknownArgument;
    }
    if (options.width == 0 or options.width > 65_536 or options.repository.len == 0 or
        options.backend.len == 0 or options.model.len == 0)
        return error.InvalidVerificationOption;
    return options;
}

fn valueAt(sample: usize, column: usize) f32 {
    // Positive values also preserve the mean through the ONNX model's ReLUs.
    return @floatFromInt(1 + sample * 64 + column % 64);
}

fn expectedMean(sample: usize, width: usize) f32 {
    var sum: f64 = 0;
    for (0..width) |column| sum += valueAt(sample, column);
    return @floatCast(sum / @as(f64, @floatFromInt(width)));
}

fn checkOutput(result: *const t.InferenceResponse, first_sample: usize, count: usize, width: usize) !void {
    const output = try result.output("OUT0");
    const expected_shape = [_]i64{ @intCast(count), @intCast(width) };
    if (!std.mem.eql(i64, &expected_shape, output.shape)) return error.InvalidOutputShape;
    const values = try output.asSlice(f32);
    if (values.len != count * width) return error.InvalidOutputBuffer;
    for (0..count) |row| {
        const expected = expectedMean(first_sample + row, width);
        for (values[row * width ..][0..width], 0..) |actual, column| {
            if (!std.math.isFinite(actual) or !std.math.approxEqAbs(f32, expected, actual, 0.001)) {
                std.debug.print("incorrect sample {d}, column {d}: expected={d} actual={d}\n", .{
                    first_sample + row, column, expected, actual,
                });
                return error.IncorrectOutput;
            }
        }
    }
}

fn runGroup(io: std.Io, server: *const t.Server, options: Options, label: []const u8, counts: []const usize) !void {
    var total: usize = 0;
    for (counts) |count| total += count;
    var scheduler = try t.Scheduler.init(alloc, io, server, options.model, 1, .{
        .max_batch_size = @intCast(total),
        .max_wait_us = if (total == 1) 0 else 1_000_000,
        .worker_count = 1,
    });
    defer scheduler.deinit();
    const futures = try alloc.alloc(t.ScheduledInference, counts.len);
    defer alloc.free(futures);
    var started: usize = 0;
    defer for (futures[0..started]) |*future| future.deinit();
    var first_sample: usize = 0;
    for (counts, futures) |count, *future| {
        const values = try alloc.alloc(f32, count * options.width);
        defer alloc.free(values);
        for (0..count) |row| {
            for (values[row * options.width ..][0..options.width], 0..) |*value, column|
                value.* = valueAt(first_sample + row, column);
        }
        const shape = [_]i64{ @intCast(count), @intCast(options.width) };
        const input = try t.InputTensor.fromSlice(f32, "IN0", &shape, values);
        future.* = try scheduler.submit(alloc, &.{input}, .{ .outputs = &.{"OUT0"} });
        started += 1;
        first_sample += count;
    }
    first_sample = 0;
    for (counts, futures) |count, *future| {
        var result = try future.wait();
        defer result.deinit();
        try checkOutput(&result, first_sample, count, options.width);
        first_sample += count;
    }
    scheduler.shutdown();
    const stats = scheduler.stats();
    if (stats.dispatched_batches != 1 or stats.max_batch_samples != total or
        stats.accepted_requests != counts.len or stats.failed_requests != 0)
        return error.UnexpectedBatchStatistics;
    std.debug.print("{s} correct: {d} requests -> one batch of {d}; output row means and shapes verified (width={d})\n", .{
        label, counts.len, total, options.width,
    });
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = try parse(args[1..]);
    if (options.help) {
        std.debug.print(
            "Usage: gpu-verify [--model-repository PATH] [--backend-directory PATH]\n" ++
                "  --model NAME --width 64\n" ++
                "Checks a single B1, eight B1 requests merged into B8, and B1/B3/B4 merged into B8.\n" ++
                "Use --width 8192 for the optional ONNX dense-mean experiment.\n",
            .{},
        );
        return;
    }
    var server = try t.Server.init(alloc, .{
        .model_repository = options.repository,
        .backend_directory = options.backend,
        .log_info = false,
    });
    defer server.deinit() catch @panic("verification server cleanup failed");
    try runGroup(init.io, &server, options, "B1", &.{1});
    try runGroup(init.io, &server, options, "Eight B1", &.{ 1, 1, 1, 1, 1, 1, 1, 1 });
    try runGroup(init.io, &server, options, "Mixed B1/B3/B4", &.{ 1, 3, 4 });
    try server.stop();
    std.debug.print("Numerical and scheduler ownership checks passed for {s}.\n", .{options.model});
}
