const std = @import("std");
const t = @import("../../root.zig");
const testing = std.testing;

fn startServer() !t.Server {
    return t.Server.init(testing.allocator, .{
        .model_repository = "tests/fixtures/scheduler_models",
        .backend_directory = ".zig-cache/phase2/backends",
        .log_info = false,
    });
}

fn destroyServer(server: *t.Server) void {
    server.deinit() catch @panic("server cleanup failed");
}

fn submit(scheduler: *t.Scheduler, shape: []const i64, values: []const i32) !t.ScheduledInference {
    const input = try t.InputTensor.fromSlice(i32, "IN0", shape, values);
    return scheduler.submit(testing.allocator, &.{input}, .{});
}

fn expectOutput(response: *const t.InferenceResponse, shape: []const i64, values: []const i32) !void {
    const output = try response.output("OUT0");
    try testing.expectEqualSlices(i64, shape, output.shape);
    try testing.expectEqualSlices(i32, values, try output.asSlice(i32));
}

fn expectBatchInfo(response: *const t.InferenceResponse, samples: usize, total: i64, execution: ?i64) !i64 {
    const output = try response.output("BATCH_INFO");
    try testing.expectEqualSlices(i64, &.{ @intCast(samples), 3 }, output.shape);
    const values = try output.asSlice(i64);
    try testing.expectEqual(samples * 3, values.len);
    try testing.expect(values[0] > 0);
    for (0..samples) |i| {
        try testing.expectEqual(execution orelse values[0], values[i * 3]);
        try testing.expectEqual(total, values[i * 3 + 1]);
        // The native scheduler is disabled. One Execute request containing N
        // samples proves concatenation happened in Zig, before the C call.
        try testing.expectEqual(@as(i64, 1), values[i * 3 + 2]);
    }
    return values[0];
}

test "scheduler merges eight copied inputs and futures outlive scheduler and server" {
    var server = try startServer();
    defer destroyServer(&server);
    var scheduler = try t.Scheduler.init(testing.allocator, &server, "phase4_fixed", -1, .{
        .worker_count = 1,
        .max_wait_us = 1_000_000,
        .queue_capacity = 32,
    });
    defer scheduler.deinit();
    var futures: [8]?t.ScheduledInference = @splat(null);
    defer for (&futures) |*future| if (future.*) |*value| value.deinit();
    var expected: [8][4]i32 = undefined;
    for (&expected, 0..) |*row, i| {
        row.* = .{ @intCast(i + 1), -@as(i32, @intCast(i + 1)), 17, 29 };
        var caller_input = row.*;
        futures[i] = try submit(&scheduler, &.{ 1, 4 }, &caller_input);
        @memset(&caller_input, 0);
    }
    scheduler.shutdown();
    scheduler.shutdown();
    const stats = scheduler.stats();
    try testing.expectEqual(@as(u64, 8), stats.accepted_requests);
    try testing.expectEqual(@as(u64, 8), stats.accepted_samples);
    try testing.expectEqual(@as(u64, 1), stats.dispatched_batches);
    try testing.expectEqual(@as(u64, 8), stats.dispatched_samples);
    try testing.expectEqual(@as(u64, 8), stats.max_batch_samples);
    try testing.expectEqual(@as(u64, 0), stats.failed_requests);
    scheduler.deinit();
    try server.stop();
    try server.deinit();
    var execution: ?i64 = null;
    for (&futures, expected) |*future, row| {
        var response = try future.*.?.wait();
        defer response.deinit();
        try expectOutput(&response, &.{ 1, 4 }, &row);
        execution = try expectBatchInfo(&response, 1, 8, execution);
    }
}

test "scheduler combines mixed logical batch sizes and splits each complete result" {
    var server = try startServer();
    defer destroyServer(&server);
    var scheduler = try t.Scheduler.init(testing.allocator, &server, "phase4_fixed", 1, .{
        .worker_count = 1,
        .max_wait_us = 1_000_000,
    });
    defer scheduler.deinit();
    const one = [_]i32{ 1, 2, 3, 4 };
    const three = [_]i32{ 11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34 };
    const four = [_]i32{ 41, 42, 43, 44, 51, 52, 53, 54, 61, 62, 63, 64, 71, 72, 73, 74 };
    var first = try submit(&scheduler, &.{ 1, 4 }, &one);
    defer first.deinit();
    var second = try submit(&scheduler, &.{ 3, 4 }, &three);
    defer second.deinit();
    var third = try submit(&scheduler, &.{ 4, 4 }, &four);
    defer third.deinit();
    var a = try first.wait();
    defer a.deinit();
    var b = try second.wait();
    defer b.deinit();
    var c = try third.wait();
    defer c.deinit();
    try expectOutput(&a, &.{ 1, 4 }, &one);
    try expectOutput(&b, &.{ 3, 4 }, &three);
    try expectOutput(&c, &.{ 4, 4 }, &four);
    const execution = try expectBatchInfo(&a, 1, 8, null);
    _ = try expectBatchInfo(&b, 3, 8, execution);
    _ = try expectBatchInfo(&c, 4, 8, execution);
    scheduler.shutdown();
    const stats = scheduler.stats();
    try testing.expectEqual(@as(u64, 3), stats.accepted_requests);
    try testing.expectEqual(@as(u64, 8), stats.accepted_samples);
    try testing.expectEqual(@as(u64, 1), stats.dispatched_batches);
}

test "scheduler canonicalizes multiple inputs and requested output order before merging" {
    var server = try startServer();
    defer destroyServer(&server);
    var scheduler = try t.Scheduler.init(testing.allocator, &server, "phase4_multi", -1, .{
        .worker_count = 1,
        .max_wait_us = 1_000_000,
    });
    defer scheduler.deinit();
    const first_input = try t.InputTensor.fromSlice(i32, "IN0", &.{ 1, 4 }, &.{ 1, 2, 3, 4 });
    const mismatched = try t.InputTensor.fromSlice(i32, "IN1", &.{ 2, 4 }, &.{ 5, 6, 7, 8, 9, 10, 11, 12 });
    try testing.expectError(error.InvalidShape, scheduler.submit(testing.allocator, &.{ first_input, mismatched }, .{}));
    var futures: [8]?t.ScheduledInference = @splat(null);
    defer for (&futures) |*future| if (future.*) |*value| value.deinit();
    var expected_zero: [8][4]i32 = undefined;
    var expected_one: [8][4]i32 = undefined;
    for (&expected_zero, &expected_one, 0..) |*zero, *one, i| {
        const base: i32 = @intCast(i * 10);
        zero.* = .{ base + 1, base + 2, base + 3, base + 4 };
        one.* = .{ -1001 - base, -1002 - base, -1003 - base, -1004 - base };
        var caller_zero = zero.*;
        var caller_one = one.*;
        const input_zero = try t.InputTensor.fromSlice(i32, "IN0", &.{ 1, 4 }, &caller_zero);
        const input_one = try t.InputTensor.fromSlice(i32, "IN1", &.{ 1, 4 }, &caller_one);
        const inputs: [2]t.InputTensor = if (i % 2 == 0)
            .{ input_zero, input_one }
        else
            .{ input_one, input_zero };
        const outputs: []const []const u8 = if (i % 2 == 0)
            &.{ "OUT0", "OUT1", "BATCH_INFO" }
        else
            &.{ "BATCH_INFO", "OUT1", "OUT0" };
        futures[i] = try scheduler.submit(testing.allocator, &inputs, .{ .outputs = outputs });
        @memset(&caller_zero, 0);
        @memset(&caller_one, 0);
    }
    scheduler.shutdown();
    var execution: ?i64 = null;
    for (&futures, expected_zero, expected_one) |*future, zero, one| {
        var response = try future.*.?.wait();
        defer response.deinit();
        try testing.expectEqual(@as(usize, 3), response.outputs.len);
        try expectOutput(&response, &.{ 1, 4 }, &zero);
        const output_one = try response.output("OUT1");
        try testing.expectEqualSlices(i64, &.{ 1, 4 }, output_one.shape);
        try testing.expectEqualSlices(i32, &one, try output_one.asSlice(i32));
        execution = try expectBatchInfo(&response, 1, 8, execution);
    }
    const stats = scheduler.stats();
    try testing.expectEqual(@as(u64, 8), stats.accepted_requests);
    try testing.expectEqual(@as(u64, 8), stats.accepted_samples);
    try testing.expectEqual(@as(u64, 1), stats.dispatched_batches);
    try testing.expectEqual(@as(u64, 8), stats.dispatched_samples);
    try testing.expectEqual(@as(u64, 1), stats.rejected_requests);
}

test "different requested outputs partition scheduled batches" {
    var server = try startServer();
    defer destroyServer(&server);
    var scheduler = try t.Scheduler.init(testing.allocator, &server, "phase4_fixed", -1, .{
        .worker_count = 1,
        .max_wait_us = 1_000,
    });
    defer scheduler.deinit();
    const input = try t.InputTensor.fromSlice(i32, "IN0", &.{ 1, 4 }, &.{ 1, 2, 3, 4 });
    var first = try scheduler.submit(testing.allocator, &.{input}, .{ .outputs = &.{"OUT0"} });
    defer first.deinit();
    var second = try scheduler.submit(testing.allocator, &.{input}, .{ .outputs = &.{"BATCH_INFO"} });
    defer second.deinit();
    var a = try first.wait();
    defer a.deinit();
    var b = try second.wait();
    defer b.deinit();
    try testing.expectEqual(@as(usize, 1), a.outputs.len);
    try testing.expectEqual(@as(usize, 1), b.outputs.len);
    try expectOutput(&a, &.{ 1, 4 }, &.{ 1, 2, 3, 4 });
    _ = try expectBatchInfo(&b, 1, 1, null);
    try testing.expectError(error.OutputNotFound, a.output("BATCH_INFO"));
    try testing.expectError(error.OutputNotFound, b.output("OUT0"));
    scheduler.shutdown();
    try testing.expectEqual(@as(u64, 2), scheduler.stats().dispatched_batches);
}

test "different dynamic inner dimensions are not concatenated" {
    var server = try startServer();
    defer destroyServer(&server);
    var scheduler = try t.Scheduler.init(testing.allocator, &server, "phase4_dynamic", -1, .{
        .worker_count = 1,
        .max_wait_us = 1_000,
    });
    defer scheduler.deinit();
    var first = try submit(&scheduler, &.{ 1, 2 }, &.{ 1, 2 });
    defer first.deinit();
    var second = try submit(&scheduler, &.{ 1, 3 }, &.{ 3, 4, 5 });
    defer second.deinit();
    var a = try first.wait();
    defer a.deinit();
    var b = try second.wait();
    defer b.deinit();
    try expectOutput(&a, &.{ 1, 2 }, &.{ 1, 2 });
    try expectOutput(&b, &.{ 1, 3 }, &.{ 3, 4, 5 });
    const execution = try expectBatchInfo(&a, 1, 1, null);
    try testing.expect(execution != try expectBatchInfo(&b, 1, 1, null));
    scheduler.shutdown();
    try testing.expectEqual(@as(u64, 2), scheduler.stats().dispatched_batches);
}

test "nonbatching models pass through one logical request at a time" {
    var server = try startServer();
    defer destroyServer(&server);
    var scheduler = try t.Scheduler.init(testing.allocator, &server, "phase4_nonbatch", -1, .{
        .worker_count = 2,
        .max_wait_us = 1_000,
    });
    defer scheduler.deinit();
    var first = try submit(&scheduler, &.{4}, &.{ 1, 2, 3, 4 });
    defer first.deinit();
    var second = try submit(&scheduler, &.{4}, &.{ 5, 6, 7, 8 });
    defer second.deinit();
    var a = try first.wait();
    defer a.deinit();
    var b = try second.wait();
    defer b.deinit();
    try expectOutput(&a, &.{4}, &.{ 1, 2, 3, 4 });
    try expectOutput(&b, &.{4}, &.{ 5, 6, 7, 8 });
    for ([_]*const t.InferenceResponse{ &a, &b }) |response| {
        const info = try response.output("BATCH_INFO");
        try testing.expectEqualSlices(i64, &.{3}, info.shape);
        const values = try info.asSlice(i64);
        try testing.expectEqualSlices(i64, &.{ 1, 1 }, values[1..]);
    }
    scheduler.shutdown();
    try testing.expectEqual(@as(u64, 2), scheduler.stats().dispatched_batches);
    try testing.expectEqual(@as(u64, 1), scheduler.stats().max_batch_samples);
}

test "scheduler rejects invalid options and tensors before dispatch" {
    var server = try startServer();
    defer destroyServer(&server);
    try testing.expectError(error.InvalidSchedulerOptions, t.Scheduler.init(testing.allocator, &server, "phase4_fixed", -1, .{ .max_batch_size = 0 }));
    try testing.expectError(error.InvalidSchedulerOptions, t.Scheduler.init(testing.allocator, &server, "phase4_fixed", -1, .{ .max_batch_size = 9 }));
    try testing.expectError(error.InvalidSchedulerOptions, t.Scheduler.init(testing.allocator, &server, "phase4_fixed", -1, .{ .worker_count = 0 }));
    try testing.expectError(error.InvalidSchedulerOptions, t.Scheduler.init(testing.allocator, &server, "phase4_fixed", -1, .{ .queue_capacity = 0 }));
    var scheduler = try t.Scheduler.init(testing.allocator, &server, "phase4_fixed", -1, .{});
    defer scheduler.deinit();
    const input = try t.InputTensor.fromSlice(i32, "IN0", &.{ 1, 4 }, &.{ 1, 2, 3, 4 });
    const wrong_type = try t.InputTensor.fromSlice(f32, "IN0", &.{ 1, 4 }, &.{ 1, 2, 3, 4 });
    var unknown = input;
    unknown.name = "UNKNOWN";
    try testing.expectError(error.MissingInput, scheduler.submit(testing.allocator, &.{}, .{}));
    try testing.expectError(error.UnknownInput, scheduler.submit(testing.allocator, &.{unknown}, .{}));
    try testing.expectError(error.DuplicateInput, scheduler.submit(testing.allocator, &.{ input, input }, .{}));
    try testing.expectError(error.DataTypeMismatch, scheduler.submit(testing.allocator, &.{wrong_type}, .{}));
    try testing.expectError(error.UnknownOutput, scheduler.submit(testing.allocator, &.{input}, .{ .outputs = &.{"UNKNOWN"} }));
    try testing.expectError(error.DuplicateOutput, scheduler.submit(testing.allocator, &.{input}, .{ .outputs = &.{ "OUT0", "OUT0" } }));
    const wrong_shape = try t.InputTensor.fromSlice(i32, "IN0", &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    try testing.expectError(error.InvalidShape, scheduler.submit(testing.allocator, &.{wrong_shape}, .{}));
    const rank_one = try t.InputTensor.fromSlice(i32, "IN0", &.{4}, &.{ 1, 2, 3, 4 });
    try testing.expectError(error.InvalidShape, scheduler.submit(testing.allocator, &.{rank_one}, .{}));
    const too_large = try t.InputTensor.fromSlice(i32, "IN0", &.{ 9, 4 }, &@as([36]i32, @splat(1)));
    try testing.expectError(error.BatchTooLarge, scheduler.submit(testing.allocator, &.{too_large}, .{}));
    var malformed = input;
    malformed.shape = &.{ -1, 4 };
    try testing.expectError(error.InvalidShape, scheduler.submit(testing.allocator, &.{malformed}, .{}));
    malformed = input;
    malformed.data = input.data[0..4];
    try testing.expectError(error.ElementCountMismatch, scheduler.submit(testing.allocator, &.{malformed}, .{}));
    scheduler.shutdown();
    try testing.expectEqual(@as(u64, 0), scheduler.stats().dispatched_batches);
    try testing.expectError(error.SchedulerClosed, scheduler.submit(testing.allocator, &.{input}, .{}));
    scheduler.deinit();
    try testing.expectError(error.SchedulerClosed, scheduler.submit(testing.allocator, &.{input}, .{}));
}

test "scheduler rejects a model already using native dynamic batching" {
    var server = try t.Server.init(testing.allocator, .{
        .model_repository = "tests/fixtures/models",
        .backend_directory = ".zig-cache/phase2/backends",
        .log_info = false,
    });
    defer destroyServer(&server);
    try testing.expectError(error.NativeBatchingEnabled, t.Scheduler.init(testing.allocator, &server, "batching", -1, .{}));
}

test "asynchronous execution errors complete futures and leave worker reusable" {
    var server = try startServer();
    defer destroyServer(&server);
    var scheduler = try t.Scheduler.init(testing.allocator, &server, "phase4_fixed", -1, .{
        .worker_count = 1,
        .max_wait_us = 1_000_000,
    });
    defer scheduler.deinit();
    var bad_values: [16]i32 = @splat(1);
    bad_values[0] = std.math.minInt(i32);
    var failed = try submit(&scheduler, &.{ 4, 4 }, &bad_values);
    defer failed.deinit();
    var partner = try submit(&scheduler, &.{ 4, 4 }, &@as([16]i32, @splat(2)));
    defer partner.deinit();
    try testing.expectError(error.InvalidArg, failed.wait());
    try testing.expectError(error.InvalidArg, partner.wait());
    var good = try submit(&scheduler, &.{ 1, 4 }, &.{ 1, 2, 3, 4 });
    defer good.deinit();
    scheduler.shutdown();
    var response = try good.wait();
    defer response.deinit();
    try expectOutput(&response, &.{ 1, 4 }, &.{ 1, 2, 3, 4 });
    const stats = scheduler.stats();
    try testing.expectEqual(@as(u64, 3), stats.accepted_requests);
    try testing.expectEqual(@as(u64, 2), stats.failed_requests);
    try testing.expectEqual(@as(u64, 2), stats.dispatched_batches);
}

fn schedulerAllocationCase(alloc: std.mem.Allocator, server: *t.Server) !void {
    var scheduler = try t.Scheduler.init(alloc, server, "phase4_fixed", -1, .{
        .worker_count = 1,
        .max_wait_us = 0,
    });
    defer scheduler.deinit();
    const input = try t.InputTensor.fromSlice(i32, "IN0", &.{ 1, 4 }, &.{ 4, 3, 2, 1 });
    var future = try scheduler.submit(alloc, &.{input}, .{ .outputs = &.{"OUT0"} });
    defer future.deinit();
    // The parent allocator can fail while copying a completed result. Future
    // and scheduler cleanup must still release all accepted native work.
    var response = try future.wait();
    defer response.deinit();
    try expectOutput(&response, &.{ 1, 4 }, &.{ 4, 3, 2, 1 });
}

test "scheduler allocation failures release partial initialization and completed work" {
    var server = try startServer();
    defer destroyServer(&server);
    try testing.checkAllAllocationFailures(testing.allocator, schedulerAllocationCase, .{&server});
}

test "malformed aggregate output fails every split including singleton batches" {
    var server = try startServer();
    defer destroyServer(&server);
    var scheduler = try t.Scheduler.init(testing.allocator, &server, "phase4_malformed", -1, .{
        .worker_count = 1,
        .max_wait_us = 1_000,
    });
    defer scheduler.deinit();
    var singleton = try submit(&scheduler, &.{ 1, 4 }, &.{ 1, 2, 3, 4 });
    defer singleton.deinit();
    try testing.expectError(error.InvalidBatchOutput, singleton.wait());
    var first = try submit(&scheduler, &.{ 4, 4 }, &@as([16]i32, @splat(1)));
    defer first.deinit();
    var second = try submit(&scheduler, &.{ 4, 4 }, &@as([16]i32, @splat(2)));
    defer second.deinit();
    try testing.expectError(error.InvalidBatchOutput, first.wait());
    try testing.expectError(error.InvalidBatchOutput, second.wait());
    scheduler.shutdown();
    try testing.expectEqual(@as(u64, 3), scheduler.stats().failed_requests);
}

fn waitForDispatch(scheduler: *t.Scheduler) !void {
    // The fixture's INT32_MAX request occupies its sole worker for one second.
    // Poll only until dispatch begins, with an independent six-second deadline.
    for (0..6_000) |_| {
        if (scheduler.stats().dispatched_batches != 0) return;
        try std.Io.sleep(testing.io, .fromMilliseconds(1), .awake);
    }
    return error.DispatchTimedOut;
}

test "bounded queue rejects overflow and shutdown drains accepted work" {
    var server = try startServer();
    defer destroyServer(&server);
    var scheduler = try t.Scheduler.init(testing.allocator, &server, "phase4_fixed", -1, .{
        .worker_count = 1,
        .max_wait_us = 0,
        // Capacity includes the executing request and the queued request.
        .queue_capacity = 2,
    });
    defer scheduler.deinit();
    var busy = try submit(&scheduler, &.{ 1, 4 }, &.{ std.math.maxInt(i32), 2, 3, 4 });
    defer busy.deinit();
    try waitForDispatch(&scheduler);
    var queued = try submit(&scheduler, &.{ 1, 4 }, &.{ 5, 6, 7, 8 });
    defer queued.deinit();
    try testing.expectError(error.QueueFull, submit(&scheduler, &.{ 1, 4 }, &.{ 9, 10, 11, 12 }));
    scheduler.shutdown();
    var a = try busy.wait();
    defer a.deinit();
    var b = try queued.wait();
    defer b.deinit();
    try expectOutput(&a, &.{ 1, 4 }, &.{ std.math.maxInt(i32), 2, 3, 4 });
    try expectOutput(&b, &.{ 1, 4 }, &.{ 5, 6, 7, 8 });
    const stats = scheduler.stats();
    try testing.expectEqual(@as(u64, 2), stats.accepted_requests);
    try testing.expectEqual(@as(u64, 1), stats.rejected_requests);
    try testing.expectEqual(@as(u64, 2), stats.queue_peak);
    try testing.expect(stats.total_queue_wait_ns > 0);
}

const Producer = struct {
    scheduler: *t.Scheduler,
    started: *std.atomic.Value(usize),
    failure: ?anyerror = null,
    accepted: u64 = 0,
    rejected: u64 = 0,

    fn run(self: *Producer) void {
        self.perform() catch |err| {
            self.failure = err;
        };
    }

    fn perform(self: *Producer) !void {
        var futures: [16]?t.ScheduledInference = @splat(null);
        defer for (&futures) |*future| if (future.*) |*value| value.deinit();
        const input = try t.InputTensor.fromSlice(i32, "IN0", &.{ 1, 4 }, &.{ 1, 2, 3, 4 });
        for (&futures, 0..) |*future, i| {
            future.* = self.scheduler.submit(std.heap.c_allocator, &.{input}, .{}) catch |err| switch (err) {
                error.QueueFull, error.SchedulerClosed => rejected: {
                    self.rejected += 1;
                    break :rejected null;
                },
                else => return err,
            };
            if (future.* != null) self.accepted += 1;
            if (i == 0) _ = self.started.fetchAdd(1, .release);
            try std.Io.sleep(testing.io, .fromMilliseconds(1), .awake);
        }
        for (&futures) |*future| if (future.*) |*value| {
            var response = try value.wait();
            defer response.deinit();
            try expectOutput(&response, &.{ 1, 4 }, &.{ 1, 2, 3, 4 });
        };
    }
};

test "concurrent producers racing shutdown complete every accepted future" {
    var server = try startServer();
    defer destroyServer(&server);
    var scheduler = try t.Scheduler.init(testing.allocator, &server, "phase4_fixed", -1, .{
        .worker_count = 2,
        .max_wait_us = 1_000,
        .queue_capacity = 8,
    });
    defer scheduler.deinit();
    var started: std.atomic.Value(usize) = .init(0);
    var producers = [_]Producer{
        .{ .scheduler = &scheduler, .started = &started },
        .{ .scheduler = &scheduler, .started = &started },
        .{ .scheduler = &scheduler, .started = &started },
        .{ .scheduler = &scheduler, .started = &started },
    };
    var threads: [producers.len]std.Thread = undefined;
    var running: usize = 0;
    defer for (threads[0..running]) |thread| thread.join();
    for (&producers) |*producer| {
        threads[running] = try std.Thread.spawn(.{}, Producer.run, .{producer});
        running += 1;
    }
    for (0..6_000) |_| {
        if (started.load(.acquire) == producers.len) break;
        try std.Io.sleep(testing.io, .fromMilliseconds(1), .awake);
    }
    scheduler.shutdown();
    scheduler.shutdown();
    for (threads[0..running]) |thread| thread.join();
    running = 0;
    try testing.expectEqual(producers.len, started.load(.acquire));
    var accepted: u64 = 0;
    var rejected: u64 = 0;
    for (producers) |producer| {
        if (producer.failure) |err| return err;
        accepted += producer.accepted;
        rejected += producer.rejected;
    }
    try testing.expect(accepted > 0);
    try testing.expect(rejected > 0);
    try testing.expectEqual(@as(u64, producers.len * 16), accepted + rejected);
    const stats = scheduler.stats();
    try testing.expectEqual(accepted, stats.accepted_requests);
    try testing.expectEqual(rejected, stats.rejected_requests);
    try testing.expectEqual(accepted, stats.dispatched_samples);
    try testing.expectEqual(@as(u64, 0), stats.failed_requests);
}
