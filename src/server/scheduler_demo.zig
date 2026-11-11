const std = @import("std");
const t = @import("tritonserver_zig");

/// Exercise the software batcher with distinguishable, independently owned jobs.
pub fn main(init: std.process.Init) !void {
    var repository = try t.Repository.scan(init.gpa, init.io, "tests/fixtures/scheduler_models");
    defer repository.deinit();
    var server = try t.Server.init(init.gpa, .{
        .model_repository = repository.path,
        .backend_directory = ".zig-cache/phase2/backends",
        .log_info = false,
    });
    defer server.deinit() catch |err| std.log.err("server cleanup: {s}", .{@errorName(err)});
    var scheduler = try t.Scheduler.init(init.gpa, init.io, &server, "phase4_fixed", 1, .{
        .max_batch_size = 8,
        .max_wait_us = 5_000,
        .worker_count = 2,
    });
    defer scheduler.deinit();

    var jobs: [16]t.ScheduledInference = undefined;
    var submitted: usize = 0;
    defer for (jobs[0..submitted]) |*job| job.deinit();
    for (&jobs, 0..) |*job, index| {
        const value: i32 = @intCast(index);
        const data = [_]i32{ value, -value, value + 100, value * 7 };
        const input = try t.InputTensor.fromSlice(i32, "IN0", &.{ 1, 4 }, &data);
        job.* = try scheduler.submit(init.gpa, &.{input}, .{ .outputs = &.{ "OUT0", "BATCH_INFO" } });
        submitted += 1;
    }

    var saw_merged_execution = false;
    for (jobs[0..submitted], 0..) |*job, index| {
        var response = try job.wait();
        defer response.deinit();
        const value: i32 = @intCast(index);
        const expected = [_]i32{ value, -value, value + 100, value * 7 };
        const output = try response.output("OUT0");
        if (!std.mem.eql(i64, output.shape, &.{ 1, 4 }) or
            !std.mem.eql(i32, try output.asSlice(i32), &expected))
            return error.ScheduledOutputMismatch;
        const diagnostic = try (try response.output("BATCH_INFO")).asSlice(i64);
        if (diagnostic.len != 3 or diagnostic[1] < 1 or diagnostic[1] > 8)
            return error.InvalidBatchDiagnostic;
        saw_merged_execution = saw_merged_execution or diagnostic[1] > 1;
        std.debug.print("job {d}: values verified; execution={d}, batch_samples={d}, native_requests={d}\n", .{
            index, diagnostic[0], diagnostic[1], diagnostic[2],
        });
    }
    scheduler.shutdown();
    const stats = scheduler.stats();
    if (!saw_merged_execution or stats.max_batch_samples <= 1 or
        stats.accepted_requests != submitted or stats.failed_requests != 0)
        return error.BatchingNotObserved;
    scheduler.deinit();
    try server.stop();
    try server.deinit();
    std.debug.print("Scheduler demo passed: {d} jobs, {d} batches, largest batch {d}.\n", .{
        stats.accepted_requests, stats.dispatched_batches, stats.max_batch_samples,
    });
}
