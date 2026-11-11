const std = @import("std");
const t = @import("tritonserver_zig");

fn printCase(model: []const u8, tensor: *const t.OutputTensor) !void {
    std.debug.print("Inference {s}: {s} int32 shape=[", .{ model, tensor.name });
    for (tensor.shape, 0..) |dim, i| {
        if (i != 0) std.debug.print(", ", .{});
        std.debug.print("{d}", .{dim});
    }
    std.debug.print("] values=[", .{});
    for (try tensor.asSlice(i32), 0..) |value, i| {
        if (i != 0) std.debug.print(", ", .{});
        std.debug.print("{d}", .{value});
    }
    std.debug.print("] (matches input)\n", .{});
}

pub fn run(alloc: std.mem.Allocator, server: *const t.Server) !void {
    {
        var request = try t.Request.init(alloc, server, "nonbatching", -1);
        defer request.deinit() catch |err| std.log.err("request cleanup: {s}", .{@errorName(err)});
        const input = [_]i32{ 42, -7, std.math.minInt(i32), std.math.maxInt(i32) };
        try request.addInput(i32, "IN0", &.{4}, &input);
        try request.requestOutput("OUT0");
        var result = try server.inferSync(alloc, &request);
        defer result.deinit();
        const output = try result.output("OUT0");
        if (!std.mem.eql(i32, &input, try output.asSlice(i32))) return error.InferenceMismatch;
        try printCase("nonbatching", output);
    }
    {
        var request = try t.Request.init(alloc, server, "batching", 1);
        defer request.deinit() catch |err| std.log.err("request cleanup: {s}", .{@errorName(err)});
        var input: [32]i32 = undefined;
        for (&input, 0..) |*value, i| value.* = @as(i32, @intCast(i)) - 16;
        try request.addInput(i32, "IN0", &.{ 8, 4 }, &input);
        // A full batch dispatches immediately instead of waiting for the
        // reference model's configured dynamic-batching delay.
        var pending = try server.inferAsync(alloc, &request);
        defer pending.deinit();
        var result = try pending.wait();
        defer result.deinit();
        const output = try result.output("OUT0");
        if (!std.mem.eql(i32, &input, try output.asSlice(i32))) return error.InferenceMismatch;
        try printCase("batching", output);
    }
    std.debug.print("Inference demo passed: both model outputs match their inputs.\n", .{});
}
