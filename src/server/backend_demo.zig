const std = @import("std");
const t = @import("tritonserver_zig");

pub fn main() !void {
    const alloc = std.heap.c_allocator;
    var plugin = try t.backend.PluginLibrary.load(".zig-cache/phase5/backends/zig_passthrough/libtriton_zig_passthrough.so");
    defer plugin.deinit();
    var server = try t.Server.init(alloc, .{
        .model_repository = "tests/fixtures/backend_models/valid",
        .backend_directory = ".zig-cache/phase5/backends",
        .log_info = false,
    });
    defer server.deinit() catch |err| std.log.err("server cleanup: {s}", .{@errorName(err)});
    var request = try t.Request.init(alloc, &server, "zig_nonbatch", 1);
    defer request.deinit() catch |err| std.log.err("request cleanup: {s}", .{@errorName(err)});
    const values = [_]i32{ 42, -7, 0, 9, std.math.minInt(i32), std.math.maxInt(i32) };
    try request.addInput(i32, "IN0", &.{ 2, 3 }, &values);
    try request.requestOutput("OUT0");
    var result = try server.inferSync(alloc, &request);
    defer result.deinit();
    const output = try result.output("OUT0");
    if (!std.mem.eql(i64, output.shape, &.{ 2, 3 }) or
        !std.mem.eql(i32, try output.asSlice(i32), &values)) return error.PassThroughMismatch;
    std.debug.print("Zig backend passed: Triton loaded the .so and returned all 6 INT32 values unchanged.\n", .{});
}
