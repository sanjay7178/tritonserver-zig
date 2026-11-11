const std = @import("std");
const t = @import("../../root.zig");
const testing = std.testing;

const backend_path = ".zig-cache/phase2/backends";

test "server validates paths, cleans up failed initialization and rejects closed operations" {
    try testing.expectError(error.InvalidPath, t.Server.init(testing.allocator, .{ .model_repository = "x\x00y" }));
    try testing.expectError(error.InvalidPath, t.Server.init(testing.allocator, .{ .model_repository = ".", .backend_directory = "x\x00y" }));
    var temp = testing.tmpDir(.{});
    defer temp.cleanup();
    const path = try temp.dir.realPathFileAlloc(testing.io, ".", testing.allocator);
    defer testing.allocator.free(path);
    const missing = try std.fs.path.join(testing.allocator, &.{ path, "missing" });
    defer testing.allocator.free(missing);
    const failed = t.Server.init(testing.allocator, .{ .model_repository = missing, .log_info = false });
    if (failed) |created| {
        var unexpected = created;
        try unexpected.deinit();
        return error.ExpectedStartupFailure;
    } else |_| {}
    var server = try t.Server.init(testing.allocator, .{ .model_repository = path, .log_info = false });
    defer server.deinit() catch @panic("server cleanup failed");
    try testing.expect(try server.isLive());
    try testing.expect(try server.isReady());
    try server.pollRepository();
    var empty = try server.listModels(testing.allocator);
    defer empty.deinit();
    try testing.expectEqual(@as(usize, 0), empty.models.len);
    try server.stop();
    try server.stop();
    try testing.expectError(error.Unavailable, server.isLive());
    try testing.expectError(error.ServerStopped, server.pollRepository());
    try server.deinit();
    try server.deinit();
    try server.stop();
    try testing.expectError(error.ServerClosed, server.isLive());
    try testing.expectError(error.ServerClosed, server.isReady());
    try testing.expectError(error.ServerClosed, server.pollRepository());
    try testing.expectError(error.ServerClosed, server.listModels(testing.allocator));
}

test "real CPU models expose typed metadata and snapshots outlive the server" {
    var server = try t.Server.init(testing.allocator, .{ .model_repository = "tests/fixtures/models", .backend_directory = backend_path, .log_info = false });
    defer server.deinit() catch @panic("server cleanup failed");
    try testing.expect(try server.isReady());
    var list = try server.listModels(testing.allocator);
    defer list.deinit();
    try server.stop();
    try server.deinit();
    try testing.expectEqual(@as(usize, 2), list.models.len);
    for (list.models, 0..) |info, i| {
        try testing.expectEqualStrings(if (i == 0) "batching" else "nonbatching", info.name);
        try testing.expectEqual(@as(?i64, 1), info.version);
        try testing.expectEqualStrings("READY", info.state);
        const detail = info.metadata.?;
        try testing.expectEqualStrings("minimal", detail.platform);
        try testing.expectEqual(@as(u32, if (i == 0) 8 else 0), detail.max_batch_size);
        try testing.expectEqual(@as(usize, 1), detail.inputs.len);
        try testing.expectEqual(@as(usize, 1), detail.outputs.len);
        try testing.expectEqual(t.DataType.int32, detail.inputs[0].datatype);
        try testing.expectEqual(t.DataType.int32, detail.outputs[0].datatype);
        try testing.expectEqualStrings("IN0", detail.inputs[0].name);
        try testing.expectEqualStrings("OUT0", detail.outputs[0].name);
        const shape: []const i64 = if (i == 0) &.{ -1, 4 } else &.{4};
        try testing.expectEqualSlices(i64, shape, detail.inputs[0].shape);
        try testing.expectEqualSlices(i64, shape, detail.outputs[0].shape);
    }
}

test "unavailable models retain failure reasons when fail-fast is disabled" {
    var temp = testing.tmpDir(.{});
    defer temp.cleanup();
    try temp.dir.createDirPath(testing.io, "broken/1");
    try temp.dir.writeFile(testing.io, .{ .sub_path = "broken/config.pbtxt", .data = "backend: \"phase2_missing_backend\"\nmax_batch_size: 0\ninput [{name: \"IN0\" data_type: TYPE_INT32 dims: [4]}]\noutput [{name: \"OUT0\" data_type: TYPE_INT32 dims: [4]}]\ninstance_group [{kind: KIND_CPU}]\n" });
    const path = try temp.dir.realPathFileAlloc(testing.io, ".", testing.allocator);
    defer testing.allocator.free(path);
    var server = try t.Server.init(testing.allocator, .{ .model_repository = path, .backend_directory = backend_path, .exit_on_error = false, .log_info = false });
    defer server.deinit() catch @panic("server cleanup failed");
    try testing.expect(try server.isLive());
    try testing.expect(!try server.isReady());
    var list = try server.listModels(testing.allocator);
    defer list.deinit();
    try testing.expectEqual(@as(usize, 1), list.models.len);
    try testing.expectEqualStrings("UNAVAILABLE", list.models[0].state);
    try testing.expect(list.models[0].reason != null);
    try testing.expect(list.models[0].reason.?.len > 0);
    try testing.expect(list.models[0].metadata == null);
}

fn inventoryAllocationCase(alloc: std.mem.Allocator, path: []const u8) !void {
    var repo = try t.Repository.scan(alloc, testing.io, path);
    defer repo.deinit();
}
fn serverAllocationCase(alloc: std.mem.Allocator, path: []const u8) !void {
    var server = try t.Server.init(alloc, .{ .model_repository = path, .log_info = false });
    defer server.deinit() catch @panic("server cleanup failed");
}

test "repository and server initialization release each partial allocation" {
    var temp = testing.tmpDir(.{});
    defer temp.cleanup();
    const path = try temp.dir.realPathFileAlloc(testing.io, ".", testing.allocator);
    defer testing.allocator.free(path);
    try testing.checkAllAllocationFailures(testing.allocator, serverAllocationCase, .{path});
    try temp.dir.createDirPath(testing.io, "model/1");
    try testing.checkAllAllocationFailures(testing.allocator, inventoryAllocationCase, .{path});
}

test "repository polling loads real models with multiple versions" {
    var temp = testing.tmpDir(.{});
    defer temp.cleanup();
    const path = try temp.dir.realPathFileAlloc(testing.io, ".", testing.allocator);
    defer testing.allocator.free(path);
    var server = try t.Server.init(testing.allocator, .{ .model_repository = path, .backend_directory = backend_path, .log_info = false });
    defer server.deinit() catch @panic("server cleanup failed");
    var before = try server.listModels(testing.allocator);
    defer before.deinit();
    try testing.expectEqual(@as(usize, 0), before.models.len);
    try temp.dir.createDirPath(testing.io, "added/2");
    try temp.dir.createDirPath(testing.io, "added/10");
    try temp.dir.writeFile(testing.io, .{ .sub_path = "added/config.pbtxt", .data =
        \\backend: "minimal"
        \\max_batch_size: 0
        \\version_policy { all {} }
        \\input [{name: "IN0" data_type: TYPE_INT32 dims: [4]}]
        \\output [{name: "OUT0" data_type: TYPE_INT32 dims: [4]}]
        \\instance_group [{kind: KIND_CPU}]
    });
    try server.pollRepository();
    try testing.expect(try server.isReady());
    var after = try server.listModels(testing.allocator);
    defer after.deinit();
    try testing.expectEqual(@as(usize, 2), after.models.len);
    for (after.models, [_]i64{ 2, 10 }) |entry, version| {
        try testing.expectEqualStrings("added", entry.name);
        try testing.expectEqual(@as(?i64, version), entry.version);
        try testing.expectEqualStrings("READY", entry.state);
        try testing.expect(entry.metadata != null);
    }
}
