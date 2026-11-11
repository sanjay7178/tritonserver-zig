const std = @import("std");
const t = @import("../../root.zig");
const model = @import("../../core/model/metadata.zig");
const testing = std.testing;

const index_json = "[{\"name\":\"demo\",\"version\":\"1\",\"state\":\"READY\"}]";
const metadata_json = "{\"platform\":\"minimal\",\"inputs\":[{\"name\":\"IN0\",\"datatype\":\"INT32\",\"shape\":[-1,4]}],\"outputs\":[{\"name\":\"OUT0\",\"datatype\":\"INT32\",\"shape\":[-1,4]}]}";

fn snapshotWithAllocator(alloc: std.mem.Allocator) !void {
    var list = try model.parseIndex(alloc, index_json);
    defer list.deinit();
    list.models[0].metadata = try model.parseMetadata(list.storage.allocator(), metadata_json, "{\"max_batch_size\":8}");
    try testing.expectEqual(@as(u32, 8), list.models[0].metadata.?.max_batch_size);
}

test "model parsing releases partial snapshots at every allocation failure" {
    try testing.checkAllAllocationFailures(testing.allocator, snapshotWithAllocator, .{});
}

test "metadata snapshot owns strings, protocol shapes and batch limits" {
    const index = try testing.allocator.dupe(u8, index_json);
    defer testing.allocator.free(index);
    const metadata = try testing.allocator.dupe(u8, metadata_json);
    defer testing.allocator.free(metadata);
    var list = try model.parseIndex(testing.allocator, index);
    defer list.deinit();
    list.models[0].metadata = try model.parseMetadata(list.storage.allocator(), metadata, "{\"max_batch_size\":8,\"unrelated\":true}");
    @memset(index, 'x');
    @memset(metadata, 'x');
    const info = list.models[0];
    const detail = info.metadata.?;
    try testing.expectEqualStrings("demo", info.name);
    try testing.expectEqualStrings("minimal", detail.platform);
    try testing.expectEqualStrings("IN0", detail.inputs[0].name);
    try testing.expectEqual(t.DataType.int32, detail.inputs[0].datatype);
    try testing.expectEqualSlices(i64, &.{ -1, 4 }, detail.inputs[0].shape);
    try testing.expectEqual(@as(u32, 8), detail.max_batch_size);
    const nonbatched = try model.parseMetadata(list.storage.allocator(), metadata_json, "{}");
    try testing.expectEqual(@as(u32, 0), nonbatched.max_batch_size);
    list.deinit(); // Idempotent on the same owner.
}

test "index ordering includes unavailable and name-only models" {
    var list = try model.parseIndex(testing.allocator, "[{\"name\":\"z\",\"version\":\"10\",\"state\":\"UNAVAILABLE\",\"reason\":\"missing backend\"}," ++
        "{\"name\":\"a\"},{\"name\":\"z\",\"version\":\"2\",\"state\":\"READY\"}]");
    defer list.deinit();
    try testing.expectEqualStrings("a", list.models[0].name);
    try testing.expectEqual(@as(?i64, null), list.models[0].version);
    try testing.expectEqual(@as(?i64, 2), list.models[1].version);
    try testing.expectEqualStrings("missing backend", list.models[2].reason.?);
    try testing.expectEqual(@as(?t.ModelMetadata, null), list.models[2].metadata);
}

test "malformed index, versions, dtypes, shapes and batch limits fail explicitly" {
    try testing.expectError(error.UnexpectedEndOfInput, model.parseIndex(testing.allocator, "["));
    try testing.expectError(error.InvalidModelVersion, model.parseIndex(testing.allocator, "[{\"name\":\"x\",\"version\":\"bad\"}]"));
    try testing.expectError(error.InvalidModelVersion, model.parseIndex(testing.allocator, "[{\"name\":\"x\",\"state\":\"READY\"}]"));
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    try testing.expectError(error.UnsupportedDataType, model.parseMetadata(alloc, "{\"inputs\":[{\"name\":\"x\",\"datatype\":\"NEW_TYPE\",\"shape\":[4]}],\"outputs\":[]}", "{}"));
    try testing.expectError(error.InvalidTensorShape, model.parseMetadata(alloc, "{\"inputs\":[{\"name\":\"x\",\"datatype\":\"INT32\",\"shape\":[-2]}],\"outputs\":[]}", "{}"));
    try testing.expectError(error.InvalidBatchLimit, model.parseMetadata(alloc, metadata_json, "{\"max_batch_size\":-1}"));
    try testing.expectError(error.UnexpectedEndOfInput, model.parseMetadata(alloc, "", "{}"));
}
