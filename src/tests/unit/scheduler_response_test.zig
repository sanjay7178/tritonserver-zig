const std = @import("std");
const response = @import("../../core/inference/response.zig");
const tensor = @import("../../core/inference/tensor.zig");
const OwnedArena = @import("../../core/internal/owned_arena.zig").OwnedArena;
const DataType = @import("../../core/bindings/datatype.zig").DataType;
const testing = std.testing;

const Spec = struct { name: []const u8 = "OUT0", datatype: DataType = .int32, shape: []const i64, data: []const u8 };

fn makeResponse(specs: []const Spec) !response.InferenceResponse {
    var storage = try OwnedArena.init(testing.allocator);
    errdefer storage.deinit();
    const alloc = storage.allocator();
    const outputs = try alloc.alloc(tensor.OutputTensor, specs.len);
    for (specs, outputs) |spec, *output| {
        const data = try alloc.alignedAlloc(u8, .@"16", spec.data.len);
        @memcpy(data, spec.data);
        output.* = .{ .name = try alloc.dupe(u8, spec.name), .datatype = spec.datatype, .shape = try alloc.dupe(i64, spec.shape), .data = data };
    }
    return .{ .storage = storage, .model_name = try alloc.dupe(u8, "scheduler_fixture"), .model_version = 7, .outputs = outputs };
}

test "response clone remains independently owned after source destruction" {
    const values = [_]i32{ 1, 2, 3, 4 };
    var source = try makeResponse(&.{.{ .shape = &.{ 2, 2 }, .data = std.mem.sliceAsBytes(&values) }});
    defer source.deinit();
    var copy = try response.clone(testing.allocator, &source);
    defer copy.deinit();
    @memset(@constCast(source.outputs[0].data), 0);
    source.deinit();
    try testing.expectEqualStrings("scheduler_fixture", copy.model_name);
    try testing.expectEqual(@as(i64, 7), copy.model_version);
    const output = try copy.output("OUT0");
    try testing.expectEqualSlices(i64, &.{ 2, 2 }, output.shape);
    try testing.expectEqualSlices(i32, &values, try output.asSlice(i32));
    try testing.expectEqual(@as(usize, 0), @intFromPtr(output.data.ptr) % 16);
}

test "batch splitting preserves multiple outputs and unequal request batch sizes" {
    const integers = [_]i32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const floats = [_]f32{ 0.5, 1.5, 2.5, 3.5 };
    var source = try makeResponse(&.{
        .{ .shape = &.{ 4, 2 }, .data = std.mem.sliceAsBytes(&integers) },
        .{ .name = "OUT1", .datatype = .fp32, .shape = &.{4}, .data = std.mem.sliceAsBytes(&floats) },
    });
    defer source.deinit();
    var middle = try response.sliceBatch(testing.allocator, &source, 1, 2, 4);
    defer middle.deinit();
    var last = try response.sliceBatch(testing.allocator, &source, 3, 1, 4);
    defer last.deinit();
    source.deinit();
    try testing.expectEqualSlices(i64, &.{ 2, 2 }, middle.outputs[0].shape);
    try testing.expectEqualSlices(i32, integers[2..6], try middle.outputs[0].asSlice(i32));
    try testing.expectEqualSlices(i64, &.{2}, middle.outputs[1].shape);
    try testing.expectEqualSlices(f32, floats[1..3], try middle.outputs[1].asSlice(f32));
    try testing.expectEqualSlices(i32, integers[6..8], try last.outputs[0].asSlice(i32));
    try testing.expectEqual(@as(usize, 0), @intFromPtr(last.outputs[0].data.ptr) % 16);
}

test "batch splitting handles empty tensor rows and rejects invalid ranges" {
    var source = try makeResponse(&.{.{ .shape = &.{ 4, 0 }, .data = &.{} }});
    defer source.deinit();
    var part = try response.sliceBatch(testing.allocator, &source, 1, 2, 4);
    defer part.deinit();
    try testing.expectEqualSlices(i64, &.{ 2, 0 }, part.outputs[0].shape);
    try testing.expectEqual(@as(usize, 0), part.outputs[0].data.len);
    try testing.expectError(error.InvalidBatchRange, response.sliceBatch(testing.allocator, &source, 0, 0, 4));
    try testing.expectError(error.InvalidBatchRange, response.sliceBatch(testing.allocator, &source, 0, 1, 0));
    try testing.expectError(error.InvalidBatchRange, response.sliceBatch(testing.allocator, &source, 4, 1, 4));
    try testing.expectError(error.InvalidBatchRange, response.sliceBatch(testing.allocator, &source, 2, 3, 4));
    if (@bitSizeOf(usize) > 63)
        try testing.expectError(error.ShapeOverflow, response.sliceBatch(testing.allocator, &source, 0, 1, std.math.maxInt(usize)));
}

test "response copying rejects malformed complete buffers and non-batch-preserving shapes" {
    const values = [_]i32{ 1, 2, 3, 4 };
    var truncated = try makeResponse(&.{.{ .shape = &.{ 4, 2 }, .data = std.mem.sliceAsBytes(&values) }});
    defer truncated.deinit();
    try testing.expectError(error.InvalidOutputBuffer, response.clone(testing.allocator, &truncated));
    try testing.expectError(error.InvalidOutputBuffer, response.sliceBatch(testing.allocator, &truncated, 0, 1, 4));
    var wrong_batch = try makeResponse(&.{.{ .shape = &.{ 2, 2 }, .data = std.mem.sliceAsBytes(&values) }});
    defer wrong_batch.deinit();
    try testing.expectError(error.InvalidBatchShape, response.sliceBatch(testing.allocator, &wrong_batch, 0, 1, 4));
    var scalar = try makeResponse(&.{.{ .shape = &.{}, .data = std.mem.sliceAsBytes(values[0..1]) }});
    defer scalar.deinit();
    try testing.expectError(error.InvalidBatchShape, response.sliceBatch(testing.allocator, &scalar, 0, 1, 1));
    var unsupported = try makeResponse(&.{.{ .datatype = .bytes, .shape = &.{1}, .data = &.{} }});
    defer unsupported.deinit();
    try testing.expectError(error.UnsupportedDataType, response.clone(testing.allocator, &unsupported));
    var invalid_shape = try makeResponse(&.{.{ .shape = &.{-1}, .data = &.{} }});
    defer invalid_shape.deinit();
    try testing.expectError(error.InvalidShape, response.clone(testing.allocator, &invalid_shape));
    var overflowing = try makeResponse(&.{.{ .datatype = .fp64, .shape = &.{std.math.maxInt(i64)}, .data = &.{} }});
    defer overflowing.deinit();
    try testing.expectError(error.ShapeOverflow, response.clone(testing.allocator, &overflowing));
    // A later malformed output must release the copy of earlier valid outputs.
    var malformed_later = try makeResponse(&.{
        .{ .shape = &.{4}, .data = std.mem.sliceAsBytes(&values) },
        .{ .name = "OUT1", .shape = &.{5}, .data = std.mem.sliceAsBytes(&values) },
    });
    defer malformed_later.deinit();
    try testing.expectError(error.InvalidOutputBuffer, response.clone(testing.allocator, &malformed_later));
    try testing.expectError(error.InvalidOutputBuffer, response.sliceBatch(testing.allocator, &malformed_later, 1, 2, 4));
}

fn cloneWithAllocator(alloc: std.mem.Allocator, source: *const response.InferenceResponse) !void {
    var copy = try response.clone(alloc, source);
    defer copy.deinit();
}

fn splitWithAllocator(alloc: std.mem.Allocator, source: *const response.InferenceResponse) !void {
    var copy = try response.sliceBatch(alloc, source, 1, 2, 4);
    defer copy.deinit();
}

test "response clone and split clean up every allocation failure" {
    const values = [_]i32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    var source = try makeResponse(&.{
        .{ .shape = &.{ 4, 2 }, .data = std.mem.sliceAsBytes(&values) },
        .{ .name = "OUT1", .shape = &.{ 4, 2 }, .data = std.mem.sliceAsBytes(&values) },
    });
    defer source.deinit();
    try testing.checkAllAllocationFailures(testing.allocator, cloneWithAllocator, .{&source});
    try testing.checkAllAllocationFailures(testing.allocator, splitWithAllocator, .{&source});
}
