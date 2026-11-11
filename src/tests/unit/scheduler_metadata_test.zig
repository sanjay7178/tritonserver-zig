const std = @import("std");
const metadata = @import("../../core/model/metadata.zig");
const testing = std.testing;
const empty_metadata = "{\"platform\":\"test\",\"inputs\":[],\"outputs\":[]}";

test "scheduler capability flags are parsed from snake_case model config" {
    const Case = struct { config: []const u8, expected: metadata.SchedulingInfo };
    for ([_]Case{
        .{ .config = "{}", .expected = .{} },
        .{ .config = "{\"dynamic_batching\":{}}", .expected = .{ .dynamic_batching = true } },
        .{ .config = "{\"sequence_batching\":{\"max_sequence_idle_microseconds\":1000}}", .expected = .{ .sequence_batching = true } },
        .{ .config = "{\"model_transaction_policy\":{\"decoupled\":true}}", .expected = .{ .decoupled = true } },
        .{ .config = "{\"model_transaction_policy\":{\"decoupled\":false}}", .expected = .{} },
        .{ .config = "{\"input\":[{\"name\":\"IN0\",\"is_shape_tensor\":true}]}", .expected = .{ .special_tensors = true } },
        .{ .config = "{\"output\":[{\"name\":\"OUT0\",\"is_shape_tensor\":true}]}", .expected = .{ .special_tensors = true } },
        .{ .config = "{\"input\":[{\"allow_ragged_batch\":true}]}", .expected = .{ .special_tensors = true } },
        .{ .config = "{\"input\":[{\"optional\":true}]}", .expected = .{ .special_tensors = true } },
        .{ .config = "{\"batch_input\":[{\"kind\":\"BATCH_ELEMENT_COUNT\"}]}", .expected = .{ .special_tensors = true } },
        .{ .config = "{\"batch_output\":[{\"kind\":\"BATCH_SCATTER_WITH_INPUT_SHAPE\"}]}", .expected = .{ .special_tensors = true } },
        .{ .config = "{\"input\":[{\"optional\":false,\"is_shape_tensor\":false,\"allow_ragged_batch\":false}],\"batch_input\":[],\"batch_output\":[]}", .expected = .{} },
    }) |case| {
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const parsed = try metadata.parseMetadata(arena.allocator(), empty_metadata, case.config);
        try testing.expectEqualDeep(case.expected, parsed.scheduling);
    }
}

test "scheduling flags preserve metadata shapes and batch limits" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const parsed = try metadata.parseMetadata(arena.allocator(),
        \\{"inputs":[{"name":"IN0","datatype":"INT32","shape":[-1,4]}],"outputs":[{"name":"OUT0","datatype":"INT32","shape":[-1,4]}]}
    ,
        \\{"max_batch_size":8,"dynamic_batching":{},"sequence_batching":{},"model_transaction_policy":{"decoupled":true},"input":[{"optional":true}],"unknown_future_field":{"nested":[1,2]}}
    );
    try testing.expectEqual(@as(u32, 8), parsed.max_batch_size);
    try testing.expectEqualSlices(i64, &.{ -1, 4 }, parsed.inputs[0].shape);
    try testing.expectEqualDeep(metadata.SchedulingInfo{ .dynamic_batching = true, .sequence_batching = true, .decoupled = true, .special_tensors = true }, parsed.scheduling);
}
