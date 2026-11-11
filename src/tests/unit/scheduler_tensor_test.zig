const std = @import("std");
const tensor = @import("../../core/inference/tensor.zig");
const DataType = @import("../../core/bindings/datatype.zig").DataType;
const testing = std.testing;

test "input tensor views borrow validated numeric data" {
    var values = [_]i32{ 1, 2, 3, 4 };
    const input = try tensor.InputTensor.fromSlice(i32, "IN0", &.{ 2, 2 }, &values);
    try testing.expectEqual(DataType.int32, input.datatype);
    try testing.expectEqualSlices(i64, &.{ 2, 2 }, input.shape);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&values), input.data);
    values[0] = 17;
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&values), input.data);
    try input.validate();
    try testing.expectError(error.ElementCountMismatch, tensor.InputTensor.fromSlice(i32, "IN0", &.{3}, &values));
    try testing.expectError(error.InvalidShape, tensor.InputTensor.fromSlice(i32, "IN0", &.{-1}, &values));
    try testing.expectError(error.InvalidTensorName, tensor.InputTensor.fromSlice(i32, "", &.{4}, &values));
    try testing.expectError(error.InvalidTensorName, tensor.InputTensor.fromSlice(i32, "IN\x000", &.{4}, &values));
    try testing.expectError(error.UnsupportedDataType, tensor.InputTensor.fromSlice(u128, "IN0", &.{1}, &.{0}));
}

test "raw input tensor views reject variable-width and malformed buffers" {
    var input: tensor.InputTensor = .{ .name = "IN0", .datatype = .int32, .shape = &.{2}, .data = &.{ 0, 0, 0, 0 } };
    try testing.expectError(error.ElementCountMismatch, input.validate());
    input.datatype = .bytes;
    try testing.expectError(error.UnsupportedDataType, input.validate());
    input.datatype = .invalid;
    try testing.expectError(error.UnsupportedDataType, input.validate());
    const scalar = try tensor.InputTensor.fromSlice(f32, "SCALAR", &.{}, &.{3.5});
    try testing.expectEqual(@as(usize, 4), scalar.data.len);
    const empty = try tensor.InputTensor.fromSlice(u16, "EMPTY", &.{ 3, 0 }, &.{});
    try testing.expectEqual(@as(usize, 0), empty.data.len);
}

test "fixed-width tensor byte counts include overflow checks" {
    const Case = struct { datatype: DataType, width: usize };
    for ([_]Case{
        .{ .datatype = .boolean, .width = 1 },
        .{ .datatype = .uint8, .width = 1 },
        .{ .datatype = .int8, .width = 1 },
        .{ .datatype = .uint16, .width = 2 },
        .{ .datatype = .int16, .width = 2 },
        .{ .datatype = .fp16, .width = 2 },
        .{ .datatype = .bf16, .width = 2 },
        .{ .datatype = .uint32, .width = 4 },
        .{ .datatype = .int32, .width = 4 },
        .{ .datatype = .fp32, .width = 4 },
        .{ .datatype = .uint64, .width = 8 },
        .{ .datatype = .int64, .width = 8 },
        .{ .datatype = .fp64, .width = 8 },
    }) |case| {
        try testing.expectEqual(case.width, try tensor.byteWidth(case.datatype));
        try testing.expectEqual(case.width * 6, try tensor.byteCount(case.datatype, &.{ 2, 3 }));
    }
    try testing.expectError(error.ShapeOverflow, tensor.byteCount(.fp64, &.{std.math.maxInt(i64)}));
    try testing.expectError(error.ShapeOverflow, tensor.byteCount(.uint8, &.{ std.math.maxInt(i64), std.math.maxInt(i64) }));
    try testing.expectError(error.InvalidShape, tensor.byteCount(.int32, &.{ 0, -1 }));
}
