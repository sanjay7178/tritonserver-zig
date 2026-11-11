const std = @import("std");
const tensor = @import("../../core/inference/tensor.zig");

test "fixed-width tensor type mapping and concrete shape sizes" {
    try std.testing.expectEqual(.boolean, try tensor.dataType(bool));
    try std.testing.expectEqual(.uint64, try tensor.dataType(u64));
    try std.testing.expectEqual(.int8, try tensor.dataType(i8));
    try std.testing.expectEqual(.fp16, try tensor.dataType(f16));
    try std.testing.expectEqual(.fp32, try tensor.dataType(f32));
    try std.testing.expectEqual(.fp64, try tensor.dataType(f64));
    try std.testing.expectError(error.UnsupportedDataType, tensor.dataType(u128));
    try std.testing.expectEqual(@as(usize, 1), try tensor.elementCount(&.{}));
    try std.testing.expectEqual(@as(usize, 0), try tensor.elementCount(&.{0}));
    try std.testing.expectEqual(@as(usize, 32), try tensor.elementCount(&.{ 8, 4 }));
    try std.testing.expectError(error.InvalidShape, tensor.elementCount(&.{ -1, 4 }));
    try std.testing.expectError(error.ShapeOverflow, tensor.elementCount(&.{ std.math.maxInt(i64), std.math.maxInt(i64) }));
}
