const std = @import("std");
const DataType = @import("../bindings/datatype.zig").DataType;

pub fn dataType(comptime T: type) !DataType {
    return if (T == bool) .boolean else if (T == u8) .uint8 else if (T == u16) .uint16 else if (T == u32) .uint32 else if (T == u64) .uint64 else if (T == i8) .int8 else if (T == i16) .int16 else if (T == i32) .int32 else if (T == i64) .int64 else if (T == f16) .fp16 else if (T == f32) .fp32 else if (T == f64) .fp64 else error.UnsupportedDataType;
}

pub fn byteWidth(datatype: DataType) !usize {
    return switch (datatype) {
        .boolean, .uint8, .int8 => 1,
        .uint16, .int16, .fp16, .bf16 => 2,
        .uint32, .int32, .fp32 => 4,
        .uint64, .int64, .fp64 => 8,
        .invalid, .bytes => error.UnsupportedDataType,
    };
}

pub fn byteCount(datatype: DataType, shape: []const i64) !usize {
    const width = try byteWidth(datatype);
    const product = @mulWithOverflow(try elementCount(shape), width);
    if (product[1] != 0) return error.ShapeOverflow;
    return product[0];
}

/// Borrowed tensor data. A scheduler copies it before submit returns.
pub const InputTensor = struct {
    name: []const u8,
    datatype: DataType,
    shape: []const i64,
    data: []const u8,

    pub fn fromSlice(comptime T: type, name: []const u8, shape: []const i64, data: []const T) !InputTensor {
        const input: InputTensor = .{
            .name = name,
            .datatype = try dataType(T),
            .shape = shape,
            .data = std.mem.sliceAsBytes(data),
        };
        try input.validate();
        return input;
    }

    pub fn validate(self: InputTensor) !void {
        if (self.name.len == 0 or std.mem.indexOfScalar(u8, self.name, 0) != null)
            return error.InvalidTensorName;
        if (try byteCount(self.datatype, self.shape) != self.data.len)
            return error.ElementCountMismatch;
    }
};

pub fn elementCount(shape: []const i64) !usize {
    var count: usize = 1;
    for (shape) |dim| {
        if (dim < 0) return error.InvalidShape;
        const value = std.math.cast(usize, dim) orelse return error.ShapeOverflow;
        const product = @mulWithOverflow(count, value);
        if (product[1] != 0) return error.ShapeOverflow;
        count = product[0];
    }
    return count;
}

pub const OutputTensor = struct {
    name: []const u8,
    datatype: DataType,
    shape: []const i64,
    data: []align(16) const u8,

    /// Borrowed typed view; valid until the containing response is deinitialized.
    pub fn asSlice(self: *const OutputTensor, comptime T: type) ![]const T {
        if (try dataType(T) != self.datatype) return error.DataTypeMismatch;
        if (self.data.len % @sizeOf(T) != 0) return error.InvalidOutputBuffer;
        return std.mem.bytesAsSlice(T, self.data);
    }
};
