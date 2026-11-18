const std = @import("std");
const c = @import("../core/bindings/c.zig").raw;
const core_tensor = @import("../core/inference/tensor.zig");
const DataType = @import("../core/bindings/datatype.zig").DataType;

/// A tagged borrowed address. GPU addresses must never be read as host slices.
pub const ConstBuffer = struct {
    ptr: ?*const anyopaque,
    byte_size: usize,
    memory_type: c.TRITONSERVER_MemoryType = c.TRITONSERVER_MEMORY_CPU,
    memory_type_id: i64 = 0,

    pub fn validate(self: ConstBuffer) !void {
        try validateAddress(self.ptr, self.byte_size, self.memory_type, self.memory_type_id);
    }

    pub fn slice(self: ConstBuffer, offset: usize, length: usize) !ConstBuffer {
        try self.validate();
        if (offset > self.byte_size or length > self.byte_size - offset) return error.InvalidBufferRange;
        var result = self;
        result.ptr = if (self.ptr) |ptr| @ptrFromInt(std.math.add(usize, @intFromPtr(ptr), offset) catch return error.BufferAddressOverflow) else null;
        result.byte_size = length;
        return result;
    }
};

pub const Buffer = struct {
    ptr: ?*anyopaque,
    byte_size: usize,
    memory_type: c.TRITONSERVER_MemoryType = c.TRITONSERVER_MEMORY_CPU,
    memory_type_id: i64 = 0,

    pub fn asConst(self: Buffer) ConstBuffer {
        return .{ .ptr = self.ptr, .byte_size = self.byte_size, .memory_type = self.memory_type, .memory_type_id = self.memory_type_id };
    }

    pub fn validate(self: Buffer) !void {
        try self.asConst().validate();
    }

    pub fn slice(self: Buffer, offset: usize, length: usize) !Buffer {
        const part = try self.asConst().slice(offset, length);
        return .{ .ptr = if (part.ptr) |ptr| @constCast(ptr) else null, .byte_size = part.byte_size, .memory_type = part.memory_type, .memory_type_id = part.memory_type_id };
    }
};

fn validateAddress(ptr: ?*const anyopaque, size: usize, memory: c.TRITONSERVER_MemoryType, id: i64) !void {
    if (size != 0 and ptr == null) return error.InvalidBuffer;
    if (ptr) |address| _ = std.math.add(usize, @intFromPtr(address), size) catch return error.BufferAddressOverflow;
    switch (memory) {
        c.TRITONSERVER_MEMORY_CPU, c.TRITONSERVER_MEMORY_CPU_PINNED => {},
        c.TRITONSERVER_MEMORY_GPU => if (id < 0) return error.InvalidMemoryTypeId,
        else => return error.UnsupportedMemoryType,
    }
}

pub fn byteCount(datatype: c.TRITONSERVER_DataType, shape: []const i64) !usize {
    const kind = std.enums.fromInt(DataType, datatype) orelse return error.UnsupportedDataType;
    return core_tensor.byteCount(kind, shape);
}

/// Copy only host-accessible memory. CUDA support belongs to a separate engine.
pub fn copyCpu(destination: Buffer, source: ConstBuffer) !void {
    try destination.validate();
    try source.validate();
    if (destination.byte_size != source.byte_size) return error.BufferSizeMismatch;
    for ([_]c.TRITONSERVER_MemoryType{ destination.memory_type, source.memory_type }) |memory|
        if (memory != c.TRITONSERVER_MEMORY_CPU and memory != c.TRITONSERVER_MEMORY_CPU_PINNED)
            return error.UnsupportedMemoryType;
    if (source.byte_size == 0) return;
    const dst = @as([*]u8, @ptrCast(destination.ptr.?))[0..destination.byte_size];
    const src = @as([*]const u8, @ptrCast(source.ptr.?))[0..source.byte_size];
    @memmove(dst, src);
}

test "tagged buffers slice addresses without reading device pointers" {
    const testing = std.testing;
    const source: ConstBuffer = .{ .ptr = @ptrFromInt(0x1000), .byte_size = 16, .memory_type = c.TRITONSERVER_MEMORY_GPU, .memory_type_id = 2 };
    const part = try source.slice(3, 5);
    try testing.expectEqual(@as(usize, 0x1003), @intFromPtr(part.ptr.?));
    try testing.expectEqual(@as(usize, 5), part.byte_size);
    try testing.expectEqual(@as(i64, 2), part.memory_type_id);
    try testing.expectError(error.InvalidBufferRange, source.slice(15, 2));
    try testing.expectError(error.InvalidBuffer, (ConstBuffer{ .ptr = null, .byte_size = 1 }).validate());
    try testing.expectError(error.BufferAddressOverflow, (ConstBuffer{ .ptr = @ptrFromInt(std.math.maxInt(usize) - 1), .byte_size = 2 }).validate());
    try testing.expectError(error.InvalidMemoryTypeId, (ConstBuffer{ .ptr = null, .byte_size = 0, .memory_type = c.TRITONSERVER_MEMORY_GPU, .memory_type_id = -1 }).validate());
    const empty = try (Buffer{ .ptr = null, .byte_size = 0 }).slice(0, 0);
    try testing.expect(empty.ptr == null);
}

test "CPU copies handle pinned buffers, overlapping ranges and empty tensors" {
    const testing = std.testing;
    var values = [_]u8{ 1, 2, 3, 4, 5 };
    const buffer: Buffer = .{ .ptr = &values, .byte_size = values.len, .memory_type = c.TRITONSERVER_MEMORY_CPU_PINNED };
    try copyCpu(try buffer.slice(1, 4), try buffer.asConst().slice(0, 4));
    try testing.expectEqualSlices(u8, &.{ 1, 1, 2, 3, 4 }, &values);
    try copyCpu(.{ .ptr = null, .byte_size = 0 }, .{ .ptr = null, .byte_size = 0 });
    try testing.expectError(error.UnsupportedMemoryType, copyCpu(.{ .ptr = &values, .byte_size = 1 }, .{ .ptr = @ptrFromInt(0x1000), .byte_size = 1, .memory_type = c.TRITONSERVER_MEMORY_GPU }));
    try testing.expectError(error.BufferSizeMismatch, copyCpu(.{ .ptr = &values, .byte_size = 1 }, .{ .ptr = &values, .byte_size = 2 }));
}

test "backend byte counts include scalar, zero dimensions and fixed-width BF16" {
    const testing = std.testing;
    try testing.expectEqual(@as(usize, 4), try byteCount(c.TRITONSERVER_TYPE_FP32, &.{}));
    try testing.expectEqual(@as(usize, 0), try byteCount(c.TRITONSERVER_TYPE_INT32, &.{ 2, 0, 9 }));
    try testing.expectEqual(@as(usize, 12), try byteCount(c.TRITONSERVER_TYPE_BF16, &.{ 2, 3 }));
    try testing.expectError(error.InvalidShape, byteCount(c.TRITONSERVER_TYPE_INT32, &.{-1}));
    try testing.expectError(error.UnsupportedDataType, byteCount(c.TRITONSERVER_TYPE_BYTES, &.{1}));
}
