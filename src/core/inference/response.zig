const std = @import("std");
const c = @import("../bindings/c.zig").raw;
const OwnedArena = @import("../internal/owned_arena.zig").OwnedArena;
const DataType = @import("../bindings/datatype.zig").DataType;
const tensor = @import("tensor.zig");
const check = @import("request.zig").check;
pub const OutputTensor = tensor.OutputTensor;

/// Fully owned result: no pointers into the C response, request, or server.
pub const InferenceResponse = struct {
    storage: OwnedArena,
    model_name: []const u8,
    model_version: i64,
    outputs: []const OutputTensor,

    pub fn output(self: *const InferenceResponse, name: []const u8) !*const OutputTensor {
        for (self.outputs) |*item| if (std.mem.eql(u8, item.name, name)) return item;
        return error.OutputNotFound;
    }

    pub fn deinit(self: *InferenceResponse) void {
        self.storage.deinit();
        self.outputs = &.{};
        self.model_name = "";
    }
};

pub fn clone(parent: std.mem.Allocator, source: *const InferenceResponse) !InferenceResponse {
    return copyResponse(parent, source, null);
}

/// Copies a contiguous range along every output's leading batch dimension.
pub fn sliceBatch(parent: std.mem.Allocator, source: *const InferenceResponse, start: usize, count: usize, total: usize) !InferenceResponse {
    if (total == 0 or count == 0 or start > total or count > total - start)
        return error.InvalidBatchRange;
    const total_dim = std.math.cast(i64, total) orelse return error.ShapeOverflow;
    const count_dim = std.math.cast(i64, count) orelse return error.ShapeOverflow;
    return copyResponse(parent, source, .{ .start = start, .count = count, .total = total_dim, .count_dim = count_dim });
}

const BatchSlice = struct { start: usize, count: usize, total: i64, count_dim: i64 };

fn copyResponse(parent: std.mem.Allocator, source: *const InferenceResponse, batch: ?BatchSlice) !InferenceResponse {
    var storage = try OwnedArena.init(parent);
    errdefer storage.deinit();
    const alloc = storage.allocator();
    const name = try alloc.dupe(u8, source.model_name);
    const outputs = try alloc.alloc(OutputTensor, source.outputs.len);
    for (source.outputs, outputs) |input, *output| {
        if (try tensor.byteCount(input.datatype, input.shape) != input.data.len)
            return error.InvalidOutputBuffer;
        const shape = try alloc.dupe(i64, input.shape);
        var begin: usize = 0;
        var length = input.data.len;
        if (batch) |part| {
            if (shape.len == 0 or shape[0] != part.total) return error.InvalidBatchShape;
            const row_bytes = try tensor.byteCount(input.datatype, shape[1..]);
            const offset = @mulWithOverflow(part.start, row_bytes);
            const size = @mulWithOverflow(part.count, row_bytes);
            if (offset[1] != 0 or size[1] != 0) return error.ShapeOverflow;
            begin = offset[0];
            length = size[0];
            if (begin > input.data.len or length > input.data.len - begin)
                return error.InvalidOutputBuffer;
            shape[0] = part.count_dim;
        }
        const data = try alloc.alignedAlloc(u8, .@"16", length);
        @memcpy(data, input.data[begin..][0..length]);
        output.* = .{
            .name = try alloc.dupe(u8, input.name),
            .datatype = input.datatype,
            .shape = shape,
            .data = data,
        };
    }
    return .{ .storage = storage, .model_name = name, .model_version = source.model_version, .outputs = outputs };
}

pub fn decode(parent: std.mem.Allocator, raw: *c.TRITONSERVER_InferenceResponse, on_error: ?*const fn ([]const u8) void) !InferenceResponse {
    // The pinned core/src/tritonserver.cc accessor creates a NEW error via
    // RETURN_IF_STATUS_ERROR (also consumed by core's query_test.cc). Its header
    // incorrectly calls it borrowed. Consume once, matching this runtime.
    try check(c.TRITONSERVER_InferenceResponseError(raw), on_error);
    var storage = try OwnedArena.init(parent);
    errdefer storage.deinit();
    const alloc = storage.allocator();
    var name: [*c]const u8 = null;
    var version: i64 = 0;
    try check(c.TRITONSERVER_InferenceResponseModel(raw, &name, &version), on_error);
    if (name == null) return error.InvalidInferenceResponse;
    const model_name = try alloc.dupe(u8, std.mem.span(name));
    var count: u32 = 0;
    try check(c.TRITONSERVER_InferenceResponseOutputCount(raw, &count), on_error);
    const outputs = try alloc.alloc(OutputTensor, count);
    for (outputs, 0..) |*output, i| {
        var dtype: c.TRITONSERVER_DataType = 0;
        var shape: [*c]const i64 = null;
        var dim_count: u64 = 0;
        var base: ?*const anyopaque = null;
        var byte_size: usize = 0;
        var memory: c.TRITONSERVER_MemoryType = 0;
        var memory_id: i64 = 0;
        var userp: ?*anyopaque = null;
        try check(c.TRITONSERVER_InferenceResponseOutput(raw, @intCast(i), &name, &dtype, &shape, &dim_count, &base, &byte_size, &memory, &memory_id, &userp), on_error);
        if (name == null or (dim_count != 0 and shape == null) or (byte_size != 0 and base == null)) return error.InvalidInferenceResponse;
        if (memory != c.TRITONSERVER_MEMORY_CPU) return error.UnsupportedOutputMemory;
        const datatype = std.enums.fromInt(DataType, dtype) orelse return error.UnsupportedDataType;
        if (datatype == .invalid or datatype == .bytes) return error.UnsupportedDataType;
        const dims: []const i64 = if (dim_count == 0) &.{} else shape[0 .. std.math.cast(usize, dim_count) orelse return error.ShapeOverflow];
        const elements = try tensor.elementCount(dims);
        const expected = @mulWithOverflow(elements, @as(usize, c.TRITONSERVER_DataTypeByteSize(dtype)));
        if (expected[1] != 0 or expected[0] != byte_size) return error.InvalidOutputBuffer;
        const data = try alloc.alignedAlloc(u8, .@"16", byte_size);
        if (byte_size != 0) @memcpy(data, @as([*]const u8, @ptrCast(base.?))[0..byte_size]);
        output.* = .{ .name = try alloc.dupe(u8, std.mem.span(name)), .datatype = datatype, .shape = try alloc.dupe(i64, dims), .data = data };
    }
    return .{ .storage = storage, .model_name = model_name, .model_version = version, .outputs = outputs };
}
