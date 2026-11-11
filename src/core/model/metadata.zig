const std = @import("std");
const c = @import("../bindings/c.zig").raw;
const DataType = @import("../bindings/datatype.zig").DataType;
const OwnedArena = @import("../internal/owned_arena.zig").OwnedArena;

pub const TensorInfo = struct {
    name: []const u8,
    datatype: DataType,
    /// Protocol shape, including the implicit leading -1 for batched models.
    shape: []const i64,
};
pub const SchedulingInfo = struct {
    dynamic_batching: bool = false,
    sequence_batching: bool = false,
    decoupled: bool = false,
    special_tensors: bool = false,
};
pub const ModelMetadata = struct {
    platform: []const u8,
    max_batch_size: u32,
    inputs: []const TensorInfo,
    outputs: []const TensorInfo,
    scheduling: SchedulingInfo = .{},
};
pub const ModelInfo = struct {
    name: []const u8,
    version: ?i64,
    state: []const u8,
    reason: ?[]const u8,
    metadata: ?ModelMetadata = null,
};

/// Owns all strings and arrays, including parsed C messages. May outlive Server.
/// Move this value; do not duplicate ownership.
pub const ModelList = struct {
    storage: OwnedArena,
    models: []ModelInfo,

    pub fn deinit(self: *ModelList) void {
        self.storage.deinit();
        self.models = &.{};
    }
};

const IndexEntry = struct {
    name: []const u8,
    version: ?[]const u8 = null,
    state: []const u8 = "UNKNOWN",
    reason: ?[]const u8 = null,
};
const WireTensor = struct { name: []const u8, datatype: []const u8, shape: []const i64 };
const WireMetadata = struct { platform: []const u8 = "", inputs: []const WireTensor, outputs: []const WireTensor };
const WireConfigTensor = struct {
    is_shape_tensor: bool = false,
    allow_ragged_batch: bool = false,
    optional: bool = false,

    fn special(self: WireConfigTensor) bool {
        return self.is_shape_tensor or self.allow_ragged_batch or self.optional;
    }
};
const WireTransactionPolicy = struct { decoupled: bool = false };
const WireConfig = struct {
    max_batch_size: i64 = 0,
    dynamic_batching: ?std.json.Value = null,
    sequence_batching: ?std.json.Value = null,
    model_transaction_policy: ?WireTransactionPolicy = null,
    input: []const WireConfigTensor = &.{},
    output: []const WireConfigTensor = &.{},
    batch_input: []const std.json.Value = &.{},
    batch_output: []const std.json.Value = &.{},

    fn scheduling(self: WireConfig) SchedulingInfo {
        var info: SchedulingInfo = .{
            .dynamic_batching = self.dynamic_batching != null,
            .sequence_batching = self.sequence_batching != null,
            .decoupled = if (self.model_transaction_policy) |policy| policy.decoupled else false,
            .special_tensors = self.batch_input.len != 0 or self.batch_output.len != 0,
        };
        for (self.input) |input| info.special_tensors = info.special_tensors or input.special();
        for (self.output) |output| info.special_tensors = info.special_tensors or output.special();
        return info;
    }
};
const parse_options: std.json.ParseOptions = .{ .allocate = .alloc_always, .ignore_unknown_fields = true };

pub fn parseIndex(parent: std.mem.Allocator, bytes: []const u8) !ModelList {
    var storage = try OwnedArena.init(parent);
    errdefer storage.deinit();
    const alloc = storage.allocator();
    const entries = try std.json.parseFromSliceLeaky([]const IndexEntry, alloc, bytes, parse_options);
    const models = try alloc.alloc(ModelInfo, entries.len);
    for (entries, models) |entry, *model| {
        const version: ?i64 = if (entry.version) |v| blk: {
            const n = std.fmt.parseInt(i64, v, 10) catch return error.InvalidModelVersion;
            if (n <= 0) return error.InvalidModelVersion;
            break :blk n;
        } else null;
        if (std.mem.eql(u8, entry.state, "READY") and version == null) return error.InvalidModelVersion;
        if (entry.name.len == 0 or std.mem.indexOfScalar(u8, entry.name, 0) != null) return error.InvalidModelName;
        model.* = .{ .name = entry.name, .version = version, .state = entry.state, .reason = entry.reason };
    }
    std.mem.sort(ModelInfo, models, {}, struct {
        fn less(_: void, a: ModelInfo, b: ModelInfo) bool {
            const order = std.mem.order(u8, a.name, b.name);
            return if (order == .eq) (a.version orelse -1) < (b.version orelse -1) else order == .lt;
        }
    }.less);
    return .{ .storage = storage, .models = models };
}

fn convertTensors(alloc: std.mem.Allocator, wires: []const WireTensor) ![]const TensorInfo {
    const tensors = try alloc.alloc(TensorInfo, wires.len);
    for (wires, tensors) |wire, *tensor| {
        const dtype_z = try alloc.dupeSentinel(u8, wire.datatype, 0);
        if (std.mem.indexOfScalar(u8, wire.datatype, 0) != null) return error.UnsupportedDataType;
        const raw = c.TRITONSERVER_StringToDataType(dtype_z);
        if (raw == c.TRITONSERVER_TYPE_INVALID) return error.UnsupportedDataType;
        for (wire.shape) |dim| if (dim < -1) return error.InvalidTensorShape;
        tensor.* = .{ .name = wire.name, .datatype = std.enums.fromInt(DataType, raw) orelse return error.UnsupportedDataType, .shape = wire.shape };
    }
    return tensors;
}

/// `alloc` must own the returned strings/shapes (normally ModelList's arena).
pub fn parseMetadata(alloc: std.mem.Allocator, metadata_json: []const u8, config_json: []const u8) !ModelMetadata {
    const wire = try std.json.parseFromSliceLeaky(WireMetadata, alloc, metadata_json, parse_options);
    const config = try std.json.parseFromSliceLeaky(WireConfig, alloc, config_json, parse_options);
    if (config.max_batch_size < 0 or config.max_batch_size > std.math.maxInt(u32)) return error.InvalidBatchLimit;
    return .{
        .platform = wire.platform,
        .max_batch_size = @intCast(config.max_batch_size),
        .inputs = try convertTensors(alloc, wire.inputs),
        .outputs = try convertTensors(alloc, wire.outputs),
        .scheduling = config.scheduling(),
    };
}
