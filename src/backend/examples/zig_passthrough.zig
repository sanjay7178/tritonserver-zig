//! Fixed-width IN0 -> OUT0 backend, built with a CPU or optional CUDA copy engine.
const std = @import("std");
const t = @import("tritonserver_zig");
const b = t.backend;
const c = b.c;
const copy_engine = @import("copy_engine");
const allocator = std.heap.c_allocator;

const WireTensor = struct {
    name: []const u8,
    data_type: []const u8,
    dims: []const i64,
    reshape: ?std.json.Value = null,
    optional: bool = false,
    is_shape_tensor: bool = false,
    allow_ragged_batch: bool = false,

    fn special(self: WireTensor) bool {
        return self.reshape != null or self.optional or self.is_shape_tensor or self.allow_ragged_batch;
    }
};

const WireConfig = struct {
    max_batch_size: u32 = 0,
    input: []const WireTensor = &.{},
    output: []const WireTensor = &.{},
    sequence_batching: ?std.json.Value = null,
    ensemble_scheduling: ?std.json.Value = null,
    model_transaction_policy: ?struct { decoupled: bool = false } = null,
    batch_input: []const std.json.Value = &.{},
    batch_output: []const std.json.Value = &.{},
};

const ModelState = struct {
    datatype: c.TRITONSERVER_DataType,
    dimensions: []i64,
    max_batch_size: u32,

    fn destroy(self: *ModelState) void {
        allocator.free(self.dimensions);
        allocator.destroy(self);
    }

    fn validate(self: *const ModelState, properties: b.InputProperties) !void {
        if (properties.datatype != self.datatype) return error.DataTypeMismatch;
        const batched = self.max_batch_size != 0;
        if (properties.shape.len != self.dimensions.len + @intFromBool(batched)) return error.InvalidShape;
        if (batched and (properties.shape[0] <= 0 or properties.shape[0] > self.max_batch_size))
            return error.InvalidShape;
        const shape = properties.shape[@intFromBool(batched)..];
        for (shape, self.dimensions) |actual, expected| {
            if (actual < 0 or (expected != -1 and actual != expected)) return error.InvalidShape;
        }
        if (try b.byteCount(properties.datatype, properties.shape) != properties.byte_size)
            return error.ElementCountMismatch;
    }
};

const InstanceState = struct {
    model: *const ModelState,
    copier: copy_engine.Context,

    fn destroy(self: *InstanceState) void {
        self.copier.deinit();
        allocator.destroy(self);
    }
};

fn loadConfiguration(model: *c.TRITONBACKEND_Model) !*ModelState {
    var message: ?*c.TRITONSERVER_Message = null;
    try b.check(c.TRITONBACKEND_ModelConfig(model, 1, &message));
    const owned = message orelse return error.InvalidModelConfiguration;
    defer b.check(c.TRITONSERVER_MessageDelete(owned)) catch |err|
        std.log.err("model config cleanup: {s}", .{@errorName(err)});
    var bytes: [*c]const u8 = null;
    var length: usize = 0;
    try b.check(c.TRITONSERVER_MessageSerializeToJson(owned, &bytes, &length));
    if (bytes == null) return error.InvalidModelConfiguration;
    const parsed = try std.json.parseFromSlice(WireConfig, allocator, bytes[0..length], .{
        .allocate = .alloc_always,
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();
    const config = parsed.value;
    if (config.input.len != 1 or config.output.len != 1) return error.InvalidModelConfiguration;
    if (config.sequence_batching != null or config.ensemble_scheduling != null or
        config.batch_input.len != 0 or config.batch_output.len != 0 or
        (if (config.model_transaction_policy) |policy| policy.decoupled else false))
        return error.UnsupportedModelConfiguration;
    const input = config.input[0];
    const output = config.output[0];
    if (input.special() or output.special()) return error.UnsupportedModelConfiguration;
    if (!std.mem.eql(u8, input.name, "IN0") or !std.mem.eql(u8, output.name, "OUT0"))
        return error.InvalidModelConfiguration;
    if (!std.mem.eql(u8, input.data_type, output.data_type)) return error.DataTypeMismatch;
    if (!std.mem.eql(i64, input.dims, output.dims)) return error.InvalidShape;
    if (input.dims.len == 0) return error.InvalidShape;
    for (input.dims) |dimension| if (dimension == 0 or dimension < -1) return error.InvalidShape;
    if (!std.mem.startsWith(u8, input.data_type, "TYPE_")) return error.UnsupportedDataType;
    const name = try allocator.dupeSentinel(u8, input.data_type[5..], 0);
    defer allocator.free(name);
    const datatype = c.TRITONSERVER_StringToDataType(name);
    if (datatype == c.TRITONSERVER_TYPE_INVALID or datatype == c.TRITONSERVER_TYPE_BYTES or
        c.TRITONSERVER_DataTypeByteSize(datatype) == 0)
        return error.UnsupportedDataType;
    const dimensions = try allocator.dupe(i64, input.dims);
    errdefer allocator.free(dimensions);
    const state = try allocator.create(ModelState);
    state.* = .{ .datatype = datatype, .dimensions = dimensions, .max_batch_size = config.max_batch_size };
    return state;
}

fn modelState(model: *c.TRITONBACKEND_Model) !*ModelState {
    var pointer: ?*anyopaque = null;
    try b.check(c.TRITONBACKEND_ModelState(model, &pointer));
    return @ptrCast(@alignCast(pointer orelse return error.InvalidModelState));
}

fn instanceState(instance: *c.TRITONBACKEND_ModelInstance) !*InstanceState {
    var pointer: ?*anyopaque = null;
    try b.check(c.TRITONBACKEND_ModelInstanceState(instance, &pointer));
    return @ptrCast(@alignCast(pointer orelse return error.InvalidInstanceState));
}

fn executeRequest(state: *InstanceState, request: b.RequestView, response: *b.Response) !void {
    if (try request.inputCount() != 1) return error.InvalidInputCount;
    const input = try request.input("IN0");
    const properties = try input.properties();
    try state.model.validate(properties);
    const outputs = try request.requestedOutputCount();
    if (outputs > 1) return error.UnknownOutput;
    if (outputs == 1 and !std.mem.eql(u8, try request.requestedOutputName(0), "OUT0"))
        return error.UnknownOutput;

    // Inspect every segment before allocating or queuing copies. Descriptors
    // carry device tags; obtaining them never dereferences a device address.
    var total: usize = 0;
    var preferred_memory: c.TRITONSERVER_MemoryType = c.TRITONSERVER_MEMORY_CPU;
    var preferred_id: i64 = 0;
    var first_memory: c.TRITONSERVER_MemoryType = c.TRITONSERVER_MEMORY_CPU;
    var first_id: i64 = 0;
    for (0..properties.buffer_count) |index| {
        const segment = try input.buffer(@intCast(index));
        try segment.validate();
        total = std.math.add(usize, total, segment.byte_size) catch return error.ShapeOverflow;
        if (index == 0) {
            first_memory = segment.memory_type;
            first_id = segment.memory_type_id;
        }
        if (segment.memory_type == c.TRITONSERVER_MEMORY_GPU) {
            preferred_memory = segment.memory_type;
            preferred_id = segment.memory_type_id;
        }
    }
    if (total != properties.byte_size) return error.ElementCountMismatch;
    try response.setIntParameter("input_memory_type", first_memory);
    try response.setIntParameter("input_memory_type_id", first_id);
    try response.setIntParameter("input_buffer_count", properties.buffer_count);
    const destination = try response.output("OUT0", properties.datatype, properties.shape, properties.byte_size, preferred_memory, preferred_id);
    var offset: usize = 0;
    for (0..properties.buffer_count) |index| {
        const segment = try input.buffer(@intCast(index));
        try state.copier.copy(try destination.slice(offset, segment.byte_size), segment);
        offset += segment.byte_size;
    }
}

const Impl = struct {
    pub fn initialize(raw: ?*c.TRITONBACKEND_Backend) callconv(.c) ?*c.TRITONSERVER_Error {
        _ = raw orelse return b.errorToC(error.InvalidArg);
        var major: u32 = 0;
        var minor: u32 = 0;
        if (c.TRITONBACKEND_ApiVersion(&major, &minor)) |err| return err;
        if (major != c.TRITONBACKEND_API_VERSION_MAJOR or minor < c.TRITONBACKEND_API_VERSION_MINOR)
            return b.errorToC(error.IncompatibleApiVersion);
        return null;
    }

    pub fn modelInitialize(raw: ?*c.TRITONBACKEND_Model) callconv(.c) ?*c.TRITONSERVER_Error {
        const model = raw orelse return b.errorToC(error.InvalidArg);
        const state = loadConfiguration(model) catch |err| return b.errorToC(err);
        if (c.TRITONBACKEND_ModelSetState(model, state)) |err| {
            state.destroy();
            return err;
        }
        return null;
    }

    pub fn modelFinalize(raw: ?*c.TRITONBACKEND_Model) callconv(.c) ?*c.TRITONSERVER_Error {
        const model = raw orelse return b.errorToC(error.InvalidArg);
        // Triton also destroys models whose initialization rejected the config.
        var pointer: ?*anyopaque = null;
        if (c.TRITONBACKEND_ModelState(model, &pointer)) |err| return err;
        const state: *ModelState = @ptrCast(@alignCast(pointer orelse return null));
        if (c.TRITONBACKEND_ModelSetState(model, null)) |err| return err;
        state.destroy();
        return null;
    }

    pub fn modelInstanceInitialize(raw: ?*c.TRITONBACKEND_ModelInstance) callconv(.c) ?*c.TRITONSERVER_Error {
        const instance = raw orelse return b.errorToC(error.InvalidArg);
        var kind: c.TRITONSERVER_InstanceGroupKind = c.TRITONSERVER_INSTANCEGROUPKIND_CPU;
        if (c.TRITONBACKEND_ModelInstanceKind(instance, &kind)) |err| return err;
        // CUDA is an optional memory-copy engine for this CPU-kind backend;
        // native GPU instance pools and their device selection are separate.
        if (kind != c.TRITONSERVER_INSTANCEGROUPKIND_CPU)
            return b.errorToC(error.UnsupportedModelConfiguration);
        var model: ?*c.TRITONBACKEND_Model = null;
        if (c.TRITONBACKEND_ModelInstanceModel(instance, &model)) |err| return err;
        const parent = modelState(model orelse return b.errorToC(error.InvalidModelState)) catch |err| return b.errorToC(err);
        const state = allocator.create(InstanceState) catch |err| return b.errorToC(err);
        state.* = .{
            .model = parent,
            .copier = copy_engine.Context.init() catch |err| {
                allocator.destroy(state);
                return b.errorToC(err);
            },
        };
        if (c.TRITONBACKEND_ModelInstanceSetState(instance, state)) |err| {
            state.destroy();
            return err;
        }
        return null;
    }

    pub fn modelInstanceFinalize(raw: ?*c.TRITONBACKEND_ModelInstance) callconv(.c) ?*c.TRITONSERVER_Error {
        const instance = raw orelse return b.errorToC(error.InvalidArg);
        var pointer: ?*anyopaque = null;
        if (c.TRITONBACKEND_ModelInstanceState(instance, &pointer)) |err| return err;
        const state: *InstanceState = @ptrCast(@alignCast(pointer orelse return null));
        if (c.TRITONBACKEND_ModelInstanceSetState(instance, null)) |err| return err;
        state.destroy();
        return null;
    }

    pub fn execute(instance: *c.TRITONBACKEND_ModelInstance, request: b.RequestView, response: *b.Response) ?*c.TRITONSERVER_Error {
        const state = instanceState(instance) catch |err| return b.errorToC(err);
        const failure: ?anyerror = blk: {
            executeRequest(state, request, response) catch |err| break :blk err;
            break :blk null;
        };
        // CUDA inputs/outputs cannot be released until pending copies finish,
        // including a request whose later segment or output operation failed.
        state.copier.finish() catch |err| return b.errorToC(failure orelse err);
        if (failure) |err| return b.errorToC(err);
        return null;
    }
};

comptime {
    b.exportBackend(Impl);
}
