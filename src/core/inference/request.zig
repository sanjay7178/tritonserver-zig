const std = @import("std");
const c = @import("../bindings/c.zig").raw;
const errors = @import("../bindings/error.zig");
const Server = @import("../server.zig").Server;
const OwnedArena = @import("../internal/owned_arena.zig").OwnedArena;
const tensor = @import("tensor.zig");
const DataType = @import("../bindings/datatype.zig").DataType;

/// Owns the C request and copies of its inputs. Move only while idle; do not
/// mutate/copy/deinitialize it until the pending operation has been waited.
pub const Request = struct {
    handle: ?*c.TRITONSERVER_InferenceRequest,
    server_handle: *c.TRITONSERVER_Server,
    storage: OwnedArena,
    inputs: std.ArrayList([]const u8) = .empty,
    outputs: std.ArrayList([]const u8) = .empty,
    in_flight: bool = false,

    pub fn init(parent: std.mem.Allocator, server: *const Server, model_name: []const u8, model_version: i64) !Request {
        const server_handle = server.handle orelse return error.ServerClosed;
        if (server.stopped) return error.ServerStopped;
        if (model_version != -1 and model_version <= 0) return error.InvalidModelVersion;
        if (model_name.len == 0 or std.mem.indexOfScalar(u8, model_name, 0) != null) return error.InvalidModelName;
        var storage = try OwnedArena.init(parent);
        errdefer storage.deinit();
        const name = try storage.allocator().dupeSentinel(u8, model_name, 0);
        var flags: u32 = 0;
        try check(c.TRITONSERVER_ServerModelTransactionProperties(server_handle, name, model_version, &flags, null), server.on_error);
        if ((flags & c.TRITONSERVER_TXN_DECOUPLED) != 0) return error.DecoupledModelUnsupported;
        var handle: ?*c.TRITONSERVER_InferenceRequest = null;
        try check(c.TRITONSERVER_InferenceRequestNew(&handle, server_handle, name, model_version), server.on_error);
        return .{ .handle = handle orelse return error.InvalidRequestHandle, .server_handle = server_handle, .storage = storage };
    }

    pub fn ensureIdle(self: *const Request) !*c.TRITONSERVER_InferenceRequest {
        const handle = self.handle orelse return error.RequestClosed;
        if (self.in_flight) return error.RequestInFlight;
        return handle;
    }

    fn nameZ(self: *Request, name: []const u8) ![:0]u8 {
        if (name.len == 0 or std.mem.indexOfScalar(u8, name, 0) != null) return error.InvalidTensorName;
        return self.storage.allocator().dupeSentinel(u8, name, 0);
    }

    pub fn addInput(self: *Request, comptime T: type, name: []const u8, shape: []const i64, data: []const T) !void {
        _ = try self.ensureIdle();
        const datatype = try tensor.dataType(T);
        if (try tensor.elementCount(shape) != data.len) return error.ElementCountMismatch;
        return self.addInputBytes(datatype, name, shape, std.mem.sliceAsBytes(data));
    }

    pub fn addInputBytes(self: *Request, datatype: DataType, name: []const u8, shape: []const i64, bytes: []const u8) !void {
        try addInputWith(self, datatype, name, shape, bytes.len, bytes, copyInput);
    }

    pub fn requestOutput(self: *Request, name: []const u8) !void {
        const handle = try self.ensureIdle();
        for (self.outputs.items) |existing| if (std.mem.eql(u8, existing, name)) return error.DuplicateOutput;
        const name_z = try self.nameZ(name);
        try self.outputs.ensureUnusedCapacity(self.storage.allocator(), 1);
        try errors.check(c.TRITONSERVER_InferenceRequestAddRequestedOutput(handle, name_z));
        self.outputs.appendAssumeCapacity(name_z);
    }

    pub fn deinit(self: *Request) !void {
        if (self.in_flight) return error.RequestInFlight;
        if (self.handle) |handle| {
            try errors.check(c.TRITONSERVER_InferenceRequestDelete(handle));
            self.handle = null;
        }
        self.storage.deinit();
        self.inputs = .empty;
        self.outputs = .empty;
    }
};

fn copyInput(source: []const u8, destination: []u8) void {
    @memcpy(destination, source);
}

/// Internal writer for inputs assembled by the scheduler. The writer fills the
/// whole buffer synchronously before C sees it and must not retain the slice.
pub fn addInputWith(
    request: *Request,
    datatype: DataType,
    name: []const u8,
    shape: []const i64,
    byte_count: usize,
    context: anytype,
    comptime fill: fn (@TypeOf(context), []u8) void,
) !void {
    const handle = try request.ensureIdle();
    if (try tensor.byteCount(datatype, shape) != byte_count) return error.ElementCountMismatch;
    for (request.inputs.items) |existing| if (std.mem.eql(u8, existing, name)) return error.DuplicateInput;
    const alloc = request.storage.allocator();
    const name_z = try request.nameZ(name);
    const dims = try alloc.dupe(i64, shape);
    try request.inputs.ensureUnusedCapacity(alloc, 1);
    const data = try alloc.alignedAlloc(u8, .@"16", byte_count);
    fill(context, data);
    try errors.check(c.TRITONSERVER_InferenceRequestAddInput(handle, name_z, @backingInt(datatype), dims.ptr, @intCast(dims.len)));
    errdefer errors.check(c.TRITONSERVER_InferenceRequestRemoveInput(handle, name_z)) catch {};
    // Triton retains this pointer. Request keeps the allocation until native
    // ownership has returned and the C request has been deleted.
    try errors.check(c.TRITONSERVER_InferenceRequestAppendInputData(handle, name_z, if (data.len == 0) null else data.ptr, data.len, c.TRITONSERVER_MEMORY_CPU, 0));
    request.inputs.appendAssumeCapacity(name_z);
}

pub fn check(err: ?*c.TRITONSERVER_Error, on_error: ?*const fn ([]const u8) void) errors.TritonErr!void {
    if (err) |e| if (on_error) |report| report(std.mem.span(c.TRITONSERVER_ErrorMessage(e)));
    return errors.check(err);
}
