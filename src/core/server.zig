const std = @import("std");
const c = @import("bindings/c.zig").raw;
const errors = @import("bindings/error.zig");
const models = @import("model/metadata.zig");
const inference = @import("inference/operation.zig");
const Request = @import("inference/request.zig").Request;
const InferenceResponse = @import("inference/response.zig").InferenceResponse;

pub const ServerOptions = struct {
    model_repository: []const u8,
    backend_directory: []const u8 = "core/build/install/backends",
    exit_on_error: bool = true,
    log_info: bool = true,
    /// Called synchronously with a borrowed C message, before it is deleted.
    on_error: ?*const fn ([]const u8) void = null,
};
pub const ApiVersion = struct { major: u32, minor: u32 };

pub fn apiVersion() !ApiVersion {
    var version: ApiVersion = undefined;
    try errors.check(c.TRITONSERVER_ApiVersion(&version.major, &version.minor));
    return version;
}
fn check(err: ?*c.TRITONSERVER_Error, on_error: ?*const fn ([]const u8) void) errors.TritonErr!void {
    if (err) |e| if (on_error) |report| report(std.mem.span(c.TRITONSERVER_ErrorMessage(e)));
    return errors.check(err);
}
fn cString(alloc: std.mem.Allocator, str: []const u8) ![:0]u8 {
    if (str.len == 0 or std.mem.indexOfScalar(u8, str, 0) != null) return error.InvalidPath;
    return alloc.dupeSentinel(u8, str, 0);
}

/// Owns one Triton server. Move, do not copy. Calls require external
/// synchronization when lifecycle operations overlap on multiple threads.
pub const Server = struct {
    handle: ?*c.TRITONSERVER_Server,
    on_error: ?*const fn ([]const u8) void,
    stopped: bool = false,

    /// Io must remain alive through completion and support native callback
    /// threads (std.Io.Threaded). The pending owner retains a copy of the handle.
    pub fn inferAsync(self: *const Server, alloc: std.mem.Allocator, io: std.Io, request: *Request) !inference.PendingInference {
        return inference.submit(self, alloc, io, request);
    }

    pub fn inferSync(self: *const Server, alloc: std.mem.Allocator, io: std.Io, request: *Request) !InferenceResponse {
        var pending = try self.inferAsync(alloc, io, request);
        defer pending.deinit();
        return pending.wait();
    }

    pub fn init(alloc: std.mem.Allocator, options: ServerOptions) !Server {
        const version = try apiVersion();
        if (version.major != c.TRITONSERVER_API_VERSION_MAJOR or version.minor < c.TRITONSERVER_API_VERSION_MINOR)
            return error.IncompatibleApiVersion;
        const repo = try cString(alloc, options.model_repository);
        defer alloc.free(repo);
        const backend = try cString(alloc, options.backend_directory);
        defer alloc.free(backend);
        var opts: ?*c.TRITONSERVER_ServerOptions = null;
        try check(c.TRITONSERVER_ServerOptionsNew(&opts), options.on_error);
        defer check(c.TRITONSERVER_ServerOptionsDelete(opts), options.on_error) catch {};
        try check(c.TRITONSERVER_ServerOptionsSetModelRepositoryPath(opts, repo), options.on_error);
        try check(c.TRITONSERVER_ServerOptionsSetBackendDirectory(opts, backend), options.on_error);
        try check(c.TRITONSERVER_ServerOptionsSetModelControlMode(opts, c.TRITONSERVER_MODEL_CONTROL_POLL), options.on_error);
        try check(c.TRITONSERVER_ServerOptionsSetExitOnError(opts, options.exit_on_error), options.on_error);
        try check(c.TRITONSERVER_ServerOptionsSetLogInfo(opts, options.log_info), options.on_error);
        var handle: ?*c.TRITONSERVER_Server = null;
        try check(c.TRITONSERVER_ServerNew(&handle, opts), options.on_error);
        return .{ .handle = handle orelse return error.InvalidServerHandle, .on_error = options.on_error };
    }

    fn getHandle(self: *const Server) !*c.TRITONSERVER_Server {
        return self.handle orelse error.ServerClosed;
    }

    pub fn isLive(self: *const Server) !bool {
        const handle = try self.getHandle();
        var live = false;
        try check(c.TRITONSERVER_ServerIsLive(handle, &live), self.on_error);
        return live;
    }

    pub fn isReady(self: *const Server) !bool {
        const handle = try self.getHandle();
        var ready = false;
        try check(c.TRITONSERVER_ServerIsReady(handle, &ready), self.on_error);
        return ready;
    }

    pub fn stop(self: *Server) !void {
        if (self.handle == null or self.stopped) return;
        try check(c.TRITONSERVER_ServerStop(self.handle), self.on_error);
        self.stopped = true;
    }

    /// Success clears ownership. On failure the handle is retained for retry.
    pub fn deinit(self: *Server) !void {
        if (self.handle) |handle| {
            try check(c.TRITONSERVER_ServerDelete(handle), self.on_error);
            self.handle = null;
            self.stopped = true;
        }
    }

    pub fn pollRepository(self: *Server) !void {
        const handle = try self.getHandle();
        if (self.stopped) return error.ServerStopped;
        try check(c.TRITONSERVER_ServerPollModelRepository(handle), self.on_error);
    }

    fn messageBytes(message: *c.TRITONSERVER_Message) ![]const u8 {
        var base: [*c]const u8 = null;
        var size: usize = 0;
        try errors.check(c.TRITONSERVER_MessageSerializeToJson(message, &base, &size));
        if (base == null) return error.InvalidMessage;
        return base[0..size];
    }

    pub fn listModels(self: *const Server, parent: std.mem.Allocator) !models.ModelList {
        const handle = try self.getHandle();
        if (self.stopped) return error.ServerStopped;
        var index: ?*c.TRITONSERVER_Message = null;
        try check(c.TRITONSERVER_ServerModelIndex(handle, 0, &index), self.on_error);
        defer check(c.TRITONSERVER_MessageDelete(index), self.on_error) catch {};
        var result = try models.parseIndex(parent, try messageBytes(index.?));
        errdefer result.deinit();
        const alloc = result.storage.allocator();
        for (result.models) |*model| {
            if (!std.mem.eql(u8, model.state, "READY")) continue;
            const name_z = try alloc.dupeSentinel(u8, model.name, 0);
            var metadata: ?*c.TRITONSERVER_Message = null;
            try check(c.TRITONSERVER_ServerModelMetadata(handle, name_z, model.version.?, &metadata), self.on_error);
            defer check(c.TRITONSERVER_MessageDelete(metadata), self.on_error) catch {};
            var config: ?*c.TRITONSERVER_Message = null;
            try check(c.TRITONSERVER_ServerModelConfig(handle, name_z, model.version.?, 1, &config), self.on_error);
            defer check(c.TRITONSERVER_MessageDelete(config), self.on_error) catch {};
            model.metadata = try models.parseMetadata(alloc, try messageBytes(metadata.?), try messageBytes(config.?));
        }
        return result;
    }
};
