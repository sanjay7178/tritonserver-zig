const b = @import("tritonserver_zig").backend;

pub const Context = struct {
    pub fn init() !Context {
        return .{};
    }

    pub fn deinit(_: *Context) void {}

    pub fn copy(_: *Context, destination: b.Buffer, source: b.ConstBuffer) !void {
        try b.copyCpu(destination, source);
    }

    pub fn finish(_: *Context) !void {}
};
