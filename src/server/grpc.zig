const std = @import("std");
const t = @import("tritonserver_zig");
const GrpcListen = @import("command_line.zig").GrpcListen;

pub const enabled = @import("server_build_options").grpc;

const NativeFrontend = opaque {};
extern fn tritonzig_grpc_start(server: *t.c.TRITONSERVER_Server, address: [*:0]const u8, port: u16, frontend: *?*NativeFrontend) ?*t.c.TRITONSERVER_Error;
extern fn tritonzig_grpc_prepare_shutdown(frontend: *NativeFrontend, remaining_seconds: *u32) ?*t.c.TRITONSERVER_Error;
extern fn tritonzig_grpc_stop(frontend: *?*NativeFrontend) ?*t.c.TRITONSERVER_Error;

fn check(operation: []const u8, err: ?*t.c.TRITONSERVER_Error) t.TritonErr!void {
    if (err) |native_error|
        std.log.err("{s}: {s}", .{ operation, std.mem.span(t.c.TRITONSERVER_ErrorMessage(native_error)) });
    try t.check(err);
}

pub const Frontend = struct {
    handle: ?*NativeFrontend = null,

    pub fn start(self: *Frontend, alloc: std.mem.Allocator, server: *const t.Server, listen: GrpcListen) !void {
        if (!enabled) return error.GrpcUnavailable;
        const server_handle = server.handle orelse return error.ServerClosed;
        const address = try alloc.dupeSentinel(u8, listen.address, 0);
        defer alloc.free(address);
        try check("Could not start gRPC", tritonzig_grpc_start(server_handle, address, listen.port, &self.handle));
    }

    pub fn shutdown(self: *Frontend, server: *t.Server) void {
        if (!enabled) return;
        const frontend = self.handle orelse return;
        var remaining_seconds: u32 = 30;
        check("Could not prepare gRPC shutdown", tritonzig_grpc_prepare_shutdown(frontend, &remaining_seconds)) catch {};
        check("Could not update Triton exit timeout", t.c.TRITONSERVER_ServerSetExitTimeout(server.handle, remaining_seconds)) catch {};

        // Match the upstream standalone server shutdown order. The gRPC
        // completion queues must stay alive while Triton drains callbacks.
        server.stop() catch |err| {
            std.log.err("Triton could not stop with gRPC active ({s})", .{@errorName(err)});
            // Avoid releasing state still reachable by native threads.
            std.c._Exit(1);
        };
        check("Could not stop gRPC", tritonzig_grpc_stop(&self.handle)) catch std.c._Exit(1);
    }
};
