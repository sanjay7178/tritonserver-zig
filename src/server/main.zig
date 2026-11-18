const std = @import("std");
const t = @import("tritonserver_zig");
const command_line = @import("command_line.zig");
const grpc = @import("grpc.zig");
const output = @import("output.zig");
const signals = @import("signals.zig");
const inference_demo = @import("inference_demo.zig");

fn reportError(message: []const u8) void {
    std.log.err("Triton: {s}", .{message});
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = command_line.parse(args[1..]) catch |err| {
        switch (err) {
            error.MissingGrpcListen, error.InvalidGrpcListen => std.log.err("--grpc-listen requires HOST:PORT with a port in 1..65535", .{}),
            error.DuplicateGrpcListen => std.log.err("--grpc-listen may only be supplied once", .{}),
            error.GrpcListenWithOnce => std.log.err("--grpc-listen cannot be combined with --once", .{}),
            else => {},
        }
        return err;
    };
    if (options.help) {
        std.debug.print(
            "Usage: tritonserver_zig [--once] [--list-models] [--model-repository PATH]\n" ++
                "  --infer-demo              Verify input/output tensors with the CPU demo models.\n" ++
                "  --backend-directory PATH  Directory containing Triton backends.\n" ++
                "  --allow-model-errors      Start despite model errors and list failure reasons.\n" ++
                "  --grpc-listen HOST:PORT   Serve native Triton gRPC (requires -Dgrpc=true).\n" ++
                "  --once  Check liveness and shut down immediately.\n" ++
                "  Otherwise, SIGINT (Ctrl-C) or SIGTERM requests clean shutdown.\n",
            .{},
        );
        return;
    }
    if (options.grpc_listen != null and !grpc.enabled) {
        std.log.err("gRPC listener requires a build with -Dgrpc=true", .{});
        return error.GrpcUnavailable;
    }

    const handlers = signals.Handlers.install();
    defer handlers.restore();

    // Only the default empty repository is created automatically.
    if (std.mem.eql(u8, options.repository, command_line.default_repository))
        try std.Io.Dir.cwd().createDirPath(init.io, command_line.default_repository);
    var repository = try t.Repository.scan(init.gpa, init.io, options.repository);
    defer repository.deinit();
    var server = try t.Server.init(init.gpa, .{
        .model_repository = repository.path,
        .backend_directory = options.backend_directory,
        .exit_on_error = !options.allow_model_errors,
        .on_error = reportError,
    });
    defer server.deinit() catch |err| std.log.err("Server cleanup failed: {s}", .{@errorName(err)});
    const version = try t.apiVersion();
    const live = try server.isLive();
    const ready = try server.isReady();
    std.debug.print("Triton API {d}.{d}; live={}; ready={}\n", .{ version.major, version.minor, live, ready });
    if (!live) return error.ServerNotLive;
    if (options.list_models) {
        var models = try server.listModels(init.gpa);
        defer models.deinit();
        output.printModels(models.models);
    }

    if (options.infer_demo) try inference_demo.run(init.gpa, init.io, &server);

    if (!options.once) {
        var frontend: grpc.Frontend = .{};
        if (options.grpc_listen) |listen| {
            try frontend.start(init.gpa, &server, listen);
            std.debug.print("gRPC listening on {s}:{d}\n", .{ listen.address, listen.port });
        }
        defer frontend.shutdown(&server);
        std.debug.print("Model repository: {s}\nPress Ctrl-C to stop.\n", .{options.repository});
        while (!signals.isRequested())
            try std.Io.sleep(init.io, .fromMilliseconds(100), .awake);
    }

    try server.stop();
    try server.deinit();
    std.debug.print("Server stopped; server and options released.\n", .{});
    if (!ready) return error.ServerNotReady;
}
