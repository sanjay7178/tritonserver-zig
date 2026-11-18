const std = @import("std");

pub const default_repository = ".tritonserver-zig-sample-model-repository";
pub const GrpcListen = struct {
    address: []const u8,
    port: u16,
};

pub const Options = struct {
    repository: [:0]const u8 = default_repository,
    backend_directory: [:0]const u8 = "core/build/install/backends",
    grpc_listen: ?GrpcListen = null,
    list_models: bool = false,
    infer_demo: bool = false,
    allow_model_errors: bool = false,
    once: bool = false,
    help: bool = false,
};

pub fn parse(args: []const [:0]const u8) !Options {
    var options: Options = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--once")) {
            options.once = true;
        } else if (std.mem.eql(u8, arg, "--infer-demo")) {
            options.infer_demo = true;
        } else if (std.mem.eql(u8, arg, "--list-models")) {
            options.list_models = true;
        } else if (std.mem.eql(u8, arg, "--allow-model-errors")) {
            options.allow_model_errors = true;
        } else if (std.mem.eql(u8, arg, "--backend-directory")) {
            i += 1;
            if (i == args.len or args[i].len == 0 or std.mem.startsWith(u8, args[i], "--"))
                return error.MissingBackendDirectory;
            options.backend_directory = args[i];
        } else if (std.mem.eql(u8, arg, "--grpc-listen")) {
            i += 1;
            if (i == args.len or args[i].len == 0 or std.mem.startsWith(u8, args[i], "--"))
                return error.MissingGrpcListen;
            if (options.grpc_listen != null) return error.DuplicateGrpcListen;
            options.grpc_listen = try parseGrpcListen(args[i]);
        } else if (std.mem.eql(u8, arg, "--help")) {
            options.help = true;
        } else if (std.mem.eql(u8, arg, "--model-repository")) {
            i += 1;
            if (i == args.len or args[i].len == 0 or std.mem.startsWith(u8, args[i], "--"))
                return error.MissingModelRepository;
            options.repository = args[i];
        } else {
            std.log.err("Unknown argument: {s}", .{arg});
            return error.UnknownArgument;
        }
    }
    if (options.once and options.grpc_listen != null) return error.GrpcListenWithOnce;
    return options;
}

fn parseGrpcListen(value: []const u8) !GrpcListen {
    const separator = std.mem.findScalarLast(u8, value, ':') orelse return error.InvalidGrpcListen;
    const address = value[0..separator];
    const port_text = value[separator + 1 ..];
    if (address.len == 0 or port_text.len == 0) return error.InvalidGrpcListen;
    for (port_text) |byte| if (!std.ascii.isDigit(byte)) return error.InvalidGrpcListen;
    const port = std.fmt.parseInt(u16, port_text, 10) catch return error.InvalidGrpcListen;
    if (port == 0) return error.InvalidGrpcListen;

    if (address[0] == '[') {
        if (address[address.len - 1] != ']') return error.InvalidGrpcListen;
        _ = std.Io.net.Ip6Address.parse(address[1 .. address.len - 1], port) catch return error.InvalidGrpcListen;
    } else {
        for (address) |byte|
            if (!std.ascii.isAlphanumeric(byte) and byte != '.' and byte != '-') return error.InvalidGrpcListen;
    }
    return .{ .address = address, .port = port };
}

test "starter accepts finite smoke mode and a sentinel-terminated repository path" {
    const options = try parse(&.{ "--once", "--model-repository", "/tmp/models" });
    try std.testing.expect(options.once);
    try std.testing.expectEqualStrings("/tmp/models", options.repository);
    try std.testing.expectEqual(@as(u8, 0), options.repository[options.repository.len]);
}

test "starter rejects a missing repository argument" {
    try std.testing.expectError(error.MissingModelRepository, parse(&.{"--model-repository"}));
    try std.testing.expectError(error.MissingModelRepository, parse(&.{ "--model-repository", "--once" }));
}

test "starter accepts model listing, backend paths and diagnostic startup" {
    const options = try parse(&.{ "--once", "--list-models", "--backend-directory", "/tmp/custom backends", "--allow-model-errors" });
    try std.testing.expect(options.once and options.list_models and options.allow_model_errors);
    try std.testing.expectEqualStrings("/tmp/custom backends", options.backend_directory);
    try std.testing.expectError(error.MissingBackendDirectory, parse(&.{"--backend-directory"}));
    try std.testing.expectError(error.MissingBackendDirectory, parse(&.{ "--backend-directory", "--once" }));
}

test "starter accepts the finite inference demo" {
    const options = try parse(&.{ "--once", "--infer-demo" });
    try std.testing.expect(options.once and options.infer_demo);
}

test "gRPC is opt-in and accepts IPv4, DNS and bracketed IPv6" {
    try std.testing.expect((try parse(&.{})).grpc_listen == null);
    const examples = [_]struct { value: [:0]const u8, address: []const u8, port: u16 }{
        .{ .value = "127.0.0.1:8001", .address = "127.0.0.1", .port = 8001 },
        .{ .value = "localhost:1", .address = "localhost", .port = 1 },
        .{ .value = "[::1]:65535", .address = "[::1]", .port = 65535 },
    };
    for (examples) |example| {
        const listen = (try parse(&.{ "--grpc-listen", example.value })).grpc_listen.?;
        try std.testing.expectEqualStrings(example.address, listen.address);
        try std.testing.expectEqual(example.port, listen.port);
    }
}

test "gRPC rejects missing, malformed and repeated listeners" {
    try std.testing.expectError(error.MissingGrpcListen, parse(&.{"--grpc-listen"}));
    try std.testing.expectError(error.MissingGrpcListen, parse(&.{ "--grpc-listen", "--once" }));
    for ([_][:0]const u8{ "localhost", ":8001", "localhost:", "localhost:0", "localhost:65536", "localhost:+8001", "localhost:8_001", "localhost :8001", "[localhost]:8001", "::1:8001" }) |value|
        try std.testing.expectError(error.InvalidGrpcListen, parse(&.{ "--grpc-listen", value }));
    try std.testing.expectError(error.DuplicateGrpcListen, parse(&.{ "--grpc-listen", "localhost:8001", "--grpc-listen", "localhost:8002" }));
    try std.testing.expectError(error.GrpcListenWithOnce, parse(&.{ "--grpc-listen", "localhost:8001", "--once" }));
}
