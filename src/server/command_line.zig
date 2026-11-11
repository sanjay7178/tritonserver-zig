const std = @import("std");

pub const default_repository = ".tritonserver-zig-sample-model-repository";
pub const Options = struct {
    repository: [:0]const u8 = default_repository,
    backend_directory: [:0]const u8 = "core/build/install/backends",
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
    return options;
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
