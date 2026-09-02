const std = @import("std");
const t = @import("tritonserver_zig");

pub fn main(init: std.process.Init) !void {
    _ = init.arena.allocator();
    var major: u32 = 0;
    var minor: u32 = 0;
    try t.check(t.c.TRITONSERVER_ApiVersion(&major, &minor));

    // Triton expects a repository containing directories named after models.
    // An empty repository is sufficient for this standalone server sample.
    const repo_path = "/home/sanjay7178/tritonserver-zig/examples/minimal";


    var opts: ?*t.c.TRITONSERVER_ServerOptions = null;
    try t.check(t.c.TRITONSERVER_ServerOptionsNew(&opts));
    defer _ = t.c.TRITONSERVER_ServerOptionsDelete(opts);

    try t.check(
        t.c.TRITONSERVER_ServerOptionsSetModelRepositoryPath(opts, repo_path),
        
    );

    try t.check(t.c.TRITONSERVER_ServerOptionsSetBackendDirectory(
    opts,
    "core/build/install/backends",
    ));

    var server: ?*t.c.TRITONSERVER_Server = null;
    try t.check(t.c.TRITONSERVER_ServerNew(&server, opts));
    defer _ = t.c.TRITONSERVER_ServerDelete(server);

    var live = false;
    try t.check(t.c.TRITONSERVER_ServerIsLive(server, &live));

    std.debug.print("Triton API {d}.{d}; live={}\n", .{ major, minor, live });
    std.debug.print("Server is running with an empty model repository. Press Ctrl-C to stop.\n", .{});

    while (true) {
        std.Io.sleep(init.io, .fromSeconds(1), .awake) catch break;
    }
}
