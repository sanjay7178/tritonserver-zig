const std = @import("std");

var stop_requested: std.atomic.Value(bool) = .init(false);

// Signal handlers only set a lock-free flag. All Triton calls and cleanup run
// on the main thread, outside the signal handler.
fn requestStop(_: std.posix.SIG) callconv(.c) void {
    stop_requested.store(true, .monotonic);
}

pub const Handlers = struct {
    old_int: std.posix.Sigaction,
    old_term: std.posix.Sigaction,

    pub fn install() Handlers {
        stop_requested.store(false, .monotonic);
        const action: std.posix.Sigaction = .{
            .handler = .{ .handler = requestStop },
            .mask = std.posix.sigemptyset(),
            .flags = 0,
        };
        var handlers: Handlers = undefined;
        std.posix.sigaction(.INT, &action, &handlers.old_int);
        std.posix.sigaction(.TERM, &action, &handlers.old_term);
        return handlers;
    }

    pub fn restore(self: *const Handlers) void {
        std.posix.sigaction(.TERM, &self.old_term, null);
        std.posix.sigaction(.INT, &self.old_int, null);
    }
};

pub fn isRequested() bool {
    return stop_requested.load(.monotonic);
}
