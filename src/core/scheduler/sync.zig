const std = @import("std");
const linux = std.os.linux;

pub fn check(status: std.c.E) void {
    if (status != .SUCCESS) @panic("scheduler synchronization failed");
}

pub fn now() u64 {
    var time: std.c.timespec = undefined;
    if (std.c.clock_gettime(.MONOTONIC, &time) != 0) @panic("monotonic clock unavailable");
    return @as(u64, @intCast(time.sec)) * std.time.ns_per_s + @as(u64, @intCast(time.nsec));
}

fn interrupted() bool {
    return std.c._errno().* == @backingInt(std.c.E.INTR);
}

/// A counting wakeup prevents a notification between queue inspection and poll
/// from being lost. Relative poll deadlines use the monotonic clock, not wall time.
pub const Wake = struct {
    fd: std.c.fd_t,

    pub fn init() !Wake {
        const fd = std.c.eventfd(0, linux.EFD.CLOEXEC | linux.EFD.NONBLOCK);
        if (fd < 0) return error.SystemResources;
        return .{ .fd = fd };
    }

    pub fn deinit(self: Wake) void {
        _ = std.c.close(self.fd);
    }

    pub fn notify(self: Wake) void {
        var one: u64 = 1;
        const bytes = std.mem.asBytes(&one);
        while (true) {
            if (std.c.write(self.fd, bytes.ptr, bytes.len) >= 0) return;
            if (interrupted()) continue;
            // A saturated counter is already readable, so a wakeup is pending.
            if (std.c._errno().* == @backingInt(std.c.E.AGAIN)) return;
            @panic("scheduler wakeup write failed");
        }
    }

    pub fn wait(self: Wake, timeout_ns: ?u64) void {
        var fds = [_]std.c.pollfd{.{ .fd = self.fd, .events = std.c.POLL.IN, .revents = 0 }};
        var timeout: std.c.timespec = undefined;
        if (timeout_ns) |ns| timeout = .{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
        const rc = std.c.ppoll(&fds, 1, if (timeout_ns != null) &timeout else null, null);
        if (rc < 0) {
            if (interrupted()) return; // Recompute the remaining deadline.
            @panic("scheduler wakeup poll failed");
        }
        if (rc == 0) return;
        var counter: u64 = undefined;
        const bytes = std.mem.asBytes(&counter);
        while (std.c.read(self.fd, bytes.ptr, bytes.len) < 0) {
            if (interrupted()) continue;
            if (std.c._errno().* == @backingInt(std.c.E.AGAIN)) return;
            @panic("scheduler wakeup read failed");
        }
    }
};
