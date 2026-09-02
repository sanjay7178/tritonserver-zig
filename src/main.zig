const std = @import("std");
const tritonserver_zig = @import("tritonserver_zig");

pub fn main() !void {
    // Prints to stderr, unbuffered, ignoring potential errors.
    std.debug.print("All your {s} are belong to us.\n", .{"codebase"});
}
