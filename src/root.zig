//! Public API for the Triton Server Zig bindings.
pub const c = @import("translated");
pub const TritonErr = @import("error.zig").TritonErr;
pub const check = @import("error.zig").check;
const std = @import("std");
const Io = std.Io;
pub const DataType = @import("datatypes.zig").DataType;
pub const MemoryType = @import("datatypes.zig").MemoryType;