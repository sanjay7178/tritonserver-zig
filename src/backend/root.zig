//! Zig-native Triton backend ABI, borrowed request views and tagged buffers.
pub const c = @import("../core/bindings/c.zig").raw;
pub const check = @import("../core/bindings/error.zig").check;
pub const tensor = @import("tensor.zig");
pub const ConstBuffer = tensor.ConstBuffer;
pub const Buffer = tensor.Buffer;
pub const byteCount = tensor.byteCount;
pub const copyCpu = tensor.copyCpu;
pub const PluginLibrary = @import("library.zig").PluginLibrary;
pub const VTable = @import("library.zig").VTable;
pub const sdk = @import("sdk.zig");
pub const RequestView = sdk.RequestView;
pub const InputView = sdk.InputView;
pub const InputProperties = sdk.InputProperties;
pub const Response = sdk.Response;
pub const errorToC = sdk.errorToC;
pub const exportBackend = sdk.exportBackend;

test {
    _ = @import("tensor.zig");
    _ = @import("library.zig");
    _ = @import("sdk.zig");
}
