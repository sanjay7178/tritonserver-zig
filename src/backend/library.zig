const std = @import("std");
const c = @import("../core/bindings/c.zig").raw;

pub const VTable = struct {
    execute: *const c.TRITONZIG_ModelInstanceExecuteFn,
    initialize: ?*const c.TRITONZIG_BackendInitializeFn = null,
    finalize: ?*const c.TRITONZIG_BackendFinalizeFn = null,
    model_initialize: ?*const c.TRITONZIG_ModelInitializeFn = null,
    model_finalize: ?*const c.TRITONZIG_ModelFinalizeFn = null,
    model_instance_initialize: ?*const c.TRITONZIG_ModelInstanceInitializeFn = null,
    model_instance_finalize: ?*const c.TRITONZIG_ModelInstanceFinalizeFn = null,
    model_instance_ready: ?*const c.TRITONZIG_ModelInstanceReadyFn = null,
    get_backend_attribute: ?*const c.TRITONZIG_GetBackendAttributeFn = null,
};

/// Resolve a trusted plugin's ABI without invoking lifecycle callbacks. Triton
/// owns their actual contexts and invocation order. Function pointers are valid
/// only while this move-only library remains open.
pub const PluginLibrary = struct {
    library: ?std.DynLib,
    vtable: VTable,

    pub fn load(path: []const u8) !PluginLibrary {
        if (path.len == 0 or std.mem.indexOfScalar(u8, path, 0) != null) return error.InvalidLibraryPath;
        var library = try std.DynLib.open(path);
        errdefer library.close();
        const execute = library.lookup(*const c.TRITONZIG_ModelInstanceExecuteFn, "TRITONBACKEND_ModelInstanceExecute") orelse return error.MissingExecuteSymbol;
        return .{ .library = library, .vtable = .{
            .execute = execute,
            .initialize = library.lookup(*const c.TRITONZIG_BackendInitializeFn, "TRITONBACKEND_Initialize"),
            .finalize = library.lookup(*const c.TRITONZIG_BackendFinalizeFn, "TRITONBACKEND_Finalize"),
            .model_initialize = library.lookup(*const c.TRITONZIG_ModelInitializeFn, "TRITONBACKEND_ModelInitialize"),
            .model_finalize = library.lookup(*const c.TRITONZIG_ModelFinalizeFn, "TRITONBACKEND_ModelFinalize"),
            .model_instance_initialize = library.lookup(*const c.TRITONZIG_ModelInstanceInitializeFn, "TRITONBACKEND_ModelInstanceInitialize"),
            .model_instance_finalize = library.lookup(*const c.TRITONZIG_ModelInstanceFinalizeFn, "TRITONBACKEND_ModelInstanceFinalize"),
            .model_instance_ready = library.lookup(*const c.TRITONZIG_ModelInstanceReadyFn, "TRITONBACKEND_ModelInstanceReady"),
            .get_backend_attribute = library.lookup(*const c.TRITONZIG_GetBackendAttributeFn, "TRITONBACKEND_GetBackendAttribute"),
        } };
    }

    pub fn deinit(self: *PluginLibrary) void {
        if (self.library) |*library| {
            library.close();
            self.library = null;
        }
    }
};

test "plugin loader requires execute but accepts absent optional callbacks" {
    var library = try PluginLibrary.load(".zig-cache/phase5/execute_only.so");
    defer library.deinit();
    try std.testing.expect(library.vtable.initialize == null);
    try std.testing.expect(library.vtable.finalize == null);
    try std.testing.expect(library.vtable.model_initialize == null);
    try std.testing.expect(library.vtable.model_instance_initialize == null);
    try std.testing.expect(library.vtable.model_instance_ready == null);
    try std.testing.expect(library.vtable.get_backend_attribute == null);
    library.deinit();
    library.deinit();
    try std.testing.expect(library.library == null);
}

test "plugin loader rejects a missing execute symbol and invalid paths" {
    try std.testing.expectError(error.InvalidLibraryPath, PluginLibrary.load(""));
    try std.testing.expectError(error.InvalidLibraryPath, PluginLibrary.load("bad\x00path"));
    // The linked server library is present even with an external Triton prefix,
    // and exports the host API rather than plugin callbacks.
    try std.testing.expectError(error.MissingExecuteSymbol, PluginLibrary.load("libtritonserver.so"));
    const missing = PluginLibrary.load(".zig-cache/phase5/not-a-real-plugin.so");
    if (missing) |value| {
        var owner = value;
        owner.deinit();
        return error.ExpectedMissingLibrary;
    } else |_| {}
}
