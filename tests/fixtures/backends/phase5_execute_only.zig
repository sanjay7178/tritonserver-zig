const c = @import("tritonserver_zig").c;

// Symbol-inspection fixture: no lifecycle callbacks, and never executed.
export fn TRITONBACKEND_ModelInstanceExecute(_: ?*c.TRITONBACKEND_ModelInstance, _: [*c]?*c.TRITONBACKEND_Request, _: u32) callconv(.c) ?*c.TRITONSERVER_Error {
    return null;
}
