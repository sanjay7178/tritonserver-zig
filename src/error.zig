const c = @import("translated");

pub const TritonErr = error{
    Unknown,
    Internal,
    NotFound,
    InvalidArg,
    Unavailable,
    Unsupported,
    AlreadyExists,
    Cancelled,
};

pub fn check(err: ?*c.TRITONSERVER_Error) TritonErr!void {
    const triton_err = err orelse return;
    defer c.TRITONSERVER_ErrorDelete(triton_err);

    return switch (c.TRITONSERVER_ErrorCode(triton_err)) {
        c.TRITONSERVER_ERROR_UNKNOWN => error.Unknown,
        c.TRITONSERVER_ERROR_INTERNAL => error.Internal,
        c.TRITONSERVER_ERROR_NOT_FOUND => error.NotFound,
        c.TRITONSERVER_ERROR_INVALID_ARG => error.InvalidArg,
        c.TRITONSERVER_ERROR_UNAVAILABLE => error.Unavailable,
        c.TRITONSERVER_ERROR_UNSUPPORTED => error.Unsupported,
        c.TRITONSERVER_ERROR_ALREADY_EXISTS => error.AlreadyExists,
        c.TRITONSERVER_ERROR_CANCELLED => error.Cancelled,
        else => error.Unknown,
    };
}