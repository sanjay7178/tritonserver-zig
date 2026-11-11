//! Optional CUDA copy engine. This module is built only with backend-cuda=true.
const std = @import("std");
const t = @import("tritonserver_zig");
const b = t.backend;
const c = t.c;
pub const raw = @import("cuda_c");

pub fn check(status: raw.cudaError_t) error{CudaFailure}!void {
    if (status != raw.cudaSuccess) {
        const message = raw.cudaGetErrorString(status);
        if (message != null) std.debug.print("CUDA: {s}\n", .{std.mem.span(message)});
        return error.CudaFailure;
    }
}

/// Move-only, single-instance stream owner. Call finish before publishing a
/// response or releasing an input, including after a failed copy.
pub const Context = struct {
    stream: raw.cudaStream_t,

    pub fn init() !Context {
        try check(raw.cudaSetDevice(0));
        var stream: raw.cudaStream_t = null;
        try check(raw.cudaStreamCreateWithFlags(&stream, raw.cudaStreamNonBlocking));
        return .{ .stream = stream orelse return error.CudaFailure };
    }

    pub fn deinit(self: *Context) void {
        const stream = self.stream orelse return;
        self.finish() catch {};
        check(raw.cudaStreamDestroy(stream)) catch {};
        self.stream = null;
    }

    pub fn copy(self: *Context, destination: b.Buffer, source: b.ConstBuffer) !void {
        if (self.stream == null) return error.CudaContextClosed;
        try destination.validate();
        try source.validate();
        if (destination.byte_size != source.byte_size) return error.BufferSizeMismatch;
        for ([_]b.ConstBuffer{ destination.asConst(), source }) |buffer|
            if (buffer.memory_type == c.TRITONSERVER_MEMORY_GPU and buffer.memory_type_id != 0)
                return error.UnsupportedDevice;
        if (source.byte_size == 0) return;
        const destination_gpu = destination.memory_type == c.TRITONSERVER_MEMORY_GPU;
        const source_gpu = source.memory_type == c.TRITONSERVER_MEMORY_GPU;
        if (!destination_gpu and !source_gpu) return b.copyCpu(destination, source);
        try check(raw.cudaSetDevice(0));
        try validateDeviceBuffer(destination.asConst());
        try validateDeviceBuffer(source);
        if (destination_gpu and source_gpu) {
            if (destination.ptr.? == source.ptr.?) return;
            const dst = @intFromPtr(destination.ptr.?);
            const src = @intFromPtr(source.ptr.?);
            const dst_end = std.math.add(usize, dst, destination.byte_size) catch return error.BufferAddressOverflow;
            const src_end = std.math.add(usize, src, source.byte_size) catch return error.BufferAddressOverflow;
            if (dst < src_end and src < dst_end) return error.OverlappingDeviceBuffers;
        }
        const kind: raw.enum_cudaMemcpyKind = if (destination_gpu)
            (if (source_gpu) raw.cudaMemcpyDeviceToDevice else raw.cudaMemcpyHostToDevice)
        else
            raw.cudaMemcpyDeviceToHost;
        try check(raw.cudaMemcpyAsync(destination.ptr, source.ptr, source.byte_size, kind, self.stream));
    }

    pub fn finish(self: *Context) !void {
        const stream = self.stream orelse return error.CudaContextClosed;
        try check(raw.cudaSetDevice(0));
        try check(raw.cudaStreamSynchronize(stream));
    }
};

fn validateDeviceBuffer(buffer: b.ConstBuffer) !void {
    if (buffer.memory_type != c.TRITONSERVER_MEMORY_GPU or buffer.byte_size == 0) return;
    var attributes: raw.struct_cudaPointerAttributes = undefined;
    if (raw.cudaPointerGetAttributes(&attributes, buffer.ptr) != raw.cudaSuccess) {
        _ = raw.cudaGetLastError();
        return error.InvalidDeviceBuffer;
    }
    // This engine handles cudaMalloc device memory, rather than managed memory
    // or host pointers carrying an incorrect GPU memory tag.
    if (attributes.type != raw.cudaMemoryTypeDevice) return error.InvalidDeviceBuffer;
    if (attributes.device != 0) return error.UnsupportedDevice;
}
