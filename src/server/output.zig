const std = @import("std");
const t = @import("tritonserver_zig");

fn printTensors(label: []const u8, tensors: []const t.TensorInfo) void {
    for (tensors) |tensor| {
        std.debug.print("  {s} {s}: {s} [", .{ label, tensor.name, @tagName(tensor.datatype) });
        for (tensor.shape, 0..) |dim, i| {
            if (i != 0) std.debug.print(", ", .{});
            std.debug.print("{d}", .{dim});
        }
        std.debug.print("]\n", .{});
    }
}

pub fn printModels(models: []const t.ModelInfo) void {
    std.debug.print("Models: {d}\n", .{models.len});
    for (models) |model| {
        std.debug.print("{s} version=", .{model.name});
        if (model.version) |version| std.debug.print("{d}", .{version}) else std.debug.print("unknown", .{});
        std.debug.print(" state={s}\n", .{model.state});
        if (model.reason) |reason| std.debug.print("  reason: {s}\n", .{reason});
        if (model.metadata) |metadata| {
            std.debug.print("  platform={s} max_batch_size={d}\n", .{ metadata.platform, metadata.max_batch_size });
            printTensors("input", metadata.inputs);
            printTensors("output", metadata.outputs);
        }
    }
}
