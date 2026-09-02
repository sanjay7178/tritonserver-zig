const c = @import("translated");

pub const DataType = enum(c.TRITONSERVER_DataType) {
    invalid = c.TRITONSERVER_TYPE_INVALID,
    boolean = c.TRITONSERVER_TYPE_BOOL,
    fp16 = c.TRITONSERVER_TYPE_FP16,
    fp32 = c.TRITONSERVER_TYPE_FP32,
};

pub const MemoryType = enum(c.TRITONSERVER_MemoryType) {
    cpu = c.TRITONSERVER_MEMORY_CPU,
    cpu_pinned = c.TRITONSERVER_MEMORY_CPU_PINNED,
    gpu = c.TRITONSERVER_MEMORY_GPU,
};