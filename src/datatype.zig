const c = @import("translated");

pub const DataType = enum(c.TRITONSERVER_DataType) {
    invalid   = c.TRITONSERVER_TYPE_INVALID,
    boolean   = c.TRITONSERVER_TYPE_BOOL,
    uint8     = c.TRITONSERVER_TYPE_UINT8,
    uint16    = c.TRITONSERVER_TYPE_UINT16,
    uint32    = c.TRITONSERVER_TYPE_UINT32,
    uint64    = c.TRITONSERVER_TYPE_UINT64,
    int8      = c.TRITONSERVER_TYPE_INT8,
    int16     = c.TRITONSERVER_TYPE_INT16,
    int32     = c.TRITONSERVER_TYPE_INT32,
    int64     = c.TRITONSERVER_TYPE_INT64,
    fp16      = c.TRITONSERVER_TYPE_FP16,
    fp32      = c.TRITONSERVER_TYPE_FP32,
    fp64      = c.TRITONSERVER_TYPE_FP64,
    bytes     = c.TRITONSERVER_TYPE_BYTES,
    bf16      = c.TRITONSERVER_TYPE_BF16,
};

pub const MemoryType = enum(c.TRITONSERVER_MemoryType) {
    cpu = c.TRITONSERVER_MEMORY_CPU,
    cpu_pinned = c.TRITONSERVER_MEMORY_CPU_PINNED,
    gpu = c.TRITONSERVER_MEMORY_GPU,
};