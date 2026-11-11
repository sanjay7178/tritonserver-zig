//! Public API for the Triton Server Zig bindings.
//!
//! `c` exposes the complete upstream tritonserver.h API. Handles are opaque;
//! follow the header's ownership rules and pass each returned error to check.
pub const c = @import("core/bindings/c.zig").raw;
pub const TritonErr = @import("core/bindings/error.zig").TritonErr;
pub const check = @import("core/bindings/error.zig").check;
pub const DataType = @import("core/bindings/datatype.zig").DataType;
pub const MemoryType = @import("core/bindings/datatype.zig").MemoryType;
pub const Server = @import("core/server.zig").Server;
pub const ServerOptions = @import("core/server.zig").ServerOptions;
pub const apiVersion = @import("core/server.zig").apiVersion;
pub const ApiVersion = @import("core/server.zig").ApiVersion;
pub const Repository = @import("core/model/repository.zig").Repository;
pub const RepositoryModel = @import("core/model/repository.zig").RepositoryModel;
pub const ModelList = @import("core/model/metadata.zig").ModelList;
pub const ModelInfo = @import("core/model/metadata.zig").ModelInfo;
pub const ModelMetadata = @import("core/model/metadata.zig").ModelMetadata;
pub const TensorInfo = @import("core/model/metadata.zig").TensorInfo;
pub const SchedulingInfo = @import("core/model/metadata.zig").SchedulingInfo;
pub const Request = @import("core/inference/request.zig").Request;
pub const PendingInference = @import("core/inference/operation.zig").PendingInference;
pub const InferenceResponse = @import("core/inference/response.zig").InferenceResponse;
pub const OutputTensor = @import("core/inference/tensor.zig").OutputTensor;
pub const InputTensor = @import("core/inference/tensor.zig").InputTensor;
pub const Scheduler = @import("core/scheduler/scheduler.zig").Scheduler;
pub const SchedulerOptions = @import("core/scheduler/scheduler.zig").SchedulerOptions;
pub const SubmitOptions = @import("core/scheduler/scheduler.zig").SubmitOptions;
pub const SchedulerStats = @import("core/scheduler/scheduler.zig").SchedulerStats;
pub const ScheduledInference = @import("core/scheduler/scheduler.zig").ScheduledInference;

test {
    _ = @import("tests/integration/bindings_test.zig");
    _ = @import("tests/unit/model_test.zig");
    _ = @import("tests/unit/repository_test.zig");
    _ = @import("tests/integration/server_test.zig");
    _ = @import("tests/integration/inference_test.zig");
    _ = @import("tests/unit/tensor_test.zig");
    _ = @import("core/inference/operation.zig");
    _ = @import("core/scheduler/queue.zig");
    _ = @import("core/scheduler/scheduler.zig");
    _ = @import("tests/unit/scheduler_tensor_test.zig");
    _ = @import("tests/unit/scheduler_response_test.zig");
    _ = @import("tests/unit/scheduler_metadata_test.zig");
    _ = @import("tests/integration/scheduler_test.zig");
}
