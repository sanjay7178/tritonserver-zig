const std = @import("std");
const Server = @import("../server.zig").Server;
const Request = @import("../inference/request.zig").Request;
const tensor = @import("../inference/tensor.zig");
const response = @import("../inference/response.zig");
const model = @import("../model/metadata.zig");
const OwnedArena = @import("../internal/owned_arena.zig").OwnedArena;
const sync = @import("sync.zig");
const MpscQueue = @import("queue.zig").MpscQueue;
const work = std.heap.c_allocator;
pub const InputTensor = tensor.InputTensor;

pub const SchedulerOptions = struct {
    max_batch_size: ?u32 = null,
    max_wait_us: u64 = 5000,
    worker_count: usize = 2,
    /// Bounds accepted queued and executing requests, not unclaimed results.
    queue_capacity: usize = 1024,
};

pub const SubmitOptions = struct {
    /// Empty means all model outputs, matching the direct inference API.
    outputs: []const []const u8 = &.{},
};

pub const SchedulerStats = struct {
    accepted_requests: u64 = 0,
    accepted_samples: u64 = 0,
    dispatched_batches: u64 = 0,
    dispatched_samples: u64 = 0,
    rejected_requests: u64 = 0,
    failed_requests: u64 = 0,
    queue_peak: u64 = 0,
    max_batch_samples: u64 = 0,
    total_queue_wait_ns: u64 = 0,
};

const Counters = struct {
    accepted_requests: std.atomic.Value(u64) = .init(0),
    accepted_samples: std.atomic.Value(u64) = .init(0),
    dispatched_batches: std.atomic.Value(u64) = .init(0),
    dispatched_samples: std.atomic.Value(u64) = .init(0),
    rejected_requests: std.atomic.Value(u64) = .init(0),
    failed_requests: std.atomic.Value(u64) = .init(0),
    queue_peak: std.atomic.Value(u64) = .init(0),
    max_batch_samples: std.atomic.Value(u64) = .init(0),
    total_queue_wait_ns: std.atomic.Value(u64) = .init(0),

    fn snapshot(self: *const Counters) SchedulerStats {
        var result: SchedulerStats = .{};
        inline for (@typeInfo(SchedulerStats).@"struct".field_names) |name|
            @field(result, name) = @field(self, name).load(.monotonic);
        return result;
    }
};

// The future owns its context. Until completion the scheduler owns a borrowed
// reference. Internal buffers/results use libc allocation across all threads;
// the caller allocator is used only by the single waiting thread.
const Job = struct {
    parent: std.mem.Allocator,
    storage: OwnedArena,
    inputs: []const InputTensor,
    outputs: []const []const u8,
    samples: usize,
    accepted_ns: u64 = 0,
    queue_next: ?*Job = null,
    batch_next: ?*Job = null,
    bucket_next: ?*Job = null,
    batch_tail: *Job = undefined,
    batch_samples: usize = 0,
    batch_oldest_ns: u64 = 0,
    ready_next: ?*Job = null,
    mutex: std.c.pthread_mutex_t = .{},
    condition: std.c.pthread_cond_t = .{},
    done: bool = false,
    failure: ?anyerror = null,
    result: ?response.InferenceResponse = null,

    fn wait(self: *Job) void {
        sync.check(std.c.pthread_mutex_lock(&self.mutex));
        while (!self.done) sync.check(std.c.pthread_cond_wait(&self.condition, &self.mutex));
        sync.check(std.c.pthread_mutex_unlock(&self.mutex));
    }

    fn destroy(self: *Job) void {
        if (self.result) |*result| result.deinit();
        self.storage.deinit();
        sync.check(std.c.pthread_cond_destroy(&self.condition));
        sync.check(std.c.pthread_mutex_destroy(&self.mutex));
        work.destroy(self);
    }
};

/// Single waiter, move-only. Deinit drains accepted work; it is not cancellation.
/// A future remains usable after scheduler shutdown/deinit.
pub const ScheduledInference = struct {
    context: ?*Job,

    pub fn wait(self: *ScheduledInference) !response.InferenceResponse {
        const job = self.context orelse return error.InferenceAlreadyCompleted;
        job.wait();
        self.context = null;
        defer job.destroy();
        if (job.failure) |err| return err;
        return response.clone(job.parent, &job.result.?);
    }

    pub fn deinit(self: *ScheduledInference) void {
        if (self.context) |job| {
            job.wait();
            self.context = null;
            job.destroy();
        }
    }
};

const closed_bit: usize = @as(usize, 1) << (@bitSizeOf(usize) - 1);

const State = struct {
    parent: std.mem.Allocator,
    server: *const Server,
    models: model.ModelList,
    model_name: []const u8,
    version: i64,
    metadata: model.ModelMetadata,
    options: SchedulerOptions,
    limit: usize,
    wake: sync.Wake,
    workers: []std.Thread,
    dispatcher: ?std.Thread = null,
    shutdown_mutex: std.c.pthread_mutex_t = .{},
    joined: bool = false,
    // High bit closes admission; low bits count producers still inside submit.
    admission: std.atomic.Value(usize) = .init(0),
    pending: std.atomic.Value(usize) = .init(0),
    counters: Counters = .{},
    ingress: MpscQueue(Job) = .{},
    buckets: ?*Job = null, // Dispatcher-only, first job doubles as bucket record.
    ready_mutex: std.c.pthread_mutex_t = .{},
    ready_condition: std.c.pthread_cond_t = .{},
    ready_head: ?*Job = null,
    ready_tail: ?*Job = null,
    workers_stopping: bool = false,

    fn enter(self: *State) !void {
        var value = self.admission.load(.acquire);
        while (true) {
            if ((value & closed_bit) != 0) return error.SchedulerClosed;
            if (self.admission.cmpxchgWeak(value, value + 1, .acq_rel, .acquire)) |actual| value = actual else return;
        }
    }

    fn leave(self: *State) void {
        const previous = self.admission.fetchSub(1, .acq_rel);
        if ((previous & closed_bit) != 0) self.wake.notify();
    }

    fn reserve(self: *State) !void {
        var count = self.pending.load(.acquire);
        while (true) {
            if (count >= self.options.queue_capacity) return error.QueueFull;
            if (self.pending.cmpxchgWeak(count, count + 1, .acq_rel, .acquire)) |actual| count = actual else {
                _ = self.counters.queue_peak.fetchMax(@intCast(count + 1), .monotonic);
                return;
            }
        }
    }

    fn enqueueReady(self: *State, batch: *Job) void {
        sync.check(std.c.pthread_mutex_lock(&self.ready_mutex));
        batch.ready_next = null;
        if (self.ready_tail) |tail| tail.ready_next = batch else self.ready_head = batch;
        self.ready_tail = batch;
        sync.check(std.c.pthread_cond_signal(&self.ready_condition));
        sync.check(std.c.pthread_mutex_unlock(&self.ready_mutex));
    }

    fn stopWorkers(self: *State) void {
        sync.check(std.c.pthread_mutex_lock(&self.ready_mutex));
        self.workers_stopping = true;
        sync.check(std.c.pthread_cond_broadcast(&self.ready_condition));
        sync.check(std.c.pthread_mutex_unlock(&self.ready_mutex));
    }

    fn append(self: *State, job: *Job) void {
        if (self.metadata.max_batch_size == 0 or self.options.max_wait_us == 0) {
            job.batch_samples = job.samples;
            self.enqueueReady(job);
            return;
        }
        var link = &self.buckets;
        while (link.*) |bucket| {
            if (!compatible(bucket, job)) {
                link = &bucket.bucket_next;
                continue;
            }
            if (bucket.batch_samples + job.samples > self.limit or
                windowExpired(bucket.batch_oldest_ns, self.options.max_wait_us, job.accepted_ns, false))
            {
                link.* = bucket.bucket_next;
                self.enqueueReady(bucket);
                break;
            }
            bucket.batch_tail.batch_next = job;
            bucket.batch_tail = job;
            bucket.batch_samples += job.samples;
            // Publication order and timestamp order can differ if a producer
            // was preempted immediately before its CAS. Do not extend its wait.
            bucket.batch_oldest_ns = @min(bucket.batch_oldest_ns, job.accepted_ns);
            if (bucket.batch_samples == self.limit) {
                link.* = bucket.bucket_next;
                self.enqueueReady(bucket);
            }
            return;
        }
        if (job.samples == self.limit) {
            job.batch_samples = job.samples;
            self.enqueueReady(job);
            return;
        }
        job.batch_tail = job;
        job.batch_samples = job.samples;
        job.batch_oldest_ns = job.accepted_ns;
        job.bucket_next = self.buckets;
        self.buckets = job;
    }

    fn dispatch(self: *State) void {
        while (true) {
            var list = self.ingress.take();
            while (list) |job| {
                const next = job.queue_next;
                job.queue_next = null;
                self.append(job);
                list = next;
            }
            const gate = self.admission.load(.acquire);
            const closing = (gate & closed_bit) != 0;
            const now = sync.now();
            var deadline: ?u64 = null;
            var link = &self.buckets;
            while (link.*) |bucket| {
                const due = deadlineFor(bucket.batch_oldest_ns, self.options.max_wait_us);
                if (windowExpired(bucket.batch_oldest_ns, self.options.max_wait_us, now, closing)) {
                    link.* = bucket.bucket_next;
                    self.enqueueReady(bucket);
                } else {
                    deadline = if (deadline) |existing| @min(existing, due) else due;
                    link = &bucket.bucket_next;
                }
            }
            // pending includes publishers and executing jobs, so no accepted
            // node can be missed by the final drain, even if publication raced
            // with the take above. Producers notify again when leaving.
            if (gate == closed_bit and self.pending.load(.acquire) == 0) break;
            self.wake.wait(if (deadline) |due| due -| sync.now() else null);
        }
        self.stopWorkers();
    }

    fn worker(self: *State) void {
        while (true) {
            sync.check(std.c.pthread_mutex_lock(&self.ready_mutex));
            while (self.ready_head == null and !self.workers_stopping)
                sync.check(std.c.pthread_cond_wait(&self.ready_condition, &self.ready_mutex));
            const batch = self.ready_head orelse {
                sync.check(std.c.pthread_mutex_unlock(&self.ready_mutex));
                return;
            };
            self.ready_head = batch.ready_next;
            if (self.ready_head == null) self.ready_tail = null;
            sync.check(std.c.pthread_mutex_unlock(&self.ready_mutex));

            const started = sync.now();
            var cursor: ?*Job = batch;
            while (cursor) |job| {
                _ = self.counters.total_queue_wait_ns.fetchAdd(started -| job.accepted_ns, .monotonic);
                cursor = job.batch_next;
            }
            _ = self.counters.dispatched_batches.fetchAdd(1, .monotonic);
            _ = self.counters.dispatched_samples.fetchAdd(@intCast(batch.batch_samples), .monotonic);
            _ = self.counters.max_batch_samples.fetchMax(@intCast(batch.batch_samples), .monotonic);
            const failure: ?anyerror = blk: {
                self.execute(batch) catch |err| break :blk err;
                break :blk null;
            };
            cursor = batch;
            while (cursor) |job| {
                // Save the next pointer before publishing completion: the
                // waiter can immediately destroy this job, including the head.
                const next = job.batch_next;
                if (failure != null) {
                    if (job.result) |*result| result.deinit();
                    job.result = null;
                    _ = self.counters.failed_requests.fetchAdd(1, .monotonic);
                }
                sync.check(std.c.pthread_mutex_lock(&job.mutex));
                job.failure = failure;
                job.done = true;
                _ = self.pending.fetchSub(1, .acq_rel);
                sync.check(std.c.pthread_cond_signal(&job.condition));
                sync.check(std.c.pthread_mutex_unlock(&job.mutex));
                cursor = next;
            }
            self.wake.notify();
        }
    }

    fn execute(self: *State, batch: *Job) !void {
        var storage = try OwnedArena.init(work);
        defer storage.deinit();
        const alloc = storage.allocator();
        var request = try Request.init(work, self.server, self.model_name, self.version);
        defer request.deinit() catch @panic("scheduler request cleanup failed");
        for (batch.inputs, 0..) |input, index| {
            const dims = try alloc.dupe(i64, input.shape);
            if (self.metadata.max_batch_size != 0) dims[0] = @intCast(batch.batch_samples);
            var byte_count: usize = 0;
            var cursor: ?*Job = batch;
            while (cursor) |job| {
                byte_count = std.math.add(usize, byte_count, job.inputs[index].data.len) catch return error.ShapeOverflow;
                cursor = job.batch_next;
            }
            const bytes = try alloc.alignedAlloc(u8, .@"16", byte_count);
            var offset: usize = 0;
            cursor = batch;
            while (cursor) |job| {
                const part = job.inputs[index].data;
                @memcpy(bytes[offset..][0..part.len], part);
                offset += part.len;
                cursor = job.batch_next;
            }
            try request.addInputBytes(input.datatype, input.name, dims, bytes);
        }
        for (batch.outputs) |name| try request.requestOutput(name);
        var aggregate = try self.server.inferSync(work, &request);
        defer aggregate.deinit();
        if (self.metadata.max_batch_size != 0) {
            for (aggregate.outputs) |output| {
                if (output.shape.len == 0 or output.shape[0] != batch.batch_samples or
                    try tensor.byteCount(output.datatype, output.shape) != output.data.len)
                    return error.InvalidBatchOutput;
            }
        }
        if (batch.batch_next == null) {
            batch.result = aggregate;
            // Transfer ownership without deinitializing the moved arena.
            aggregate.storage.arena = null;
            return;
        }
        var cursor: ?*Job = batch;
        var offset: usize = 0;
        while (cursor) |job| {
            job.result = response.sliceBatch(work, &aggregate, offset, job.samples, batch.batch_samples) catch |err| switch (err) {
                error.InvalidBatchShape, error.InvalidBatchRange, error.InvalidOutputBuffer => return error.InvalidBatchOutput,
                else => return err,
            };
            offset += job.samples;
            cursor = job.batch_next;
        }
    }
};

fn deadlineFor(accepted_ns: u64, wait_us: u64) u64 {
    return accepted_ns +| (wait_us * std.time.ns_per_us);
}

fn windowExpired(accepted_ns: u64, wait_us: u64, now_ns: u64, closing: bool) bool {
    return closing or now_ns >= deadlineFor(accepted_ns, wait_us);
}

fn compatible(first: *const Job, other: *const Job) bool {
    for (first.inputs, other.inputs) |a, b| {
        if (a.datatype != b.datatype or !std.mem.eql(i64, a.shape[1..], b.shape[1..])) return false;
    }
    if (first.outputs.len != other.outputs.len) return false;
    for (first.outputs, other.outputs) |a, b| if (!std.mem.eql(u8, a, b)) return false;
    return true;
}

/// A move-only owner of heap-stable scheduling state. Submission and shutdown
/// may overlap. Serialize destruction with all other API calls. Keep Server
/// stable/alive until shutdown returns; do not poll or stop it concurrently.
/// Server.on_error may run concurrently on worker threads.
pub const Scheduler = struct {
    state: ?*State,

    pub fn init(parent: std.mem.Allocator, server: *const Server, name: []const u8, version: i64, options: SchedulerOptions) !Scheduler {
        if (options.worker_count == 0 or options.queue_capacity == 0 or options.queue_capacity >= closed_bit or
            options.max_wait_us > std.math.maxInt(u64) / std.time.ns_per_us or options.max_batch_size == 0)
            return error.InvalidSchedulerOptions;
        if (version != -1 and version <= 0) return error.InvalidModelVersion;
        var models = try server.listModels(parent);
        errdefer models.deinit();
        var selected: ?model.ModelInfo = null;
        for (models.models) |item| {
            if (!std.mem.eql(u8, name, item.name) or !std.mem.eql(u8, item.state, "READY")) continue;
            if (version != -1 and version != item.version.?) continue;
            if (selected == null or item.version.? > selected.?.version.?) selected = item;
        }
        const target = selected orelse return error.NotFound;
        const metadata = target.metadata orelse return error.Unavailable;
        const flags = metadata.scheduling;
        if (flags.dynamic_batching) return error.NativeBatchingEnabled;
        if (flags.sequence_batching or flags.decoupled or flags.special_tensors) return error.UnsupportedSchedulingModel;
        if (metadata.max_batch_size != 0 and metadata.inputs.len == 0) return error.UnsupportedSchedulingModel;
        for (metadata.inputs) |input| _ = try tensor.byteWidth(input.datatype);
        for (metadata.outputs) |output| _ = try tensor.byteWidth(output.datatype);
        const limit: usize = options.max_batch_size orelse @max(metadata.max_batch_size, 1);
        if (limit > @max(metadata.max_batch_size, 1)) return error.InvalidSchedulerOptions;
        const workers = try parent.alloc(std.Thread, options.worker_count);
        errdefer parent.free(workers);
        const wake = try sync.Wake.init();
        errdefer wake.deinit();
        const state = try parent.create(State);
        errdefer parent.destroy(state);
        state.* = .{
            .parent = parent,
            .server = server,
            .models = models,
            .model_name = target.name,
            .version = target.version.?,
            .metadata = metadata,
            .options = options,
            .limit = limit,
            .wake = wake,
            .workers = workers,
        };
        errdefer {
            sync.check(std.c.pthread_cond_destroy(&state.ready_condition));
            sync.check(std.c.pthread_mutex_destroy(&state.ready_mutex));
            sync.check(std.c.pthread_mutex_destroy(&state.shutdown_mutex));
        }
        var started: usize = 0;
        errdefer {
            state.stopWorkers();
            for (workers[0..started]) |thread| thread.join();
        }
        for (workers) |*thread| {
            thread.* = try std.Thread.spawn(.{}, State.worker, .{state});
            started += 1;
        }
        state.dispatcher = try std.Thread.spawn(.{}, State.dispatch, .{state});
        return .{ .state = state };
    }

    pub fn submit(self: *const Scheduler, parent: std.mem.Allocator, inputs: []const InputTensor, options: SubmitOptions) !ScheduledInference {
        const state = self.state orelse return error.SchedulerClosed;
        errdefer _ = state.counters.rejected_requests.fetchAdd(1, .monotonic);
        try state.enter();
        defer state.leave();
        try state.reserve();
        errdefer {
            _ = state.pending.fetchSub(1, .acq_rel);
            state.wake.notify();
        }
        for (inputs, 0..) |input, i| {
            try input.validate();
            for (inputs[0..i]) |previous|
                if (std.mem.eql(u8, input.name, previous.name)) return error.DuplicateInput;
        }
        if (inputs.len != state.metadata.inputs.len) return error.MissingInput;
        var storage = try OwnedArena.init(work);
        errdefer storage.deinit();
        const alloc = storage.allocator();
        const copied = try alloc.alloc(InputTensor, inputs.len);
        var samples: ?usize = null;
        for (state.metadata.inputs, copied) |expected, *destination| {
            var matched: ?InputTensor = null;
            for (inputs) |input| {
                if (std.mem.eql(u8, input.name, expected.name)) {
                    if (matched != null) return error.DuplicateInput;
                    matched = input;
                }
            }
            const input = matched orelse return error.UnknownInput;
            try input.validate();
            if (input.datatype != expected.datatype) return error.DataTypeMismatch;
            if (input.shape.len != expected.shape.len) return error.InvalidShape;
            for (input.shape, expected.shape) |actual, dimension| {
                if (dimension != -1 and actual != dimension) return error.InvalidShape;
            }
            const count: usize = if (state.metadata.max_batch_size == 0) 1 else blk: {
                if (input.shape.len == 0 or input.shape[0] <= 0) return error.InvalidShape;
                break :blk std.math.cast(usize, input.shape[0]) orelse return error.ShapeOverflow;
            };
            if (count > state.limit) return error.BatchTooLarge;
            if (samples) |existing| {
                if (existing != count) return error.InvalidShape;
            } else samples = count;
            const bytes = try alloc.alignedAlloc(u8, .@"16", input.data.len);
            @memcpy(bytes, input.data);
            destination.* = .{ .name = expected.name, .datatype = input.datatype, .shape = try alloc.dupe(i64, input.shape), .data = bytes };
        }
        // Resolve all-output requests and canonicalize subsets in model order.
        // This makes output-set equality independent of caller ordering.
        for (options.outputs, 0..) |name, i| {
            for (options.outputs[0..i]) |previous| if (std.mem.eql(u8, previous, name)) return error.DuplicateOutput;
            var found = false;
            for (state.metadata.outputs) |output| if (std.mem.eql(u8, output.name, name)) {
                found = true;
                break;
            };
            if (!found) return error.UnknownOutput;
        }
        const outputs = try alloc.alloc([]const u8, if (options.outputs.len == 0) state.metadata.outputs.len else options.outputs.len);
        var output_index: usize = 0;
        for (state.metadata.outputs) |output| {
            var include = options.outputs.len == 0;
            for (options.outputs) |name| if (std.mem.eql(u8, name, output.name)) {
                include = true;
                break;
            };
            if (include) {
                outputs[output_index] = try alloc.dupe(u8, output.name);
                output_index += 1;
            }
        }
        // Names must not borrow the scheduler's snapshot: futures can outlive it.
        for (copied) |*input| input.name = try alloc.dupe(u8, input.name);
        const job = try work.create(Job);
        job.* = .{ .parent = parent, .storage = storage, .inputs = copied, .outputs = outputs, .samples = samples orelse 1 };
        job.accepted_ns = sync.now();
        _ = state.counters.accepted_requests.fetchAdd(1, .monotonic);
        _ = state.counters.accepted_samples.fetchAdd(@intCast(job.samples), .monotonic);
        state.ingress.push(job);
        state.wake.notify();
        return .{ .context = job };
    }

    /// Snapshots counters independently; call after shutdown for a final report.
    pub fn stats(self: *const Scheduler) SchedulerStats {
        return if (self.state) |state| state.counters.snapshot() else .{};
    }

    pub fn shutdown(self: *Scheduler) void {
        const state = self.state orelse return;
        sync.check(std.c.pthread_mutex_lock(&state.shutdown_mutex));
        defer sync.check(std.c.pthread_mutex_unlock(&state.shutdown_mutex));
        if (state.joined) return;
        _ = state.admission.fetchOr(closed_bit, .acq_rel);
        state.wake.notify();
        state.dispatcher.?.join();
        for (state.workers) |thread| thread.join();
        state.joined = true;
    }

    pub fn deinit(self: *Scheduler) void {
        const state = self.state orelse return;
        self.shutdown();
        self.state = null;
        state.wake.deinit();
        state.models.deinit();
        state.parent.free(state.workers);
        sync.check(std.c.pthread_cond_destroy(&state.ready_condition));
        sync.check(std.c.pthread_mutex_destroy(&state.ready_mutex));
        sync.check(std.c.pthread_mutex_destroy(&state.shutdown_mutex));
        state.parent.destroy(state);
    }
};

test "batch deadline is based on the oldest arrival and saturates safely" {
    try std.testing.expectEqual(@as(u64, 5_000_123), deadlineFor(123, 5000));
    try std.testing.expectEqual(@as(u64, 123), deadlineFor(123, 0));
    try std.testing.expectEqual(std.math.maxInt(u64), deadlineFor(std.math.maxInt(u64) - 1, 1));
    // A fake monotonic clock exercises the exact deadline without sleeping.
    try std.testing.expect(!windowExpired(123, 5000, 5_000_122, false));
    try std.testing.expect(windowExpired(123, 5000, 5_000_123, false));
    try std.testing.expect(windowExpired(123, 5000, 5_000_124, false));
    try std.testing.expect(windowExpired(123, 5000, 123, true));
    try std.testing.expect(windowExpired(123, 0, 123, false));
}

test "dispatcher preserves FIFO and oldest deadlines across full and delayed arrivals" {
    var state: State = .{
        .parent = std.testing.allocator,
        .server = undefined,
        .models = undefined,
        .model_name = "policy-only",
        .version = 1,
        .metadata = .{ .platform = "", .max_batch_size = 8, .inputs = &.{}, .outputs = &.{} },
        .options = .{},
        .limit = 8,
        .wake = undefined,
        .workers = &.{},
    };
    defer sync.check(std.c.pthread_cond_destroy(&state.ready_condition));
    defer sync.check(std.c.pthread_mutex_destroy(&state.ready_mutex));
    defer sync.check(std.c.pthread_mutex_destroy(&state.shutdown_mutex));
    // These records test dispatcher policy only; no inference or future owns
    // their borrowed tensors, so storage is deliberately not instantiated.
    var older: Job = .{
        .parent = undefined,
        .storage = undefined,
        .inputs = &.{.{ .name = "IN0", .datatype = .int32, .shape = &.{ 1, 4 }, .data = &.{} }},
        .outputs = &.{"OUT0"},
        .samples = 1,
        .accepted_ns = 123,
    };
    var middle: Job = .{
        .parent = undefined,
        .storage = undefined,
        .inputs = &.{.{ .name = "IN0", .datatype = .int32, .shape = &.{ 2, 4 }, .data = &.{} }},
        .outputs = &.{"OUT0"},
        .samples = 2,
        .accepted_ns = 456,
    };
    var full: Job = .{
        .parent = undefined,
        .storage = undefined,
        .inputs = &.{.{ .name = "IN0", .datatype = .int32, .shape = &.{ 8, 4 }, .data = &.{} }},
        .outputs = &.{"OUT0"},
        .samples = 8,
        .accepted_ns = 789,
    };
    state.append(&older);
    state.append(&middle);
    try std.testing.expect(state.buckets == &older);
    try std.testing.expectEqual(@as(usize, 3), older.batch_samples);
    try std.testing.expectEqual(@as(u64, 5_000_123), deadlineFor(older.accepted_ns, state.options.max_wait_us));
    try std.testing.expect(older.batch_next == &middle and state.ready_head == null);
    state.append(&full);
    try std.testing.expect(state.buckets == null);
    try std.testing.expect(state.ready_head == &older and state.ready_tail == &full);
    try std.testing.expect(older.ready_next == &full and full.ready_next == null);
    try std.testing.expectEqual(@as(usize, 8), full.batch_samples);

    state.ready_head = null;
    state.ready_tail = null;
    older.batch_next = null;
    older.accepted_ns = 1000;
    var preempted: Job = .{
        .parent = undefined,
        .storage = undefined,
        .inputs = older.inputs,
        .outputs = older.outputs,
        .samples = 1,
        .accepted_ns = 10,
    };
    state.append(&older);
    state.append(&preempted);
    try std.testing.expectEqual(@as(u64, 10), older.batch_oldest_ns);
    try std.testing.expectEqual(@as(u64, 1000), older.accepted_ns);
    try std.testing.expect(state.ready_head == null);
    // A request arriving at the expired deadline starts a new window rather
    // than being appended to the old batch. No real clock or sleep is involved.
    middle.accepted_ns = 5_000_010;
    middle.batch_next = null;
    state.append(&middle);
    try std.testing.expect(state.ready_head == &older and state.ready_tail == &older);
    try std.testing.expect(state.buckets == &middle);
    try std.testing.expectEqual(@as(usize, 2), older.batch_samples);
}
