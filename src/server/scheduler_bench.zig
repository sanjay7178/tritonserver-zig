const std = @import("std");
const t = @import("tritonserver_zig");
const alloc = std.heap.c_allocator;

const Options = struct {
    repository: []const u8 = "tests/fixtures/scheduler_models",
    backend: []const u8 = ".zig-cache/phase2/backends",
    model: []const u8 = "phase4_fixed",
    version: i64 = 1,
    duration_seconds: u32 = 10,
    warmup_seconds: u32 = 2,
    concurrency: usize = 32,
    workers: usize = 2,
    max_batch_size: u32 = 8,
    max_wait_us: u64 = 5_000,
    workload: ?[]const u8 = null,
    gpu_index: ?u32 = null,
    help: bool = false,
};

fn parse(args: []const []const u8) !Options {
    var options: Options = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const flag = args[i];
        if (std.mem.eql(u8, flag, "--help")) {
            options.help = true;
            continue;
        }
        i += 1;
        if (i == args.len) return error.MissingArgumentValue;
        const value = args[i];
        if (std.mem.eql(u8, flag, "--model-repository")) {
            options.repository = value;
        } else if (std.mem.eql(u8, flag, "--backend-directory")) {
            options.backend = value;
        } else if (std.mem.eql(u8, flag, "--model")) {
            options.model = value;
        } else if (std.mem.eql(u8, flag, "--version")) {
            options.version = try std.fmt.parseInt(i64, value, 10);
        } else if (std.mem.eql(u8, flag, "--duration-seconds")) {
            options.duration_seconds = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--warmup-seconds")) {
            options.warmup_seconds = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--concurrency")) {
            options.concurrency = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, flag, "--workers")) {
            options.workers = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, flag, "--max-batch-size")) {
            options.max_batch_size = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--max-wait-us")) {
            options.max_wait_us = try std.fmt.parseInt(u64, value, 10);
        } else if (std.mem.eql(u8, flag, "--workload")) {
            options.workload = value;
        } else if (std.mem.eql(u8, flag, "--gpu-index")) {
            options.gpu_index = try std.fmt.parseInt(u32, value, 10);
        } else return error.UnknownArgument;
    }
    if (options.duration_seconds == 0 or options.duration_seconds > 3600 or
        options.warmup_seconds > 3600 or options.concurrency == 0 or
        options.concurrency > 1024 or options.workers == 0 or options.workers > 128 or
        options.max_batch_size == 0 or options.max_wait_us > 60_000_000 or
        options.version < -1 or options.version == 0)
        return error.InvalidBenchmarkOption;
    return options;
}

const Workload = struct {
    inputs: []const t.InputTensor,
    outputs: []const []const u8,
};

fn objectField(value: std.json.Value, key: []const u8) !std.json.Value {
    if (value != .object) return error.ExpectedJsonObject;
    return value.object.get(key) orelse error.MissingWorkloadField;
}

fn string(value: std.json.Value) ![]const u8 {
    if (value != .string or value.string.len == 0) return error.ExpectedJsonString;
    return value.string;
}

fn number(comptime T: type, value: std.json.Value) !T {
    return switch (@typeInfo(T)) {
        .int => switch (value) {
            .integer => |v| std.math.cast(T, v) orelse error.WorkloadNumberOutOfRange,
            .number_string => |v| std.fmt.parseInt(T, v, 10),
            else => error.ExpectedJsonInteger,
        },
        .float => blk: {
            const result: T = switch (value) {
                .integer => |v| @floatFromInt(v),
                .float => |v| @floatCast(v),
                .number_string => |v| try std.fmt.parseFloat(T, v),
                else => return error.ExpectedJsonNumber,
            };
            if (!std.math.isFinite(result)) return error.WorkloadNumberOutOfRange;
            break :blk result;
        },
        .bool => if (value == .bool) value.bool else error.ExpectedJsonBoolean,
        else => @compileError("unsupported workload scalar"),
    };
}

fn tensorData(comptime T: type, arena: std.mem.Allocator, value: std.json.Value) ![]const u8 {
    if (value != .array) return error.ExpectedJsonArray;
    const values = try arena.alloc(T, value.array.items.len);
    for (values, value.array.items) |*destination, item| destination.* = try number(T, item);
    return std.mem.sliceAsBytes(values);
}

fn parseWorkload(arena: std.mem.Allocator, io: std.Io, path: ?[]const u8) !Workload {
    if (path == null) {
        const inputs = try arena.alloc(t.InputTensor, 1);
        inputs[0] = try t.InputTensor.fromSlice(i32, "IN0", &.{ 1, 4 }, &.{ 1, 2, 3, 4 });
        return .{ .inputs = inputs, .outputs = &.{"OUT0"} };
    }
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path.?, arena, .limited(16 * 1024 * 1024));
    const parsed = try std.json.parseFromSlice(std.json.Value, arena, bytes, .{ .allocate = .alloc_always, .parse_numbers = false });
    const raw_inputs = try objectField(parsed.value, "inputs");
    if (raw_inputs != .array or raw_inputs.array.items.len == 0) return error.ExpectedInputArray;
    const inputs = try arena.alloc(t.InputTensor, raw_inputs.array.items.len);
    for (inputs, raw_inputs.array.items) |*input, raw| {
        const raw_shape = try objectField(raw, "shape");
        if (raw_shape != .array or raw_shape.array.items.len == 0) return error.InvalidWorkloadShape;
        const shape = try arena.alloc(i64, raw_shape.array.items.len);
        for (shape, raw_shape.array.items) |*dimension, item| {
            dimension.* = try number(i64, item);
            if (dimension.* <= 0) return error.InvalidWorkloadShape;
        }
        // The baseline dispatches one sample. Both cases must use identical work.
        if (shape[0] != 1) return error.WorkloadMustHaveOneSample;
        const datatype_name = try string(try objectField(raw, "datatype"));
        const data = try objectField(raw, "data");
        const Scalar = struct { name: []const u8, datatype: t.DataType, Type: type };
        const scalars = .{
            Scalar{ .name = "BOOL", .datatype = .boolean, .Type = bool },
            Scalar{ .name = "UINT8", .datatype = .uint8, .Type = u8 },
            Scalar{ .name = "UINT16", .datatype = .uint16, .Type = u16 },
            Scalar{ .name = "UINT32", .datatype = .uint32, .Type = u32 },
            Scalar{ .name = "UINT64", .datatype = .uint64, .Type = u64 },
            Scalar{ .name = "INT8", .datatype = .int8, .Type = i8 },
            Scalar{ .name = "INT16", .datatype = .int16, .Type = i16 },
            Scalar{ .name = "INT32", .datatype = .int32, .Type = i32 },
            Scalar{ .name = "INT64", .datatype = .int64, .Type = i64 },
            Scalar{ .name = "FP16", .datatype = .fp16, .Type = f16 },
            Scalar{ .name = "FP32", .datatype = .fp32, .Type = f32 },
            Scalar{ .name = "FP64", .datatype = .fp64, .Type = f64 },
        };
        var found = false;
        inline for (scalars) |scalar| {
            if (std.mem.eql(u8, datatype_name, scalar.name)) {
                input.* = .{
                    .name = try string(try objectField(raw, "name")),
                    .datatype = scalar.datatype,
                    .shape = shape,
                    .data = try tensorData(scalar.Type, arena, data),
                };
                found = true;
            }
        }
        if (!found) return error.UnsupportedWorkloadDataType;
        try input.validate();
    }
    const raw_outputs = try objectField(parsed.value, "outputs");
    if (raw_outputs != .array) return error.ExpectedOutputArray;
    const outputs = try arena.alloc([]const u8, raw_outputs.array.items.len);
    for (outputs, raw_outputs.array.items) |*name, value| name.* = try string(value);
    return .{ .inputs = inputs, .outputs = outputs };
}

fn now(io: std.Io) i96 {
    return std.Io.Timestamp.now(io, .awake).nanoseconds;
}

const Stage = struct {
    io: std.Io,
    scheduler: *t.Scheduler,
    workload: Workload,
    gate: std.atomic.Value(bool) = .init(false),
    stop: std.atomic.Value(bool) = .init(false),
    deadline: i96 = 0,
};

const Client = struct {
    stage: *Stage,
    latencies: std.ArrayList(u64) = .empty,
    failure: ?anyerror = null,
    collect: bool,

    fn run(self: *Client) void {
        while (!self.stage.gate.load(.acquire)) std.Thread.yield() catch {};
        self.perform() catch |err| {
            self.failure = err;
            self.stage.stop.store(true, .release);
        };
    }

    fn perform(self: *Client) !void {
        while (!self.stage.stop.load(.acquire) and now(self.stage.io) < self.stage.deadline) {
            const started = now(self.stage.io);
            var pending = try self.stage.scheduler.submit(alloc, self.stage.workload.inputs, .{
                .outputs = self.stage.workload.outputs,
            });
            defer pending.deinit();
            var response = try pending.wait();
            defer response.deinit();
            if (self.collect) try self.latencies.append(alloc, @intCast(@max(0, now(self.stage.io) - started)));
        }
    }
};

fn gpuUtilization(io: std.Io, index: u32) !f64 {
    var index_buffer: [16]u8 = undefined;
    const index_text = try std.fmt.bufPrint(&index_buffer, "{d}", .{index});
    const result = std.process.run(alloc, io, .{
        .argv = &.{ "nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "--id", index_text },
        .stdout_limit = .limited(4096),
        .stderr_limit = .limited(4096),
        .timeout = .{ .duration = .{ .raw = .fromSeconds(3), .clock = .awake } },
    }) catch |err| {
        std.log.err("GPU utilization query failed: {s}", .{@errorName(err)});
        return error.GpuQueryFailed;
    };
    defer alloc.free(result.stdout);
    defer alloc.free(result.stderr);
    if (result.term != .exited or result.term.exited != 0) {
        std.log.err("nvidia-smi failed: {s}", .{std.mem.trim(u8, result.stderr, " \r\n\t")});
        return error.GpuQueryFailed;
    }
    const utilization = std.fmt.parseFloat(f64, std.mem.trim(u8, result.stdout, " \r\n\t")) catch return error.InvalidGpuUtilization;
    if (!std.math.isFinite(utilization) or utilization < 0 or utilization > 100)
        return error.InvalidGpuUtilization;
    return utilization;
}

const GpuSampler = struct {
    stage: *Stage,
    index: u32,
    total: f64 = 0,
    count: u64 = 0,
    failure: ?anyerror = null,

    fn run(self: *GpuSampler) void {
        while (!self.stage.gate.load(.acquire)) std.Thread.yield() catch {};
        while (!self.stage.stop.load(.acquire) and now(self.stage.io) < self.stage.deadline) {
            const value = gpuUtilization(self.stage.io, self.index) catch |err| {
                self.failure = err;
                self.stage.stop.store(true, .release);
                return;
            };
            self.total += value;
            self.count += 1;
            std.Io.sleep(self.stage.io, .fromMilliseconds(200), .awake) catch |err| {
                self.failure = err;
                self.stage.stop.store(true, .release);
                return;
            };
        }
    }
};

const Measurement = struct {
    seconds: f64,
    latencies: []u64,
    gpu_mean: ?f64,
    gpu_samples: u64,
};

fn stageRun(io: std.Io, scheduler: *t.Scheduler, workload: Workload, options: Options, seconds: u32, collect: bool) !Measurement {
    var stage: Stage = .{ .io = io, .scheduler = scheduler, .workload = workload };
    const clients = try alloc.alloc(Client, options.concurrency);
    defer alloc.free(clients);
    for (clients) |*client| client.* = .{ .stage = &stage, .collect = collect };
    defer for (clients) |*client| client.latencies.deinit(alloc);
    const threads = try alloc.alloc(std.Thread, options.concurrency);
    defer alloc.free(threads);
    var started: usize = 0;
    var sampler: GpuSampler = .{ .stage = &stage, .index = options.gpu_index orelse 0 };
    var gpu_thread: ?std.Thread = null;
    // Release and join every thread even when spawning a later thread fails.
    defer {
        stage.stop.store(true, .release);
        stage.gate.store(true, .release);
        for (threads[0..started]) |thread| thread.join();
        if (gpu_thread) |thread| thread.join();
    }
    for (clients) |*client| {
        threads[started] = try std.Thread.spawn(.{}, Client.run, .{client});
        started += 1;
    }
    if (collect and options.gpu_index != null)
        gpu_thread = try std.Thread.spawn(.{}, GpuSampler.run, .{&sampler});
    const begin = now(io);
    stage.deadline = begin + @as(i96, seconds) * std.time.ns_per_s;
    stage.gate.store(true, .release);
    for (threads[0..started]) |thread| thread.join();
    started = 0;
    const elapsed: f64 = @floatFromInt(@max(1, now(io) - begin));
    stage.stop.store(true, .release);
    if (gpu_thread) |thread| thread.join();
    gpu_thread = null;
    for (clients) |client| if (client.failure) |err| return err;
    if (sampler.failure) |err| return err;
    if (collect and options.gpu_index != null and sampler.count == 0) return error.NoGpuSamples;
    var latencies: std.ArrayList(u64) = .empty;
    errdefer latencies.deinit(alloc);
    for (clients) |client| try latencies.appendSlice(alloc, client.latencies.items);
    return .{
        .seconds = elapsed / std.time.ns_per_s,
        .latencies = try latencies.toOwnedSlice(alloc),
        .gpu_mean = if (sampler.count == 0) null else sampler.total / @as(f64, @floatFromInt(sampler.count)),
        .gpu_samples = sampler.count,
    };
}

const CaseReport = struct {
    max_batch_size: u32,
    max_wait_us: u64,
    measurement_seconds: f64,
    completed_requests: u64,
    completed_samples: u64,
    samples_per_second: f64,
    latency_p50_us: f64,
    latency_p95_us: f64,
    latency_p99_us: f64,
    queue_latency_mean_us: f64,
    mean_batch_samples: f64,
    dispatch_count: u64,
    gpu_utilization_mean_percent: ?f64,
    gpu_utilization_samples: u64,
};

fn percentile(sorted: []const u64, percent: usize) f64 {
    const index = ((sorted.len * percent + 99) / 100) - 1;
    return @as(f64, @floatFromInt(sorted[index])) / std.time.ns_per_us;
}

fn runCase(io: std.Io, server: *const t.Server, workload: Workload, options: Options, batch: u32, wait: u64) !CaseReport {
    var scheduler = try t.Scheduler.init(alloc, io, server, options.model, options.version, .{
        .max_batch_size = batch,
        .max_wait_us = wait,
        .worker_count = options.workers,
        .queue_capacity = @max(1024, options.concurrency),
    });
    defer scheduler.deinit();
    if (options.warmup_seconds != 0) {
        const warmup = try stageRun(io, &scheduler, workload, options, options.warmup_seconds, false);
        alloc.free(warmup.latencies);
    }
    const before = scheduler.stats();
    const measured = try stageRun(io, &scheduler, workload, options, options.duration_seconds, true);
    defer alloc.free(measured.latencies);
    scheduler.shutdown();
    const after = scheduler.stats();
    if (measured.latencies.len == 0) return error.NoCompletedRequests;
    const requests = after.accepted_requests - before.accepted_requests;
    const samples = after.dispatched_samples - before.dispatched_samples;
    const batches = after.dispatched_batches - before.dispatched_batches;
    if (after.failed_requests != before.failed_requests or batches == 0 or requests != measured.latencies.len)
        return error.BenchmarkRequestFailure;
    std.mem.sort(u64, measured.latencies, {}, std.sort.asc(u64));
    return .{
        .max_batch_size = batch,
        .max_wait_us = wait,
        .measurement_seconds = measured.seconds,
        .completed_requests = requests,
        .completed_samples = samples,
        .samples_per_second = @as(f64, @floatFromInt(samples)) / measured.seconds,
        .latency_p50_us = percentile(measured.latencies, 50),
        .latency_p95_us = percentile(measured.latencies, 95),
        .latency_p99_us = percentile(measured.latencies, 99),
        .queue_latency_mean_us = @as(f64, @floatFromInt(after.total_queue_wait_ns - before.total_queue_wait_ns)) /
            @as(f64, @floatFromInt(requests)) / std.time.ns_per_us,
        .mean_batch_samples = @as(f64, @floatFromInt(samples)) / @as(f64, @floatFromInt(batches)),
        .dispatch_count = batches,
        .gpu_utilization_mean_percent = measured.gpu_mean,
        .gpu_utilization_samples = measured.gpu_samples,
    };
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = parse(args[1..]) catch |err| {
        std.log.err("invalid benchmark arguments: {s}; use --help", .{@errorName(err)});
        return err;
    };
    if (options.help) {
        std.debug.print(
            "Usage: scheduler-bench [--duration-seconds 10] [--warmup-seconds 2]\n" ++
                "  --concurrency 32 --workers 2 --max-batch-size 8 --max-wait-us 5000\n" ++
                "  --model-repository PATH --backend-directory PATH --model NAME --version 1\n" ++
                "  --workload FILE.json   Concrete tensors with leading batch dimension 1.\n" ++
                "  --gpu-index INDEX      Sample nvidia-smi; a failed query is an error.\n" ++
                "Compares batch=1/wait=0 with the configured batcher; writes JSON to stdout.\n",
            .{},
        );
        return;
    }
    // A requested but unavailable GPU is an explicit failure, never a CPU result.
    if (options.gpu_index) |index| _ = try gpuUtilization(init.io, index);
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const workload = try parseWorkload(arena.allocator(), init.io, options.workload);
    var repository = try t.Repository.scan(init.gpa, init.io, options.repository);
    defer repository.deinit();
    var server = try t.Server.init(init.gpa, .{
        .model_repository = repository.path,
        .backend_directory = options.backend,
        .log_info = false,
    });
    defer server.deinit() catch |err| std.log.err("server cleanup: {s}", .{@errorName(err)});
    const baseline = try runCase(init.io, &server, workload, options, 1, 0);
    const batched = try runCase(init.io, &server, workload, options, options.max_batch_size, options.max_wait_us);
    try server.stop();
    try server.deinit();
    const ratio = batched.samples_per_second / baseline.samples_per_second;
    const gpu_target: ?bool = if (batched.gpu_utilization_mean_percent) |mean| mean > 80 else null;
    const report = .{
        .model = options.model,
        .version = options.version,
        .concurrency = options.concurrency,
        .workers = options.workers,
        .warmup_seconds_per_case = options.warmup_seconds,
        .requested_measurement_seconds_per_case = options.duration_seconds,
        .latency_scope = "submit to response; includes queue, inference and copies",
        .queue_latency_scope = "enqueue to worker dispatch; excludes inference",
        .gpu_scope = "whole selected device; sampled with nvidia-smi, not process attribution",
        .gpu_index = options.gpu_index,
        .baseline = baseline,
        .batched = batched,
        .throughput_ratio = ratio,
        .throughput_improved = ratio > 1,
        .gpu_utilization_over_80_percent = gpu_target,
        .observed_gpu_thresholds_met = if (gpu_target) |met| @as(?bool, met and ratio > 1) else null,
        // Whole-device activity cannot establish which backend did the work or
        // whether its numerical outputs were correct. Qualify that separately.
        .phase4_performance_target_met = @as(?bool, null),
        .performance_target_qualification_requires = "GPU execution provenance and output correctness in addition to these device-level metrics",
    };
    var buffer: [4096]u8 = undefined;
    var writer = std.Io.File.stdout().writer(init.io, &buffer);
    try std.json.Stringify.value(report, .{ .whitespace = .indent_2 }, &writer.interface);
    try writer.interface.writeByte('\n');
    try writer.interface.flush();
}
