const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const custom_prefix = b.option([]const u8, "triton-prefix", "Triton installation with matching headers, libraries and backend utilities");
    const triton_prefix = custom_prefix orelse "core/build/install";
    const prefix_path = if (std.fs.path.isAbsolute(triton_prefix)) triton_prefix else b.root.joinString(b.allocator, triton_prefix) catch @panic("OOM");

    // This Zig toolchain uses translate-c in place of @cImport. Translate the
    // upstream header so the entire C API stays available through t.c.
    const Translator = @import("translate_c").Translator;
    const translated: Translator = .init(b.dependency("translate_c", .{}), .{
        .c_source_file = if (custom_prefix != null) .{ .cwd_relative = b.pathJoin(&.{ prefix_path, "include/triton/core/tritonserver.h" }) } else b.path("core/include/triton/core/tritonserver.h"),
        .target = target,
        .optimize = optimize,
    });

    const mod = b.addModule("tritonserver_zig", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mod.addImport("translated", translated.mod);
    // Link on the bindings module, so consumers and tests inherit the library.
    mod.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ prefix_path, "lib" }) });
    mod.addRPath(.{ .cwd_relative = b.pathJoin(&.{ prefix_path, "lib" }) });
    mod.linkSystemLibrary("tritonserver", .{});

    const exe = b.addExecutable(.{
        .name = "tritonserver_zig",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/server/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "tritonserver_zig", .module = mod }},
        }),
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    b.step("run", "Run the starter until SIGINT or SIGTERM").dependOn(&run_cmd.step);

    const smoke_cmd = b.addRunArtifact(exe);
    smoke_cmd.addArg("--once");
    b.step("smoke", "Start Triton, check liveness, stop and release resources").dependOn(&smoke_cmd.step);

    const demo_backend = b.addSystemCommand(&.{ "sh", "tools/build/build_demo_backend.sh" });
    demo_backend.setEnvironmentVariable("TRITON_INSTALL_DIR", prefix_path);
    b.step("demo-backend", "Build the upstream minimal CPU backend").dependOn(&demo_backend.step);
    const demo_cmd = b.addRunArtifact(exe);
    demo_cmd.step.dependOn(&demo_backend.step);
    demo_cmd.addArgs(&.{ "--once", "--list-models", "--model-repository", "tests/fixtures/models", "--backend-directory", ".zig-cache/phase2/backends" });
    b.step("demo", "Load CPU models and print tensor metadata and batch limits").dependOn(&demo_cmd.step);

    const infer_cmd = b.addRunArtifact(exe);
    infer_cmd.step.dependOn(&demo_backend.step);
    infer_cmd.addArgs(&.{ "--once", "--infer-demo", "--model-repository", "tests/fixtures/models", "--backend-directory", ".zig-cache/phase2/backends" });
    b.step("infer-demo", "Run real inference and verify typed CPU outputs").dependOn(&infer_cmd.step);

    const scheduler_backend = b.addSystemCommand(&.{ "sh", "tools/build/build_scheduler_backend.sh" });
    scheduler_backend.setEnvironmentVariable("TRITON_INSTALL_DIR", prefix_path);
    b.step("scheduler-backend", "Build the instrumented CPU batching backend").dependOn(&scheduler_backend.step);
    for ([_][]const u8{ "scheduler-demo", "scheduler-bench" }) |name| {
        const scheduler_exe = b.addExecutable(.{
            .name = name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(if (std.mem.eql(u8, name, "scheduler-demo")) "src/server/scheduler_demo.zig" else "src/server/scheduler_bench.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{.{ .name = "tritonserver_zig", .module = mod }},
            }),
        });
        b.installArtifact(scheduler_exe);
        const command = b.addRunArtifact(scheduler_exe);
        command.step.dependOn(&scheduler_backend.step);
        command.addPassthruArgs();
        b.step(name, if (std.mem.eql(u8, name, "scheduler-demo")) "Verify Zig request merging and output splitting" else "Compare single-request and dynamically batched inference").dependOn(&command.step);
    }

    // Optional hardware experiment, outside the default build and CI.
    const gpu_verify = b.addExecutable(.{
        .name = "gpu-verify",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/experiments/gpu_batch/verify.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "tritonserver_zig", .module = mod }},
        }),
    });
    const gpu_verify_cmd = b.addRunArtifact(gpu_verify);
    gpu_verify_cmd.addPassthruArgs();
    b.step("gpu-verify", "Check GPU experiment outputs and Zig batch partitioning").dependOn(&gpu_verify_cmd.step);

    const test_step = b.step("test", "Test bindings, server wrappers and real CPU models");
    const mod_tests = b.addTest(.{ .root_module = mod });
    const run_mod_tests = b.addRunArtifact(mod_tests);
    run_mod_tests.step.dependOn(&demo_backend.step);
    run_mod_tests.step.dependOn(&scheduler_backend.step);
    test_step.dependOn(&run_mod_tests.step);
    const exe_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("src/server/command_line.zig"),
        .target = target,
        .optimize = optimize,
    }) });
    test_step.dependOn(&b.addRunArtifact(exe_tests).step);
}
