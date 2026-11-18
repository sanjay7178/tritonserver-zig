#!/usr/bin/env python3
"""Prepare a TensorRT workload and optionally benchmark it through embedded Zig."""

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import threading
from time import time_ns

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULT_PREFIX = "TENSORRT_BENCH_RESULT "
GPU_FIELDS = ("uuid", "name", "driver_version", "compute_cap", "memory.total")
PROBE_RESULT = re.compile(r"^TensorRT backend cudaGraphLaunch successes: (\d+)$", re.M)


def gpu_query(selector, fields):
    return [
        "nvidia-smi",
        "-i",
        selector,
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]


def gpu_info(selector):
    output = subprocess.check_output(gpu_query(selector, GPU_FIELDS), text=True)
    (row,) = csv.reader(output.splitlines(), skipinitialspace=True)
    result = dict(zip(GPU_FIELDS, row, strict=True))
    result["memory.total"] = int(result["memory.total"])
    return result


def with_samples(device, run):
    fields = ("utilization.gpu", "memory.used")
    samples = []
    baseline = subprocess.check_output(gpu_query(device, fields), text=True)

    def read_samples(pipe):
        for line in pipe:
            samples.append((time_ns(), line))

    with subprocess.Popen(
        gpu_query(device, fields) + ["--loop-ms=50"], stdout=subprocess.PIPE, text=True
    ) as monitor:
        reader = threading.Thread(target=read_samples, args=(monitor.stdout,))
        reader.start()
        try:
            result = run()
        finally:
            if monitor.poll() is None:
                monitor.terminate()
            reader.join()

    def parse(line):
        values = line.strip().split(",")
        if len(values) != 2:
            return None, None
        return tuple(
            int(value) if value.strip().isdigit() else None for value in values
        )

    rows = [
        parse(line)
        for timestamp, line in samples
        if result["start_unix_ns"] <= timestamp <= result["end_unix_ns"]
    ]
    utilization = [row[0] for row in rows if row[0] is not None]
    memory = [row[1] for row in rows if row[1] is not None]
    result["device_telemetry"] = {
        "sample_period_requested_ms": 50,
        "samples": len(rows),
        "utilization_samples": len(utilization),
        "memory_samples": len(memory),
        "gpu_utilization_mean_pct": statistics.mean(utilization)
        if utilization
        else None,
        "gpu_utilization_peak_pct": max(utilization, default=None),
        "memory_baseline_mib": parse(baseline)[1],
        "memory_peak_mib": max(memory, default=None),
    }
    return result


def run_json(command, log_path, env, cwd=None):
    with log_path.open("w") as log:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=log, text=True, env=env, cwd=cwd
        )
    if result.returncode:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-12:])
        raise RuntimeError(
            f"{command[0]} exited {result.returncode}; see {log_path}\n{tail}"
        )
    lines = result.stdout.splitlines()
    records = [
        line.removeprefix(RESULT_PREFIX)
        for line in lines
        if line.startswith(RESULT_PREFIX)
    ]
    if len(records) != 1:
        raise RuntimeError(
            f"expected one benchmark result from {command[0]}; see {log_path}\n{lines[:4]}"
        )
    return json.loads(records[0])


def summarize_ns(values):
    percentiles = statistics.quantiles(values, n=100, method="inclusive")
    return {
        "mean": statistics.mean(values) / 1000,
        **{f"p{p}": percentiles[p - 1] / 1000 for p in (50, 95, 99)},
    }


def summarize_result(raw, batch):
    iterations = len(raw["latency_ns"])
    result = {
        "iterations": iterations,
        "elapsed_s": raw["elapsed_ns"] / 1e9,
        "requests_per_s": iterations * 1e9 / raw["elapsed_ns"],
        "samples_per_s": batch * iterations * 1e9 / raw["elapsed_ns"],
        "max_abs_error": raw["max_abs_error"],
        "device_telemetry": raw["device_telemetry"],
    }
    for phase in ("latency", "submit", "gpu_compute", "pinned_h2d", "pinned_d2h"):
        if f"{phase}_ns" in raw:
            result[f"{phase}_us"] = summarize_ns(raw[f"{phase}_ns"])
    return result


def triton_stats(raw, batch, iterations):
    (before,) = raw["statistics_before"]["model_stats"]
    (after,) = raw["statistics_after"]["model_stats"]
    executions = int(after["execution_count"]) - int(before["execution_count"])
    inferences = int(after["inference_count"]) - int(before["inference_count"])
    if (executions, inferences) != (iterations, iterations * batch):
        raise RuntimeError(f"unexpected Triton counters: {executions=} {inferences=}")
    phases = {}
    for name in (
        "success",
        "queue",
        "compute_input",
        "compute_infer",
        "compute_output",
    ):
        old, new = (snapshot["inference_stats"][name] for snapshot in (before, after))
        count = int(new["count"]) - int(old["count"])
        if count != iterations:
            raise RuntimeError(
                f"unexpected Triton statistics count for {name}: {count}"
            )
        phases[name] = (int(new["ns"]) - int(old["ns"])) / count / 1000
    return {"executions": executions, "samples": inferences, "phase_mean_us": phases}


def model_repository(directory, plan, batch, width, graphs):
    repository = directory / "models"
    model = repository / "bench"
    version = model / "1"
    version.mkdir(parents=True)
    (version / "model.plan").symlink_to(plan)
    graph_config = (
        f"graphs: true graph_spec {{ batch_size: {batch} }}"
        if graphs
        else "graphs: false"
    )
    (model / "config.pbtxt").write_text(f"""name: "bench"
platform: "tensorrt_plan"
max_batch_size: {batch}
input [{{ name: "IN0" data_type: TYPE_FP32 dims: [{width}] }}]
output [{{ name: "OUT0" data_type: TYPE_FP32 dims: [{width}] }}]
instance_group [{{ kind: KIND_GPU count: 1 gpus: [0] }}]
optimization {{ cuda {{ {graph_config} }} }}
""")
    return repository


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triton-prefix", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="new or empty directory"
    )
    parser.add_argument("--gpu", default="0", help="nvidia-smi GPU index or UUID")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="generate the plan, input fixtures, graph probe, and model repositories",
    )
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=20000)
    args = parser.parse_args()
    batches = sorted(set(args.batches))
    if not 1 <= args.width <= 8192 or not 1 <= min(batches) <= max(batches) <= 1024:
        parser.error("width must be in 1..8192 and batch sizes in 1..1024")
    if not args.prepare_only and min(args.warmup, args.iterations) < 2:
        parser.error("warmup and iterations must both be at least 2")
    if sys.platform != "linux" or sys.byteorder != "little":
        parser.error("requires little-endian Linux with NVIDIA CUDA")
    prefix = args.triton_prefix.resolve()
    triton_library = prefix / "lib/libtritonserver.so"
    backend = prefix / "backends/tensorrt/libtriton_tensorrt.so"
    if not triton_library.is_file() or not backend.is_file():
        parser.error(
            "provide a GPU Triton prefix with lib/libtritonserver.so and its TensorRT backend"
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("output directory must be empty")

    gpu = gpu_info(args.gpu)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        part
        for part in (str(triton_library.parent), env.get("LD_LIBRARY_PATH"))
        if part
    )
    env.pop("LD_PRELOAD", None)
    if not args.prepare_only:
        install = output / "zig"
        subprocess.run(
            [
                "zig",
                "build",
                "tensorrt-bench",
                f"-Dtriton-prefix={prefix}",
                "-Doptimize=ReleaseSafe",
                "--prefix",
                install,
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
        client = install / "bin/tensorrt-bench"
    probe = output / "graph-probe.so"
    subprocess.run(
        [
            os.getenv("CC", "cc"),
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-O2",
            "-shared",
            "-fPIC",
            HERE / "graph_probe.c",
            "-ldl",
            "-o",
            probe,
        ],
        check=True,
    )
    worker = [sys.executable, HERE / "engine.py"]
    metadata = run_json(
        worker + ["prepare", output, str(args.width), *map(str, batches)],
        output / "prepare.log",
        env,
    )
    plan = output / "model.plan"
    version_path = prefix / "TRITON_VERSION"
    report = {
        "gpu": {
            "name": gpu["name"],
            "driver_version": gpu["driver_version"],
            "compute_capability": gpu["compute_cap"],
            "memory_total_mib": gpu["memory.total"],
        },
        "kernel": platform.release(),
        "python": platform.python_version(),
        "triton_distribution_version": version_path.read_text().strip()
        if version_path.is_file()
        else None,
        "triton_library_sha256": sha256(triton_library.read_bytes()).hexdigest(),
        "triton_tensorrt_backend_sha256": sha256(backend.read_bytes()).hexdigest(),
        "plan_sha256": sha256(plan.read_bytes()).hexdigest(),
        "plan_size_bytes": plan.stat().st_size,
        "request_batch_sizes": batches,
        **metadata,
        "cases": [],
    }
    if not args.prepare_only:
        report.update(
            zig=subprocess.check_output(["zig", "version"], text=True).strip(),
            warmup_requests=args.warmup,
            requests_in_flight=1,
        )
    for batch in batches:
        for graphs in (False, True):
            case = output / f"b{batch}-graphs-{int(graphs)}"
            repository = model_repository(case, plan, batch, args.width, graphs)
            case_report = {
                "batch_size": batch,
                "graphs_enabled": graphs,
                "tensor_bytes_per_direction": batch * args.width * 4,
            }
            report["cases"].append(case_report)
            if args.prepare_only:
                continue

            zig_command = [
                client,
                repository,
                prefix / "backends",
                "bench",
                output / f"batch-{batch}.bin",
                str(batch),
                str(args.width),
            ]
            log = case / "replay.log"
            proof = run_json(
                zig_command + ["4", "8", "1"],
                log,
                dict(env, LD_PRELOAD="./graph-probe.so"),
                cwd=output,
            )
            matches = PROBE_RESULT.findall(log.read_text(errors="replace"))
            if len(matches) != 1:
                raise RuntimeError(
                    f"missing or ambiguous graph probe result; see {log}"
                )
            launches = int(matches[0])
            expected = 12 if graphs else 0
            if launches != expected:
                raise RuntimeError(
                    f"expected {expected} backend graph launches, observed {launches}; see {log}"
                )
            print(
                f"B{batch} graphs={graphs}: {launches} backend graph launches in proof run",
                flush=True,
            )
            counts = [str(args.warmup), str(args.iterations)]
            engine = with_samples(
                gpu["uuid"],
                lambda: run_json(
                    worker
                    + [
                        "measure",
                        output,
                        str(args.width),
                        str(batch),
                        str(int(graphs)),
                        *counts,
                    ],
                    case / "engine.log",
                    env,
                ),
            )
            if engine["graph_launches_timed"] != (args.iterations if graphs else 0):
                raise RuntimeError("unexpected number of timed engine graph launches")
            zig = with_samples(
                gpu["uuid"],
                lambda: run_json(
                    zig_command + [*counts, "0"],
                    case / "embedded.log",
                    env,
                ),
            )
            case_report.update(
                replay_proof={
                    "inferences": 12,
                    "backend_cudaGraphLaunch_successes": launches,
                    "triton_c_api": proof["triton_c_api"],
                    "max_abs_error": proof["max_abs_error"],
                },
                engine_only={
                    **summarize_result(engine, batch),
                    "graph_launches_timed": engine["graph_launches_timed"],
                },
                embedded_zig={
                    **summarize_result(zig, batch),
                    "triton_statistics": triton_stats(zig, batch, args.iterations),
                },
            )
    result_path = output / "report.json"
    result_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(result_path)


if __name__ == "__main__":
    main()
