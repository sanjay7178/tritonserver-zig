#!/usr/bin/env python3
"""Compare Zig and stock Triton loopback gRPC with the same TensorRT artifacts."""

import argparse
from contextlib import contextmanager
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
from time import monotonic, perf_counter_ns, sleep, time_ns

import numpy as np
import tritonclient.grpc as grpc
from tritonclient.utils import InferenceServerException

from benchmark import (
    PROBE_RESULT,
    gpu_info,
    summarize_result,
    triton_stats,
    with_samples,
)

MODEL = "bench"
MODEL_VERSION = "1"


@contextmanager
def server_client(server, kind, repository, prefix, env, log_path):
    with socket.socket() as reserved_port:
        reserved_port.bind(("127.0.0.1", 0))
        port = reserved_port.getsockname()[1]
    address = f"127.0.0.1:{port}"
    command = [
        str(server),
        "--model-repository",
        str(repository),
        "--backend-directory",
        str(prefix / "backends"),
    ]
    if kind == "zig":
        command += ["--grpc-listen", address]
    else:
        command += [
            "--allow-grpc=true",
            "--allow-http=false",
            "--allow-metrics=false",
            "--allow-client-shm=false",
            "--grpc-address=127.0.0.1",
            f"--grpc-port={port}",
            "--model-control-mode=poll",
            "--repository-poll-secs=0",
            "--disable-auto-complete-config",
        ]
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            with grpc.InferenceServerClient(address) as client:
                deadline = monotonic() + 60
                while monotonic() < deadline and process.poll() is None:
                    try:
                        ready = client.is_model_ready(
                            MODEL, model_version=MODEL_VERSION, client_timeout=0.5
                        )
                    except InferenceServerException:
                        ready = False
                    if ready:
                        break
                    sleep(0.05)
                else:
                    raise RuntimeError(f"{kind} gRPC model did not become ready")
                yield client
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    if process.returncode != 0:
        raise RuntimeError(
            f"{kind} server exited with {process.returncode}, see {log_path}"
        )


def infer(client, values):
    tensor = grpc.InferInput("IN0", values.shape, "FP32")
    tensor.set_data_from_numpy(values)
    result = client.infer(
        MODEL,
        [tensor],
        model_version=MODEL_VERSION,
        outputs=[grpc.InferRequestedOutput("OUT0")],
        client_timeout=15,
    )
    return result.as_numpy("OUT0")


def check_output(actual, expected, tolerance):
    if actual is None or actual.shape != expected.shape or actual.dtype != np.float32:
        raise RuntimeError("gRPC response has an unexpected shape or datatype")
    np.testing.assert_allclose(actual, expected, equal_nan=False, **tolerance)
    return float(np.max(np.abs(actual - expected)))


def replay_proof(
    server, kind, repository, prefix, env, log, fixture, graphs, tolerance
):
    inputs, expected = fixture
    count = 12
    max_error = 0.0
    with server_client(server, kind, repository, prefix, env, log) as client:
        for index in range(count):
            variant = index & 1
            actual = infer(client, inputs[variant])
            max_error = max(
                max_error, check_output(actual, expected[variant], tolerance)
            )
    matches = PROBE_RESULT.findall(log.read_text(errors="replace"))
    expected_launches = count if graphs else 0
    if matches != [str(expected_launches)]:
        raise RuntimeError(
            f"expected {expected_launches} backend graph launches, found {matches}, see {log}"
        )
    return {
        "requests": count,
        "backend_cudaGraphLaunch_successes": expected_launches,
        "max_abs_error": max_error,
    }


def run_timed(client, fixture, counts, tolerance):
    inputs, expected = fixture
    warmup, iterations = counts
    max_error = 0.0
    for index in range(warmup):
        variant = index & 1
        actual = infer(client, inputs[variant])
        if index < 2:
            max_error = max(
                max_error, check_output(actual, expected[variant], tolerance)
            )

    def stats():
        return client.get_inference_statistics(
            MODEL, model_version=MODEL_VERSION, as_json=True, client_timeout=5
        )

    before = stats()
    times_ns = []
    start_unix_ns = time_ns()
    start = perf_counter_ns()
    for index in range(iterations):
        variant = index & 1
        request_start = perf_counter_ns()
        actual = infer(client, inputs[variant])
        times_ns.append(perf_counter_ns() - request_start)
        if index >= iterations - 2:
            max_error = max(
                max_error, check_output(actual, expected[variant], tolerance)
            )
    elapsed_ns = perf_counter_ns() - start
    end_unix_ns = time_ns()
    return {
        "latency_ns": times_ns,
        "elapsed_ns": elapsed_ns,
        "start_unix_ns": start_unix_ns,
        "end_unix_ns": end_unix_ns,
        "max_abs_error": max_error,
        "statistics_before": before,
        "statistics_after": stats(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zig-server", type=Path, required=True)
    parser.add_argument("--stock-server", type=Path, required=True)
    parser.add_argument("--triton-prefix", type=Path, required=True)
    parser.add_argument(
        "--artifacts", type=Path, required=True, help="benchmark.py output directory"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="new or empty directory"
    )
    parser.add_argument("--gpu", default="0", help="nvidia-smi GPU index or UUID")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=20000)
    args = parser.parse_args()
    if args.warmup < 2 or args.iterations < 2:
        parser.error("warmup and iterations must both be at least 2")

    artifacts = args.artifacts.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    prefix = args.triton_prefix.resolve()
    library = prefix / "lib/libtritonserver.so"
    backend = prefix / "backends/tensorrt/libtriton_tensorrt.so"
    probe = artifacts / "graph-probe.so"
    original = json.loads((artifacts / "report.json").read_text())
    # Compare the actual runtime, backend, and plan with the in-process report.
    expected_hashes = {
        "triton_library_sha256": library,
        "triton_tensorrt_backend_sha256": backend,
        "plan_sha256": artifacts / "model.plan",
    }
    for key, path in expected_hashes.items():
        if sha256(path.read_bytes()).hexdigest() != original[key]:
            parser.error(f"{path} differs from the source benchmark")
    if not probe.is_file():
        parser.error(f"missing graph launch probe {probe}")

    gpu = gpu_info(args.gpu)
    if gpu["name"] != original["gpu"]["name"]:
        parser.error("use the GPU that ran the in-process benchmark")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(None, (str(library.parent), env.get("LD_LIBRARY_PATH")))
    )
    env.pop("LD_PRELOAD", None)
    proof_env = dict(env, LD_PRELOAD=str(probe))
    report = {
        "gpu": {key: value for key, value in gpu.items() if key != "uuid"},
        "triton_distribution_version": original["triton_distribution_version"],
        "tritonclient": version("tritonclient"),
        "grpcio": version("grpcio"),
        "python": platform.python_version(),
        "plan_sha256": original["plan_sha256"],
        "triton_library_sha256": original["triton_library_sha256"],
        "triton_tensorrt_backend_sha256": original["triton_tensorrt_backend_sha256"],
        "embedded_warmup_requests": original["warmup_requests"],
        "warmup_requests": args.warmup,
        "requests_in_flight": 1,
        "cases": [],
    }
    config = original["engine_config"]
    counts = (args.warmup, args.iterations)
    for case in original["cases"]:
        batch, graphs = case["batch_size"], case["graphs_enabled"]
        name = f"b{batch}-graphs-{int(graphs)}"
        repository = artifacts / name / "models"
        mounted_plan = repository / MODEL / MODEL_VERSION / "model.plan"
        if mounted_plan.resolve() != (artifacts / "model.plan").resolve():
            parser.error(f"{mounted_plan} does not point to the verified plan")
        data = np.fromfile(artifacts / f"batch-{batch}.bin", dtype="<f4")
        data = data.reshape(4, batch, config["width"])
        fixture = (data[:2], data[2:])
        result = {
            "batch_size": batch,
            "graphs_enabled": graphs,
            "embedded_zig": case["embedded_zig"],
        }
        for kind, server in (("zig", args.zig_server), ("stock", args.stock_server)):
            server = server.resolve()
            proof_log = output / f"{name}-{kind}-proof.log"
            timed_log = output / f"{name}-{kind}-timed.log"
            proof = replay_proof(
                server,
                kind,
                repository,
                prefix,
                proof_env,
                proof_log,
                fixture,
                graphs,
                config["tolerance"],
            )
            with server_client(
                server, kind, repository, prefix, env, timed_log
            ) as client:
                raw = with_samples(
                    gpu["uuid"],
                    lambda: run_timed(client, fixture, counts, config["tolerance"]),
                )
            result[f"{kind}_grpc"] = {
                **summarize_result(raw, batch),
                "triton_statistics": triton_stats(raw, batch, args.iterations),
                "replay_proof": proof,
            }
            print(f"{name} {kind} gRPC verified", flush=True)
        report["cases"].append(result)
    path = output / "grpc-report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
