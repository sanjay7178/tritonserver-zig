#!/usr/bin/env python3
"""Exercise the native gRPC listener with the stock Triton client on Linux."""

import argparse
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time

import numpy as np
import tritonclient.grpc as grpc
from tritonclient.utils import InferenceServerException

ROOT = Path(__file__).resolve().parents[2]
MODEL = "phase4_nonbatch"


def wait_for_ready(client, process):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"gRPC server exited with status {process.returncode}")
        try:
            if client.is_server_ready(client_timeout=0.5):
                return
        except InferenceServerException:
            pass
        time.sleep(0.05)
    raise TimeoutError("gRPC server did not become ready")


def check_core_mapping(process, library):
    mappings = []
    for line in Path(f"/proc/{process.pid}/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and int(fields[2], 16) == 0:
            mapped = Path(fields[5])
            if mapped.name.startswith("libtritonserver.so"):
                mappings.append(mapped.resolve())
    if mappings != [library.resolve()]:
        raise RuntimeError(f"expected a single mapped Triton core, found {mappings}")


def make_input(values):
    tensor = grpc.InferInput("IN0", values.shape, "INT32")
    tensor.set_data_from_numpy(values)
    return [tensor]


def exercise(client, process, command, env, directory, library):
    wait_for_ready(client, process)
    check_core_mapping(process, library)
    if not client.is_server_live(client_timeout=2):
        raise RuntimeError("gRPC ServerLive returned false")
    if not client.is_model_ready(MODEL, model_version="1", client_timeout=2):
        raise RuntimeError("gRPC ModelReady returned false for the versioned model")
    if not client.get_server_metadata(client_timeout=2).name:
        raise RuntimeError("gRPC ServerMetadata returned an empty server name")
    metadata = client.get_model_metadata(MODEL, client_timeout=2)
    if metadata.name != MODEL or list(metadata.versions) != ["1"]:
        raise RuntimeError(f"unexpected model metadata: {metadata}")
    if [(i.name, i.datatype, list(i.shape)) for i in metadata.inputs] != [
        ("IN0", "INT32", [4])
    ]:
        raise RuntimeError(f"unexpected model input metadata: {metadata.inputs}")

    values = np.array([19, 0, -32, 701], dtype=np.int32)
    outputs = [grpc.InferRequestedOutput("OUT0")]
    result = client.infer(
        MODEL, make_input(values), model_version="1", outputs=outputs, client_timeout=2
    )
    np.testing.assert_array_equal(result.as_numpy("OUT0"), values)
    try:
        client.infer(MODEL, make_input(values[:3]), outputs=outputs, client_timeout=2)
    except InferenceServerException as exc:
        if exc.status() != "StatusCode.INVALID_ARGUMENT":
            raise
    else:
        raise RuntimeError("gRPC inference accepted a malformed input shape")

    with (directory / "port-conflict.log").open("w") as conflict_log:
        conflict = subprocess.run(
            command, env=env, stdout=conflict_log, stderr=subprocess.STDOUT, timeout=15
        )
    conflict_log = (directory / "port-conflict.log").read_text(errors="replace")
    if conflict.returncode <= 0 or "already in use" not in conflict_log:
        raise RuntimeError(
            f"occupied-port startup did not fail cleanly:\n{conflict_log}"
        )

    # The existing fixture spends one second in ModelInstanceExecute when
    # the first value is INT32_MAX.
    slow_values = values.copy()
    slow_values[0] = np.iinfo(np.int32).max
    responses = []
    completed = threading.Event()

    def callback(result, error):
        responses.append((result, error))
        completed.set()

    client.async_infer(
        MODEL, make_input(slow_values), callback, outputs=outputs, client_timeout=15
    )
    if completed.wait(0.1):
        raise RuntimeError(f"slow fixture completed before SIGTERM: {responses}")
    process.send_signal(signal.SIGTERM)
    if not completed.wait(15):
        raise TimeoutError("in-flight gRPC request did not complete on SIGTERM")
    result, error = responses[0]
    if error is not None:
        raise error
    np.testing.assert_array_equal(result.as_numpy("OUT0"), slow_values)
    if (status := process.wait(timeout=15)) != 0:
        raise RuntimeError(f"gRPC server exited with status {status} after SIGTERM")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, type=Path)
    parser.add_argument("--triton-prefix", required=True, type=Path)
    parser.add_argument(
        "--backend-directory", type=Path, default=ROOT / ".zig-cache/phase2/backends"
    )
    args = parser.parse_args()
    prefix = args.triton_prefix.resolve()
    library = prefix / "lib/libtritonserver.so"
    backend_directory = args.backend_directory.resolve()
    if not library.is_file():
        parser.error(f"missing {library}")
    if not (backend_directory / "phase4_batch/libtriton_phase4_batch.so").is_file():
        parser.error("build the phase4 fixture with zig build scheduler-backend")

    with tempfile.TemporaryDirectory(prefix="triton-grpc-") as directory:
        directory = Path(directory)
        repository = directory / "models"
        model = repository / MODEL
        (model / "1").mkdir(parents=True)
        shutil.copyfile(
            ROOT / f"tests/fixtures/scheduler_models/{MODEL}/config.pbtxt",
            model / "config.pbtxt",
        )
        with socket.socket() as socket_hint:
            socket_hint.bind(("127.0.0.1", 0))
            address = f"127.0.0.1:{socket_hint.getsockname()[1]}"

        command = [
            str(args.server.resolve()),
            "--model-repository",
            str(repository),
            "--backend-directory",
            str(backend_directory),
            "--grpc-listen",
            address,
        ]
        env = {**os.environ}
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            filter(None, (str(library.parent), env.get("LD_LIBRARY_PATH")))
        )
        log_path = directory / "server.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            try:
                with grpc.InferenceServerClient(address) as client:
                    exercise(client, process, command, env, directory, library)
            except Exception as exc:
                logs = log_path.read_text(errors="replace")
                raise RuntimeError(f"native gRPC integration failed:\n{logs}") from exc
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
    print("Native gRPC health, metadata, inference, port conflict and drain passed")


if __name__ == "__main__":
    main()
