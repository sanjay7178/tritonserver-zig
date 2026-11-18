"""Build and time the fixed-width TensorRT workload."""

import ctypes
from contextlib import ExitStack
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter_ns, time_ns

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cuda

SEED = 4050
TOLERANCE = {"atol": 0.0005, "rtol": 0.002}
WORKSPACE = 256 * 1024**2
HTOD = cuda.cudaMemcpyKind.cudaMemcpyHostToDevice
DTOH = cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost
LOGGER = trt.Logger(trt.Logger.WARNING)


def checked(result):
    status, *values = result
    if int(status):
        raise RuntimeError(f"CUDA {status.name}: {status.value}")
    return values[0] if len(values) == 1 else tuple(values)


def owned(stack, create, destroy, *args):
    handle = checked(create(*args))
    stack.callback(lambda: checked(destroy(handle)))
    return handle


def build_plan(width, max_batch, path):
    rng = np.random.default_rng(SEED)
    weights = [
        rng.standard_normal((width, width), dtype=np.float32) / np.float32(width**0.5)
        for _ in range(3)
    ]
    builder = trt.Builder(LOGGER)
    network = builder.create_network()
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE)
    tensor = network.add_input("IN0", trt.float32, (-1, width))
    for i, matrix in enumerate(weights):
        constant = network.add_constant(matrix.shape, matrix).get_output(0)
        tensor = network.add_matrix_multiply(
            tensor, trt.MatrixOperation.NONE, constant, trt.MatrixOperation.NONE
        ).get_output(0)
        if i < len(weights) - 1:
            tensor = network.add_activation(tensor, trt.ActivationType.RELU).get_output(
                0
            )
    tensor.name = "OUT0"
    network.mark_output(tensor)
    profile = builder.create_optimization_profile()
    profile.set_shape("IN0", (1, width), (max_batch, width), (max_batch, width))
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT did not build a serialized plan")
    path.write_bytes(bytes(plan))
    return weights


def fixture(batch, width, weights, path):
    rng = np.random.default_rng(SEED + batch)
    inputs = [rng.standard_normal((batch, width), dtype=np.float32) for _ in range(2)]
    outputs = []
    for reference in inputs:
        for i, matrix in enumerate(weights):
            reference = reference @ matrix
            if i < len(weights) - 1:
                reference = np.maximum(reference, 0)
        outputs.append(reference)
    np.asarray(inputs + outputs, dtype="<f4").tofile(path)


def run_engine(plan, fixture_data, graphs, warmup, iterations):
    inputs, expected = fixture_data
    size = inputs[0].nbytes
    runtime = trt.Runtime(LOGGER)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("could not deserialize TensorRT plan")
    context = engine.create_execution_context()
    if context is None or not context.set_input_shape("IN0", inputs[0].shape):
        raise RuntimeError("could not set TensorRT execution shape")
    if tuple(context.get_tensor_shape("OUT0")) != expected[0].shape:
        raise RuntimeError("unexpected TensorRT output shape")

    with ExitStack() as stack:
        stream = owned(
            stack,
            cuda.cudaStreamCreateWithFlags,
            cuda.cudaStreamDestroy,
            cuda.cudaStreamNonBlocking,
        )
        host_in = owned(stack, cuda.cudaMallocHost, cuda.cudaFreeHost, size)
        host_out = owned(stack, cuda.cudaMallocHost, cuda.cudaFreeHost, size)
        device_in = owned(stack, cuda.cudaMalloc, cuda.cudaFree, size)
        device_out = owned(stack, cuda.cudaMalloc, cuda.cudaFree, size)
        start = owned(stack, cuda.cudaEventCreate, cuda.cudaEventDestroy)
        stop = owned(stack, cuda.cudaEventCreate, cuda.cudaEventDestroy)
        if not (
            context.set_tensor_address("IN0", device_in)
            and context.set_tensor_address("OUT0", device_out)
        ):
            raise RuntimeError("could not set TensorRT device addresses")

        def enqueue():
            if not context.execute_async_v3(int(stream)):
                raise RuntimeError("TensorRT enqueueV3 failed")

        def upload():
            checked(cuda.cudaMemcpyAsync(device_in, host_in, size, HTOD, stream))

        def download():
            checked(cuda.cudaMemcpyAsync(host_out, device_out, size, DTOH, stream))

        def sync():
            checked(cuda.cudaStreamSynchronize(stream))

        ctypes.memmove(host_in, inputs[0].ctypes.data, size)
        upload()
        enqueue()
        sync()
        graph_launches = 0
        if graphs:
            checked(
                cuda.cudaStreamBeginCapture(
                    stream, cuda.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal
                )
            )
            try:
                enqueue()
            finally:
                graph = owned(
                    stack, cuda.cudaStreamEndCapture, cuda.cudaGraphDestroy, stream
                )
            executable = owned(
                stack, cuda.cudaGraphInstantiate, cuda.cudaGraphExecDestroy, graph, 0
            )

            def submit():
                nonlocal graph_launches
                checked(cuda.cudaGraphLaunch(executable, stream))
                graph_launches += 1
        else:
            submit = enqueue

        def verify():
            errors = []
            for value, reference in zip(inputs, expected, strict=True):
                ctypes.memmove(host_in, value.ctypes.data, size)
                upload()
                submit()
                download()
                sync()
                actual = np.ctypeslib.as_array(
                    (ctypes.c_float * value.size).from_address(host_out)
                ).reshape(value.shape)
                np.testing.assert_allclose(
                    actual, reference, equal_nan=False, **TOLERANCE
                )
                errors.append(float(np.max(np.abs(actual - reference))))
            return max(errors)

        before = verify()
        for _ in range(warmup):
            submit()
            sync()

        latency_ns, submit_ns = [], []
        launches_before = graph_launches
        start_unix_ns = time_ns()
        started = perf_counter_ns()
        for _ in range(iterations):
            request_start = perf_counter_ns()
            submit()
            submitted = perf_counter_ns()
            sync()
            latency_ns.append(perf_counter_ns() - request_start)
            submit_ns.append(submitted - request_start)
        elapsed_ns = perf_counter_ns() - started
        end_unix_ns = time_ns()
        timed_launches = graph_launches - launches_before
        after = verify()

        def event_timings(operation):
            times = []
            for _ in range(iterations):
                checked(cuda.cudaEventRecord(start, stream))
                operation()
                checked(cuda.cudaEventRecord(stop, stream))
                sync()
                times.append(checked(cuda.cudaEventElapsedTime(start, stop)) * 1e6)
            return times

        return {
            "elapsed_ns": elapsed_ns,
            "start_unix_ns": start_unix_ns,
            "end_unix_ns": end_unix_ns,
            "max_abs_error": max(before, after),
            "graph_launches_timed": timed_launches,
            "latency_ns": latency_ns,
            "submit_ns": submit_ns,
            "gpu_compute_ns": event_timings(submit),
            "pinned_h2d_ns": event_timings(upload),
            "pinned_d2h_ns": event_timings(download),
        }


def main():
    command, location, width_arg, *args = sys.argv[1:]
    directory = Path(location)
    plan = directory / "model.plan"
    width = int(width_arg)
    if command == "prepare":
        batches = list(map(int, args))
        weights = build_plan(width, max(batches), plan)
        for batch in batches:
            fixture(batch, width, weights, directory / f"batch-{batch}.bin")
        result = {
            "tensorrt": trt.__version__,
            "cuda_bindings": version("cuda-bindings"),
            "cuda_runtime": checked(cuda.cudaRuntimeGetVersion()),
            "cuda_driver_api": checked(cuda.cudaDriverGetVersion()),
            "numpy": np.__version__,
            "engine_config": {
                "layers": ["matmul", "relu", "matmul", "relu", "matmul"],
                "precision": "FP32",
                "tf32": False,
                "width": width,
                "profile_min": [1, width],
                "profile_opt": [max(batches), width],
                "profile_max": [max(batches), width],
                "workspace_bytes": WORKSPACE,
                "seed": SEED,
                "tolerance": TOLERANCE,
            },
        }
    elif command == "measure":
        batch, graphs, warmup, iterations = map(int, args)
        data = np.fromfile(directory / f"batch-{batch}.bin", dtype="<f4")
        x0, x1, y0, y1 = data.reshape((4, batch, width))
        result = run_engine(
            plan.read_bytes(), ([x0, x1], [y0, y1]), bool(graphs), warmup, iterations
        )
    else:
        raise ValueError(f"unknown engine command {command}")
    print("TENSORRT_BENCH_RESULT " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
