#!/usr/bin/env python3
"""Generate a local CUDA workload; generation never creates a GPU session.

Use the Python 3.12 environment containing NumPy and onnxruntime-gpu. Missing
ONNX generator packages can be installed only into the experiment directory by
passing --install-deps. All generated models, dependencies and profiles stay
under .zig-cache/gpu-experiment unless --output is supplied.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import shlex
import site
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ONNX_VERSION = "1.20.1"
ML_DTYPES_VERSION = "0.5.4"

# The generated runtime is shared by direct verification and the Triton model.
# Profiling ends after one correctness/provider check, before measured requests.
RUNTIME_SOURCE = '''"""Strict CUDA execution and profiling check for the generated experiment."""
import json
import sys
from collections import Counter
from pathlib import Path


def load_modules(config):
    if list(sys.version_info[:2]) != config["python_version"]:
        raise RuntimeError("Python backend stub and package Python versions must match")
    for path in reversed(config["python_site_packages"]):
        if path not in sys.path:
            sys.path.insert(0, path)
    import numpy as np
    import onnxruntime as ort
    return np, ort


def verified_session(model_path, config, profile_prefix, batch_size=1):
    np, ort = load_modules(config)
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("onnxruntime-gpu CUDAExecutionProvider is unavailable")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    # Preserve the explicit MatMul/Relu nodes for auditable provider evidence.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.enable_profiling = True
    options.profile_file_prefix = str(profile_prefix)
    session = ort.InferenceSession(
        str(model_path), sess_options=options,
        providers=[("CUDAExecutionProvider", {
            "device_id": str(config["gpu_index"]),
            "use_tf32": "0",
        })],
    )
    session.disable_fallback()
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("ORT silently replaced CUDAExecutionProvider")
    values = np.ones((batch_size, config["width"]), dtype=np.float32)
    result = session.run(["OUT0"], {"IN0": values})[0]
    np.testing.assert_allclose(result, values, rtol=3e-4, atol=3e-4)
    profile = Path(session.end_profiling())
    events = json.loads(profile.read_text())
    kernels = [event for event in events if event.get("cat") == "Node"
               and event.get("name", "").endswith("_kernel_time")]
    if not kernels:
        raise RuntimeError("ORT profile contains no execution kernels")
    wrong = [event.get("name") for event in kernels
             if event.get("args", {}).get("provider") != "CUDAExecutionProvider"]
    if wrong:
        raise RuntimeError("CPU/unverified execution in ORT profile: " + repr(wrong))
    counts = Counter(event.get("args", {}).get("op_name") for event in kernels)
    for op in ("MatMul", "Relu"):
        if counts[op] != config["layers"]:
            raise RuntimeError("Expected all explicit " + op + " nodes in profile: " + repr(counts))
    evidence = {
        "onnxruntime_version": ort.__version__,
        "provider": "CUDAExecutionProvider",
        "cpu_fallback_disabled": True,
        "use_tf32": False,
        "batch_size": batch_size,
        "output_shape": list(result.shape),
        "ones_output_verified": True,
        "max_absolute_error": float(np.max(np.abs(result - values))),
        "kernel_counts": dict(counts),
        "profile": str(profile),
    }
    profile.with_suffix(".summary.json").write_text(json.dumps(evidence, indent=2) + "\\n")
    return session, np, evidence
'''

MODEL_SOURCE = '''"""CPU-transport Triton Python backend; model computation must execute on CUDA."""
import json
import os
import sys
from pathlib import Path

VERSION_DIRECTORY = Path(__file__).resolve().parent
RUNTIME_CONFIG = json.loads((VERSION_DIRECTORY / "runtime_config.json").read_text())
for package_path in reversed(RUNTIME_CONFIG["python_site_packages"]):
    if package_path not in sys.path:
        sys.path.insert(0, package_path)
sys.path.insert(0, str(VERSION_DIRECTORY))
from gpu_runtime import verified_session
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.config = RUNTIME_CONFIG
        profile = Path(self.config["profile_directory"]) / ("triton-" + str(os.getpid()))
        self.session, self.np, evidence = verified_session(
            VERSION_DIRECTORY / "model.onnx", self.config, profile)
        self.executions = 0
        self.max_batch_seen = 0
        pb_utils.Logger.log_info("GPU_MLP_CUDA_VERIFIED " + json.dumps(evidence))

    def execute(self, requests):
        # With native dynamic batching disabled, Zig must pass one concatenated
        # C request. A request list larger than one would invalidate this proof.
        if len(requests) != 1:
            error = pb_utils.TritonError("Expected one explicitly batched request; native batching is forbidden")
            return [pb_utils.InferenceResponse(error=error) for _ in requests]
        try:
            values = pb_utils.get_input_tensor_by_name(requests[0], "IN0").as_numpy()
            if (values.dtype != self.np.float32 or values.ndim != 2
                    or values.shape[1] != self.config["width"]
                    or not 1 <= values.shape[0] <= self.config["max_batch_size"]):
                raise ValueError("Expected FP32 [B,width] within configured batch limit")
            values = self.np.ascontiguousarray(values)
            result = self.session.run(["OUT0"], {"IN0": values})[0]
            self.executions += 1
            self.max_batch_seen = max(self.max_batch_seen, values.shape[0])
            return [pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("OUT0", result)])]
        except Exception as error:
            return [pb_utils.InferenceResponse(error=pb_utils.TritonError(str(error)))]

    def finalize(self):
        pb_utils.Logger.log_info("GPU_MLP_EXECUTIONS " + json.dumps({
            "executions": self.executions, "max_batch_seen": self.max_batch_seen,
            "native_request_count": 1,
        }))
'''


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def generator_modules(packages: Path, install: bool):
    sys.path.insert(0, str(packages))
    try:
        import numpy as np
        import onnx
    except ImportError as error:
        if not install:
            raise SystemExit("Generator dependency missing; rerun with --install-deps: " + str(error))
        packages.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--target", str(packages),
            "--no-deps", "onnx==" + ONNX_VERSION, "ml_dtypes==" + ML_DTYPES_VERSION,
        ], check=True)
        importlib.invalidate_caches()
        import numpy as np
        import onnx
    return np, onnx


def generate(options, output: Path) -> None:
    np, onnx = generator_modules(output / "python-packages", options.install_deps)
    import onnxruntime as ort
    version = output / "models" / "gpu_mlp" / "1"
    version.mkdir(parents=True, exist_ok=True)
    profiles = output / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    weight_size = options.width * options.width * 4
    # Stream one shared FP32 matrix rather than retaining several 268MB protobuf
    # copies in the generator. Four MatMul nodes reference the same initializer.
    rows = min(options.width, 512)
    block = np.full((rows, options.width), 1.0 / options.width, dtype=np.float32)
    with (version / "weights.bin").open("wb") as weights:
        for begin in range(0, options.width, rows):
            weights.write(block[: min(rows, options.width - begin)].tobytes())
    weight = onnx.TensorProto()
    weight.name = "W"
    weight.data_type = onnx.TensorProto.FLOAT
    weight.dims.extend([options.width, options.width])
    weight.data_location = onnx.TensorProto.EXTERNAL
    for key, value in {"location": "weights.bin", "offset": "0", "length": str(weight_size)}.items():
        entry = weight.external_data.add()
        entry.key, entry.value = key, value
    nodes = []
    previous = "IN0"
    for index in range(options.layers):
        product = "PRODUCT_" + str(index)
        current = "OUT0" if index == options.layers - 1 else "ACTIVATION_" + str(index)
        nodes.append(onnx.helper.make_node("MatMul", [previous, "W"], [product], name="MatMul_" + str(index)))
        nodes.append(onnx.helper.make_node("Relu", [product], [current], name="Relu_" + str(index)))
        previous = current
    graph = onnx.helper.make_graph(
        nodes, "shared_weight_dense_mlp",
        [onnx.helper.make_tensor_value_info("IN0", onnx.TensorProto.FLOAT, ["B", options.width])],
        [onnx.helper.make_tensor_value_info("OUT0", onnx.TensorProto.FLOAT, ["B", options.width])],
        initializer=[weight],
    )
    model = onnx.helper.make_model(graph, producer_name="tritonserver-zig-gpu-experiment",
                                   opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save_model(model, str(version / "model.onnx"))
    onnx.checker.check_model(str(version / "model.onnx"))
    package_paths = [str((output / "python-packages").resolve())]
    package_paths.extend(options.python_site_packages or site.getsitepackages())
    config = {
        "width": options.width,
        "layers": options.layers,
        "max_batch_size": options.max_batch_size,
        "gpu_index": options.gpu_index,
        "python_version": list(sys.version_info[:2]),
        "python_site_packages": package_paths,
        "profile_directory": str(profiles),
    }
    write_json(version / "runtime_config.json", config)
    (version / "gpu_runtime.py").write_text(RUNTIME_SOURCE)
    (version / "model.py").write_text(MODEL_SOURCE)
    (version.parent / "config.pbtxt").write_text(f'''name: "gpu_mlp"
backend: "python"
max_batch_size: {options.max_batch_size}
# Native dynamic_batching is deliberately absent. ORT explicitly selects CUDA.
input [{{ name: "IN0" data_type: TYPE_FP32 dims: [ {options.width} ] }}]
output [{{ name: "OUT0" data_type: TYPE_FP32 dims: [ {options.width} ] }}]
# CPU transport is compatible with a GPU-disabled Triton core; model.py executes
# every MatMul/Relu in ORT CUDA, verified before measured inference begins.
instance_group [{{ kind: KIND_CPU count: 1 }}]
''')
    write_json(output / "workload.json", {
        "inputs": [{"name": "IN0", "datatype": "FP32", "shape": [1, options.width],
                    "data": [1.0] * options.width}],
        "outputs": ["OUT0"],
    })
    write_json(output / "manifest.json", {
        "generator_python": sys.version,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "numpy_version": np.__version__,
        "opset": 17, "ir_version": 10,
        "width": options.width, "layers": options.layers,
        "max_batch_size": options.max_batch_size,
        "shared_weight_bytes": weight_size,
        "model": str(version / "model.onnx"),
        "native_dynamic_batching": False,
        "transport": "CPU", "compute": "CUDAExecutionProvider",
        "gpu_verified": False,
    })
    print(f"Generated {options.layers} MatMul+Relu layers, width {options.width}, "
          f"one shared {weight_size:,}-byte matrix. GPU execution has not run.")


def verify(output: Path, batch_size: int) -> None:
    version = output / "models" / "gpu_mlp" / "1"
    config = json.loads((version / "runtime_config.json").read_text())
    if batch_size > config["max_batch_size"]:
        raise SystemExit("Verification batch exceeds model max_batch_size")
    spec = importlib.util.spec_from_file_location("experiment_gpu_runtime", version / "gpu_runtime.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    _, _, evidence = runtime.verified_session(
        version / "model.onnx", config, output / "profiles" / "direct", batch_size)
    write_json(output / "cuda-verification.json", evidence)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["gpu_verified"] = True
    manifest["cuda_verification"] = str(output / "cuda-verification.json")
    write_json(manifest_path, manifest)
    print(json.dumps(evidence, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".zig-cache/gpu-experiment")
    parser.add_argument("--width", type=int, default=8192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--python-site-packages", action="append")
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--verify-only", action="store_true",
                        help="Run real CUDA inference and inspect its profile; do not regenerate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch for --verify-only")
    options = parser.parse_args()
    if min(options.width, options.layers, options.max_batch_size, options.batch_size) < 1 or options.gpu_index < 0:
        parser.error("width/layers/batch sizes must be positive and GPU index nonnegative")
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if options.verify_only:
        verify(output, options.batch_size)
        return
    generate(options, output)
    print("\nDirect GPU verification (runs GPU computation):")
    print(shlex.join([sys.executable, str(Path(__file__).resolve()), "--output", str(output),
                      "--verify-only", "--batch-size", "1"]))
    print("\nRelease benchmark (baseline batch1 versus configured Zig batching):")
    print(shlex.join(["zig", "build", "scheduler-bench", "-Doptimize=ReleaseFast", "--",
                      "--model-repository", str(output / "models"),
                      "--backend-directory", str(ROOT / "core/build/install/backends"),
                      "--model", "gpu_mlp", "--workload", str(output / "workload.json"),
                      "--max-batch-size", str(min(32, options.max_batch_size)),
                      "--max-wait-us", "5000", "--workers", "2", "--concurrency", "64",
                      "--gpu-index", str(options.gpu_index)]))


if __name__ == "__main__":
    main()
