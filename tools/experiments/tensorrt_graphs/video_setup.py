#!/usr/bin/env python3
"""Prepare the pinned YOLOv8n and crop regression workload for Triton."""

import argparse
from hashlib import file_digest
import json
import os
from pathlib import Path
import subprocess
from typing import NamedTuple

import numpy as np
import onnx
from onnx import numpy_helper
import tensorrt as trt
from cuda.bindings import runtime as cuda

from benchmark import HERE, gpu_info
from engine import LOGGER, WORKSPACE, checked

MODEL_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx"
)
MODEL_SHA256 = "b2bc52f40e8e1c532427d5bde3575a5d5b571b739fab2c6df443733ed1589cbd"
VIDEO_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    "ground-vehicles-airport.mp4"
)
VIDEO_SHA256 = "6ebe37d3e0207184511f03777000d425d041dc6995d9843fa45aea7d91d54cca"
BATCHES = (1, 8)
PROFILE_BATCHES = {"min": BATCHES[0], "opt": BATCHES[-1], "max": BATCHES[-1]}


class Model(NamedTuple):
    input_name: str
    input_shape: tuple[int, ...]
    output_name: str
    output_shape: tuple[int, ...]


MODELS = {
    "yolov8n": Model("images", (3, 640, 640), "output0", (84, 8400)),
    "crop_regression": Model("crops", (4, 3, 64, 64), "offsets", (4, 4)),
}

# This pinned export has four shape constants that embed batch 1 in the head.
YOLO_RESHAPES = {
    "/model.22/Constant_output_0": (1, 64, -1),
    "/model.22/Constant_3_output_0": (1, 80, -1),
    "/model.22/dfl/Constant_output_0": (1, 4, 16, 8400),
    "/model.22/dfl/Constant_1_output_0": (1, 4, 8400),
}
# Input rows are RGB crop means and outputs are the four normalized box deltas.
REGRESSION_WEIGHTS = (
    (0.006, 0.0, -0.006, 0.0),
    (0.0, 0.006, 0.0, -0.006),
    (-0.003, 0.003, 0.003, -0.003),
)


def digest(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def download(url, checksum, path):
    subprocess.run(
        ["curl", "-fsSL", "--retry", "3", "--max-time", "120", url, "-o", path],
        check=True,
    )
    if digest(path) != checksum:
        raise RuntimeError(f"download checksum mismatch for {path}")


def make_batched_onnx(source, target):
    model = onnx.load(source, load_external_data=False)
    constants = {tensor.name: tensor for tensor in model.graph.initializer}
    for name, shape in YOLO_RESHAPES.items():
        tensor = constants.get(name)
        if tensor is None or tuple(numpy_helper.to_array(tensor)) != shape:
            raise RuntimeError(f"unexpected YOLOv8n reshape constant {name}")
        # ONNX Reshape uses 0 to copy an input axis of the same index.
        tensor.CopyFrom(
            numpy_helper.from_array(np.array((0, *shape[1:]), np.int64), name=name)
        )
    for tensor in (*model.graph.input, *model.graph.output):
        axis = tensor.type.tensor_type.shape.dim[0]
        if axis.dim_value != 1:
            raise RuntimeError(f"unexpected YOLOv8n batch size in {tensor.name}")
        axis.dim_param = "batch"
    # Recompute intermediate shapes without the source export's fixed batch annotations.
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    onnx.save(model, target)


def build_plan(name, output, source):
    spec = MODELS[name]
    builder = trt.Builder(LOGGER)
    network = builder.create_network()
    if source is not None:
        parser = trt.OnnxParser(network, LOGGER)
        if not parser.parse_from_file(str(source)):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("TensorRT could not parse YOLOv8n\n" + "\n".join(errors))
    else:
        crops = network.add_input(spec.input_name, trt.float32, (-1, *spec.input_shape))
        means = network.add_reduce(
            crops, trt.ReduceOperation.AVG, (1 << 3) | (1 << 4), False
        ).get_output(0)
        weights = np.asarray([REGRESSION_WEIGHTS], dtype=np.float32)
        affine = network.add_constant(weights.shape, weights).get_output(0)
        offsets = network.add_matrix_multiply(
            means, trt.MatrixOperation.NONE, affine, trt.MatrixOperation.NONE
        ).get_output(0)
        offsets.name = spec.output_name
        network.mark_output(offsets)

    if network.num_inputs != 1 or network.num_outputs != 1:
        raise RuntimeError(f"unexpected tensor count for {name}")
    for tensor, expected_name, expected_shape in (
        (network.get_input(0), spec.input_name, spec.input_shape),
        (network.get_output(0), spec.output_name, spec.output_shape),
    ):
        if (tensor.name, tuple(tensor.shape), tensor.dtype) != (
            expected_name,
            (-1, *expected_shape),
            trt.float32,
        ):
            raise RuntimeError(f"unexpected {name} tensor {tensor.name} {tensor.shape}")

    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE)
    profile = builder.create_optimization_profile()
    profile_shapes = [(batch, *spec.input_shape) for batch in PROFILE_BATCHES.values()]
    profile.set_shape(spec.input_name, *profile_shapes)
    if config.add_optimization_profile(profile) < 0:
        raise RuntimeError(f"invalid optimization profile for {name}")
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"TensorRT did not build a plan for {name}")
    output.write_bytes(bytes(plan))


def write_repository(output, batch, graphs):
    models = output / f"b{batch}-graphs-{int(graphs)}" / "models"
    cuda_config = (
        f"graphs: true graph_spec {{ batch_size: {batch} }}"
        if graphs
        else "graphs: false"
    )
    for name, spec in MODELS.items():
        model = models / name
        version_dir = model / "1"
        version_dir.mkdir(parents=True)
        plan = output / f"{name}.plan"
        (version_dir / "model.plan").symlink_to(os.path.relpath(plan, version_dir))
        input_shape = ", ".join(map(str, spec.input_shape))
        output_shape = ", ".join(map(str, spec.output_shape))
        (model / "config.pbtxt").write_text(f'''name: "{name}"
platform: "tensorrt_plan"
max_batch_size: {batch}
input [{{ name: "{spec.input_name}" data_type: TYPE_FP32 dims: [{input_shape}] }}]
output [{{ name: "{spec.output_name}" data_type: TYPE_FP32 dims: [{output_shape}] }}]
instance_group [{{ kind: KIND_GPU count: 1 gpus: [0] }}]
optimization {{ cuda {{ {cuda_config} }} }}
''')


def prepare_video(source, output):
    # Normalize the public 720p example to the benchmark's 1080p frame size.
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            source,
            "-map",
            "0:v:0",
            "-map_metadata",
            "-1",
            "-an",
            "-vf",
            "scale=1920:1080:flags=bilinear,setsar=1",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-qp",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            "-fflags",
            "+bitexact",
            "-flags:v",
            "+bitexact",
            output,
        ],
        check=True,
    )
    metadata = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name,r_frame_rate,nb_frames",
            "-of",
            "json",
            output,
        ],
        text=True,
    )
    (stream,) = json.loads(metadata)["streams"]
    if (stream["width"], stream["height"]) != (1920, 1080):
        raise RuntimeError("expected a 1920x1080 video")
    return {
        "path": output.name,
        "sha256": digest(output),
        "source": VIDEO_URL,
        "source_sha256": VIDEO_SHA256,
        "source_size": [1280, 720],
        "width": stream["width"],
        "height": stream["height"],
        "codec": stream["codec_name"],
        "fps": stream["r_frame_rate"],
        "frames": int(stream["nb_frames"]),
        "ffmpeg": subprocess.check_output(
            ["ffmpeg", "-version"], text=True
        ).splitlines()[0],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="new or empty directory"
    )
    parser.add_argument("--gpu", default="0", help="nvidia-smi GPU index or UUID")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("output directory must be empty")

    device = gpu_info(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = device.pop("uuid")
    source_onnx = output / "yolov8n-source.onnx"
    batched_onnx = output / "yolov8n-batched.onnx"
    source_video = output / "video-source.mp4"
    download(MODEL_URL, MODEL_SHA256, source_onnx)
    make_batched_onnx(source_onnx, batched_onnx)
    download(VIDEO_URL, VIDEO_SHA256, source_video)
    video = prepare_video(source_video, output / "video.mp4")

    manifest_models = {}
    for name, spec in MODELS.items():
        plan = output / f"{name}.plan"
        print(f"Building {name} on {device['name']}", flush=True)
        build_plan(name, plan, batched_onnx if name == "yolov8n" else None)
        manifest_models[name] = {
            "plan": plan.name,
            "sha256": digest(plan),
            "input_name": spec.input_name,
            "input_shape": spec.input_shape,
            "output_name": spec.output_name,
            "output_shape": spec.output_shape,
        }
    manifest_models["yolov8n"].update(
        source=MODEL_URL,
        source_sha256=MODEL_SHA256,
        batched_onnx_sha256=digest(batched_onnx),
    )
    manifest_models["crop_regression"].update(
        architecture="fixture: crop channel means times a fixed RGB-to-box-deltas matrix",
        weights=REGRESSION_WEIGHTS,
    )

    for batch in BATCHES:
        for graphs in (False, True):
            write_repository(output, batch, graphs)
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
            output / "graph-probe.so",
        ],
        check=True,
    )
    manifest = {
        "video": video,
        "models": manifest_models,
        "batches": BATCHES,
        "gpu": device,
        "tensorrt": trt.__version__,
        "cuda_runtime": checked(cuda.cudaRuntimeGetVersion()),
        "onnx": onnx.__version__,
        "numpy": np.__version__,
        "precision": "FP32",
        "tf32": False,
        "workspace_bytes": WORKSPACE,
        "profile_batches": PROFILE_BATCHES,
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
