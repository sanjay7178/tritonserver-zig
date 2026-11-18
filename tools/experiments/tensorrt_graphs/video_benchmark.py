#!/usr/bin/env python3
"""Measure a YOLOv8n video pipeline on Zig and stock Triton over loopback gRPC."""

import argparse
from dataclasses import dataclass
from hashlib import file_digest
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
from time import perf_counter_ns, time_ns

import numpy as np
from PIL import Image

from benchmark import PROBE_RESULT, gpu_info, summarize_ns, triton_stats, with_samples
from grpc_benchmark import MODEL_VERSION, infer, server_client

FRAME_WIDTH, FRAME_HEIGHT = 1920, 1080
DETECTOR = "yolov8n"
REGRESSOR = "crop_regression"
MODELS = (DETECTOR, REGRESSOR)
SIZE = 640
TOP_K = 4
CROP_SIZE = 64
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
TOLERANCE = {"rtol": 1e-4, "atol": 2e-3}
PROOF_BATCHES = 2


def sha256(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def decode(video, count):
    # Predecode once so every timed trial sees exactly the same RGB frames.
    data = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            video,
            "-map",
            "0:v:0",
            "-filter_threads",
            "1",
            "-vf",
            f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={FRAME_WIDTH}:{FRAME_HEIGHT}:(ow-iw)/2:(oh-ih)/2",
            "-frames:v",
            str(count),
            "-pix_fmt",
            "rgb24",
            "-threads",
            "1",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    frame_bytes = FRAME_HEIGHT * FRAME_WIDTH * 3
    if len(data) != count * frame_bytes:
        raise ValueError(
            f"expected {count} video frames, decoded {len(data) // frame_bytes}"
        )
    return np.frombuffer(data, dtype=np.uint8).reshape(
        count, FRAME_HEIGHT, FRAME_WIDTH, 3
    )


def letterbox(frame):
    scale = SIZE / max(FRAME_WIDTH, FRAME_HEIGHT)
    width = round(FRAME_WIDTH * scale)
    height = round(FRAME_HEIGHT * scale)
    left, top = (SIZE - width) // 2, (SIZE - height) // 2
    image = Image.fromarray(frame).resize((width, height), Image.Resampling.BILINEAR)
    padded = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)
    padded[top : top + height, left : left + width] = np.asarray(image)
    return padded.transpose(2, 0, 1), (
        left,
        top,
        width / FRAME_WIDTH,
        height / FRAME_HEIGHT,
    )


def detections(predictions, transform):
    class_scores = predictions[4:]
    classes = class_scores.argmax(axis=0)
    scores = class_scores[classes, np.arange(predictions.shape[1])]
    candidates = np.flatnonzero(scores >= CONFIDENCE_THRESHOLD)
    coords = predictions[:4, candidates].T
    center, half = coords[:, :2], coords[:, 2:] * 0.5
    boxes = np.column_stack((center - half, center + half))
    left, top, scale_x, scale_y = transform
    boxes[:, 0::2] = ((boxes[:, 0::2] - left) / scale_x).clip(0, FRAME_WIDTH)
    boxes[:, 1::2] = ((boxes[:, 1::2] - top) / scale_y).clip(0, FRAME_HEIGHT)
    classes, scores = classes[candidates], scores[candidates]

    keep = []
    for index in np.argsort(-scores, kind="stable"):
        box = boxes[index]
        extent = box[2:] - box[:2]
        if (extent <= 0).any():
            continue
        others = [previous for previous in keep if classes[previous] == classes[index]]
        previous = boxes[others]
        overlap = (
            np.minimum(previous[:, 2:], box[2:]) - np.maximum(previous[:, :2], box[:2])
        ).clip(0)
        intersection = overlap.prod(axis=1)
        union = (
            extent.prod()
            + (previous[:, 2:] - previous[:, :2]).prod(axis=1)
            - intersection
        )
        if (intersection > IOU_THRESHOLD * union).any():
            continue
        keep.append(index)
        if len(keep) == TOP_K:
            break
    return classes[keep], scores[keep], boxes[keep]


def crop_tensor(frames, found):
    crops = np.zeros((len(frames), TOP_K, 3, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    for batch, (frame, (_, _, boxes)) in enumerate(zip(frames, found, strict=True)):
        for index, box in enumerate(boxes):
            left, top = np.floor(box[:2]).astype(int)
            right, bottom = np.ceil(box[2:]).astype(int)
            image = Image.fromarray(frame[top:bottom, left:right]).resize(
                (CROP_SIZE, CROP_SIZE), Image.Resampling.BILINEAR
            )
            crops[batch, index] = np.asarray(image).transpose(2, 0, 1) * np.float32(
                1 / 255
            )
    return crops


def refined_detections(found, offsets):
    result = []
    for (classes, scores, boxes), delta in zip(found, offsets, strict=True):
        extent = boxes[:, 2:] - boxes[:, :2]
        refined = boxes + delta[: len(boxes)] * np.tile(extent, (1, 2))
        refined[:, 0::2] = refined[:, 0::2].clip(0, FRAME_WIDTH)
        refined[:, 1::2] = refined[:, 1::2].clip(0, FRAME_HEIGHT)
        result.append((classes, scores, refined))
    return result


@dataclass
class FrameBatch:
    predictions: np.ndarray
    offsets: np.ndarray
    detections: list
    detector_ns: int
    regressor_ns: int


def run_pipeline(client, frames):
    batch = len(frames)
    images = np.empty((batch, 3, SIZE, SIZE), dtype=np.float32)
    transforms = []
    for index, frame in enumerate(frames):
        image, transform = letterbox(frame)
        np.multiply(image, np.float32(1 / 255), out=images[index])
        transforms.append(transform)

    start = perf_counter_ns()
    predictions = infer(
        client, images, model=DETECTOR, input_name="images", output_name="output0"
    )
    detector_ns = perf_counter_ns() - start
    if (
        predictions is None
        or predictions.dtype != np.float32
        or predictions.shape != (batch, 84, 8400)
    ):
        raise RuntimeError("YOLOv8n returned an unexpected output shape or type")
    found = [
        detections(pred, transform)
        for pred, transform in zip(predictions, transforms, strict=True)
    ]
    crops = crop_tensor(frames, found)

    start = perf_counter_ns()
    offsets = infer(
        client, crops, model=REGRESSOR, input_name="crops", output_name="offsets"
    )
    regressor_ns = perf_counter_ns() - start
    if (
        offsets is None
        or offsets.dtype != np.float32
        or offsets.shape != (batch, TOP_K, 4)
    ):
        raise RuntimeError(
            "crop regression returned an unexpected output shape or type"
        )
    return FrameBatch(
        predictions,
        offsets,
        refined_detections(found, offsets),
        detector_ns,
        regressor_ns,
    )


def compare_results(actual, reference):
    errors = {}
    for name in ("predictions", "offsets"):
        value = getattr(actual, name)
        expected = getattr(reference, name)
        if not np.isfinite(value).all():
            raise RuntimeError(f"{name} contains non-finite values")
        np.testing.assert_allclose(value, expected, equal_nan=False, **TOLERANCE)
        errors[f"{name}_max_abs_error"] = float(
            np.max(np.abs(value - expected), initial=0)
        )
    box_error = 0.0
    for actual_frame, reference_frame in zip(
        actual.detections, reference.detections, strict=True
    ):
        classes, scores, boxes = actual_frame
        expected_classes, expected_scores, expected_boxes = reference_frame
        np.testing.assert_array_equal(classes, expected_classes)
        np.testing.assert_allclose(
            scores, expected_scores, equal_nan=False, **TOLERANCE
        )
        np.testing.assert_allclose(boxes, expected_boxes, equal_nan=False, **TOLERANCE)
        box_error = max(
            box_error, float(np.max(np.abs(boxes - expected_boxes), initial=0))
        )
    errors["refined_boxes_max_abs_error_px"] = box_error
    return errors


def prove(client_context, log, clips, graphs, baselines):
    with client_context as client:
        actual = [run_pipeline(client, frames) for frames in clips]
    expected_launches = len(clips) * len(MODELS) if graphs else 0
    launches = PROBE_RESULT.findall(log.read_text(errors="replace"))
    if launches != [str(expected_launches)]:
        raise RuntimeError(
            f"expected {expected_launches} graph launches, found {launches}, see {log}"
        )

    reference = actual if baselines is None else baselines
    errors = [
        compare_results(result, expected)
        for result, expected in zip(actual, reference, strict=True)
    ]
    return {
        "frame_batches": len(clips),
        "rpc_requests": len(clips) * len(MODELS),
        "backend_cudaGraphLaunch_successes": expected_launches,
        "detections": sum(
            len(frame[0]) for result in actual for frame in result.detections
        ),
        **{key: max(error[key] for error in errors) for key in errors[0]},
    }, reference


def stats(client, model):
    return client.get_inference_statistics(
        model, model_version=MODEL_VERSION, as_json=True, client_timeout=5
    )


def timed(client, clips, batch, warmup, iterations):
    def clip(index):
        return clips[index % len(clips)]

    for index in range(warmup):
        run_pipeline(client, clip(index))
    before = {model: stats(client, model) for model in MODELS}
    pipeline_ns, detector_ns, regressor_ns = [], [], []
    detection_count = 0
    start_unix_ns = time_ns()
    start = perf_counter_ns()
    for index in range(iterations):
        frame_batch_start = perf_counter_ns()
        result = run_pipeline(client, clip(index))
        pipeline_ns.append(perf_counter_ns() - frame_batch_start)
        detector_ns.append(result.detector_ns)
        regressor_ns.append(result.regressor_ns)
        detection_count += sum(len(frame[0]) for frame in result.detections)
    elapsed_ns = perf_counter_ns() - start
    end_unix_ns = time_ns()
    return {
        "elapsed_s": elapsed_ns / 1e9,
        "start_unix_ns": start_unix_ns,
        "end_unix_ns": end_unix_ns,
        "frame_batches": iterations,
        "frames": iterations * batch,
        "frames_per_s": iterations * batch * 1e9 / elapsed_ns,
        "rpc_requests": iterations * len(MODELS),
        "rpc_requests_per_s": iterations * len(MODELS) * 1e9 / elapsed_ns,
        "failed_requests": 0,
        "detections": detection_count,
        "frame_pipeline_latency_us": summarize_ns(pipeline_ns),
        "detector_rpc_latency_us": summarize_ns(detector_ns),
        "crop_regression_rpc_latency_us": summarize_ns(regressor_ns),
        "triton_statistics": {
            model: triton_stats(
                {
                    "statistics_before": before[model],
                    "statistics_after": stats(client, model),
                },
                batch,
                iterations,
            )
            for model in MODELS
        },
    }


def verify_plans(artifacts, repository, models):
    for model in MODELS:
        path = repository / model / MODEL_VERSION / "model.plan"
        pinned = artifacts / models[model]["plan"]
        if (
            path.resolve() != pinned.resolve()
            and sha256(path) != models[model]["sha256"]
        ):
            raise ValueError(f"{path} differs from the pinned plan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zig-server", type=Path, required=True)
    parser.add_argument("--stock-server", type=Path, required=True)
    parser.add_argument("--triton-prefix", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--video", type=Path, help="defaults to the clip in the artifact manifest"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="0", help="nvidia-smi GPU index or UUID")
    parser.add_argument(
        "--batches", type=int, nargs="+", help="defaults to the manifest's batches"
    )
    parser.add_argument(
        "--frames", type=int, default=32, help="distinct source frames to predecode"
    )
    parser.add_argument("--warmup", type=int, default=4, help="warmup frame batches")
    parser.add_argument(
        "--iterations", type=int, default=32, help="timed frame batches per trial"
    )
    parser.add_argument("--repeats", type=int, default=3, help="paired trials per case")
    args = parser.parse_args()

    artifacts = args.artifacts.resolve()
    manifest = json.loads((artifacts / "manifest.json").read_text())
    batches = args.batches or manifest["batches"]
    if len(set(batches)) != len(batches) or any(
        batch not in manifest["batches"] or batch <= 0 for batch in batches
    ):
        parser.error(
            "--batches must select distinct positive batch sizes from the artifact manifest"
        )
    if args.frames < PROOF_BATCHES * max(batches) or any(
        args.frames % batch for batch in batches
    ):
        parser.error(
            "--frames must be divisible by each batch size and supply at least two full batches"
        )
    if args.warmup < 1 or args.iterations < 2 or args.repeats < 1:
        parser.error(
            "use at least 1 warmup, 2 timed frame batches, and 1 paired repeat"
        )
    video = args.video or artifacts / manifest["video"]["path"]
    video_sha = sha256(video)
    if args.video is None and video_sha != manifest["video"]["sha256"]:
        parser.error("the clip differs from the artifact manifest")
    for model in MODELS:
        model_info = manifest["models"][model]
        if sha256(artifacts / model_info["plan"]) != model_info["sha256"]:
            parser.error(f"{model} plan differs from the artifact manifest")
    if not (artifacts / "graph-probe.so").is_file():
        parser.error("missing graph launch probe in the artifact directory")

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    prefix = args.triton_prefix.resolve()
    servers = {"stock": args.stock_server.resolve(), "zig": args.zig_server.resolve()}
    gpu = gpu_info(args.gpu)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(None, (str(prefix / "lib"), env.get("LD_LIBRARY_PATH")))
    )
    env.pop("LD_PRELOAD", None)
    proof_env = dict(env, LD_PRELOAD="./graph-probe.so")
    frames = decode(video, args.frames)
    report = {
        "gpu": {key: value for key, value in gpu.items() if key != "uuid"},
        "python": platform.python_version(),
        "tritonclient": version("tritonclient"),
        "grpcio": version("grpcio"),
        "numpy": np.__version__,
        "pillow": version("Pillow"),
        "ffmpeg": subprocess.check_output(
            ["ffmpeg", "-version"], text=True
        ).splitlines()[0],
        "triton_library_sha256": sha256(prefix / "lib/libtritonserver.so"),
        "triton_tensorrt_backend_sha256": sha256(
            prefix / "backends/tensorrt/libtriton_tensorrt.so"
        ),
        "servers": {
            kind: {"name": path.name, "sha256": sha256(path)}
            for kind, path in servers.items()
        },
        "assets": manifest,
        "video": {
            **(manifest["video"] if args.video is None else {}),
            "path": video.name,
            "sha256": video_sha,
            "decoded_dimensions_wh": [FRAME_WIDTH, FRAME_HEIGHT],
            "distinct_frames": args.frames,
        },
        "method": {
            "frame_latency": "from CPU letterbox through both RPCs and refined boxes, with the explicit frame batch ready at start",
            "video_decode": "predecoded RGB frames, excluded from timing",
            "rpc_latency": "Python request marshalling, unary loopback gRPC, and response array decoding",
            "max_rpc_requests_in_flight": 1,
            "native_metrics": "enabled on both servers, stock exporter bound to loopback",
            "warmup_frame_batches": args.warmup,
            "timed_frame_batches_per_trial": args.iterations,
            "paired_repeats": args.repeats,
            "top_k_crops_per_frame": TOP_K,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "nms_iou_threshold": IOU_THRESHOLD,
            "proof_tolerance": TOLERANCE,
        },
        "cases": [],
    }
    for batch in batches:
        clips = frames.reshape(-1, batch, FRAME_HEIGHT, FRAME_WIDTH, 3)
        baseline = None
        expected_detections = None
        for graphs in (False, True):
            case = f"b{batch}-graphs-{int(graphs)}"
            repository = artifacts / case / "models"
            verify_plans(artifacts, repository, manifest["models"])
            result = {"frame_batch_size": batch, "graphs_enabled": graphs}
            for kind, server in servers.items():
                log = output / f"{case}-{kind}-proof.log"
                proof, baseline = prove(
                    server_client(
                        server,
                        kind,
                        repository,
                        prefix,
                        proof_env,
                        log,
                        models=MODELS,
                        cwd=artifacts,
                    ),
                    log,
                    clips[:PROOF_BATCHES],
                    graphs,
                    baseline,
                )
                result[f"{kind}_grpc"] = {"replay_proof": proof, "trials": []}
            for trial in range(args.repeats):
                order = ("zig", "stock") if trial % 2 == 0 else ("stock", "zig")
                for kind in order:
                    log = output / f"{case}-{kind}-trial-{trial + 1}.log"
                    with server_client(
                        servers[kind],
                        kind,
                        repository,
                        prefix,
                        env,
                        log,
                        models=MODELS,
                    ) as client:
                        measurement = with_samples(
                            gpu["uuid"],
                            lambda: timed(
                                client, clips, batch, args.warmup, args.iterations
                            ),
                        )
                    if expected_detections is None:
                        expected_detections = measurement["detections"]
                    if measurement["detections"] != expected_detections:
                        raise RuntimeError(
                            f"{case} {kind} produced a different detection count in timed frames"
                        )
                    result[f"{kind}_grpc"]["trials"].append(
                        {"server_order": order, **measurement}
                    )
            for kind in servers:
                metrics = result[f"{kind}_grpc"]
                metrics["median_frames_per_s"] = median(
                    trial["frames_per_s"] for trial in metrics["trials"]
                )
                metrics["median_frame_pipeline_latency_us"] = {
                    p: median(
                        trial["frame_pipeline_latency_us"][p]
                        for trial in metrics["trials"]
                    )
                    for p in ("p50", "p95", "p99")
                }
            report["cases"].append(result)
            print(
                f"{case} YOLOv8n video pipeline verified for both gRPC servers",
                flush=True,
            )

    path = output / "video-grpc-report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
