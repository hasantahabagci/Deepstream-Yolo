#!/usr/bin/env python3

import argparse
import os
from urllib.request import urlopen

import cv2
import numpy as np

from project_paths import DEFAULT_TRT_ENGINE, ROOT_LABELS_PATH, TENSORRT_OUTPUT_DIR
from tensorrt_inference import TensorRTDetector, draw_detections, load_labels


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a direct TensorRT smoke test.")
    parser.add_argument(
        "--engine",
        default=str(DEFAULT_TRT_ENGINE),
        help="Path to the TensorRT engine file.",
    )
    parser.add_argument(
        "--labels",
        default=str(ROOT_LABELS_PATH),
        help="Optional labels file, one class name per line.",
    )
    parser.add_argument(
        "--source",
        default="https://ultralytics.com/images/bus.jpg",
        help="Image path, image URL, or local video path.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output path for the annotated image or video. Defaults based on source type.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.25,
        help="Minimum score to keep a detection.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="IoU threshold used by NMS.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Maximum detections to keep after NMS.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display frames while processing video or image output.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for video processing. 0 means process the entire file.",
    )
    return parser.parse_args()


def is_video_source(source: str) -> bool:
    return os.path.splitext(source)[1].lower() in VIDEO_EXTENSIONS and not source.startswith(
        ("http://", "https://")
    )


def default_output_path(source: str, is_video: bool) -> str:
    if is_video:
        source_name = os.path.splitext(os.path.basename(source))[0] or "video"
        return str(TENSORRT_OUTPUT_DIR / f"{source_name}_tensorrt_output.mp4")
    return str(TENSORRT_OUTPUT_DIR / "tensorrt_test_output.jpg")


def load_image(source: str) -> np.ndarray:
    if source.startswith(("http://", "https://")):
        with urlopen(source) as response:
            data = np.asarray(bytearray(response.read()), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(source, cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"Failed to load image from {source}")
    return image


def run_image(detector: TensorRTDetector, source: str, output: str, display: bool) -> None:
    image = load_image(source)
    detections, inference_ms = detector.infer(image)
    annotated = draw_detections(image, detections)
    if not cv2.imwrite(output, annotated):
        raise RuntimeError(f"Failed to save annotated image to {output}")

    print(f"Inference time: {inference_ms:.2f} ms")
    print(f"Detections: {len(detections)}")
    for det in detections:
        print(
            f"- class={det.class_id} label={det.label} score={det.score:.3f} "
            f"box=({det.x1:.1f}, {det.y1:.1f}, {det.x2:.1f}, {det.y2:.1f})"
        )
    print(f"Annotated image saved to {output}")

    if display:
        cv2.imshow("TensorRT Test", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video(
    detector: TensorRTDetector,
    source: str,
    output: str,
    display: bool,
    max_frames: int,
) -> None:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video source {source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to open video writer for {output}")

    total_frames = 0
    total_detections = 0
    total_inference_ms = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            detections, inference_ms = detector.infer(frame)
            annotated = draw_detections(frame, detections)
            writer.write(annotated)

            total_frames += 1
            total_detections += len(detections)
            total_inference_ms += inference_ms

            print(
                f"frame={total_frames} detections={len(detections)} "
                f"inference_ms={inference_ms:.2f}"
            )

            if display:
                cv2.imshow("TensorRT Test", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if max_frames > 0 and total_frames >= max_frames:
                break
    finally:
        capture.release()
        writer.release()
        if display:
            cv2.destroyAllWindows()

    if total_frames == 0:
        raise RuntimeError(f"No frames were read from {source}")

    average_inference_ms = total_inference_ms / total_frames
    print(f"Processed frames: {total_frames}")
    print(f"Total detections: {total_detections}")
    print(f"Average inference time: {average_inference_ms:.2f} ms")
    print(f"Annotated video saved to {output}")


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    detector = TensorRTDetector(
        engine_path=args.engine,
        labels=labels,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        top_k=args.top_k,
    )
    video_source = is_video_source(args.source)
    output = args.output or default_output_path(args.source, video_source)

    print(f"Loading {args.engine} for TensorRT inference...")
    print(f"TensorRT input: {detector.input_name} {detector.input_shape}")
    print(f"TensorRT output: {detector.output_name} {detector.output_shape}")
    print(f"Source: {args.source}")
    print(f"Output: {output}")

    if video_source:
        run_video(detector, args.source, output, args.display, args.max_frames)
    else:
        run_image(detector, args.source, output, args.display)


if __name__ == "__main__":
    main()
