#!/usr/bin/env python3

import argparse
import pathlib

from project_paths import ROOT_SAHI_PREPROCESS_CONFIG


def get_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
):
    if not (0.0 <= overlap_height_ratio < 1.0):
        raise ValueError("overlap_height_ratio must be in [0, 1)")
    if not (0.0 <= overlap_width_ratio < 1.0):
        raise ValueError("overlap_width_ratio must be in [0, 1)")

    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)

    slice_bboxes = []
    y_min = 0
    y_max = 0

    while y_max < image_height:
        x_min = 0
        x_max = 0
        y_max = y_min + slice_height

        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap

        y_min = y_max - y_overlap

    return slice_bboxes


def to_roi_params(slice_bboxes):
    values = []
    for left, top, right, bottom in slice_bboxes:
        values.extend([left, top, right - left, bottom - top])
    return values


def render_config(
    source_width: int,
    source_height: int,
    slice_width: int,
    slice_height: int,
    overlap_width_ratio: float,
    overlap_height_ratio: float,
    tensor_name: str,
    target_unique_id: int,
    src_id: int,
    gpu_id: int,
    include_full_frame: bool,
):
    bboxes = get_slice_bboxes(
        image_height=source_height,
        image_width=source_width,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )
    if include_full_frame:
        bboxes.append([0, 0, source_width, source_height])

    roi_params = to_roi_params(bboxes)
    roi_params_str = ";".join(str(v) for v in roi_params)
    roi_count = len(bboxes)

    content = f"""################################################################################
# Auto-generated SAHI-style nvdspreprocess config
# Source: {source_width}x{source_height}
# Slice: {slice_width}x{slice_height}
# Overlap: width={overlap_width_ratio:.4f}, height={overlap_height_ratio:.4f}
# ROI count: {roi_count}
################################################################################

[property]
enable=1
target-unique-ids={target_unique_id}
network-input-order=0
process-on-frame=1
unique-id=15
gpu-id={gpu_id}
maintain-aspect-ratio=1
symmetric-padding=1
processing-width={slice_width}
processing-height={slice_height}
scaling-buf-pool-size=16
tensor-buf-pool-size=16
network-input-shape={roi_count};3;{slice_height};{slice_width}
network-color-format=0
tensor-data-type=0
tensor-name={tensor_name}
scaling-pool-memory-type=0
scaling-pool-compute-hw=0
scaling-filter=0
custom-lib-path=/opt/nvidia/deepstream/deepstream/lib/gst-plugins/libcustom2d_preprocess.so
custom-tensor-preparation-function=CustomTensorPreparation

[user-configs]
pixel-normalization-factor=0.00392156862745

[group-0]
src-ids={src_id}
custom-input-transformation-function=CustomAsyncTransformation
process-on-roi=1
draw-roi=0
roi-params-src-{src_id}={roi_params_str}
"""
    return content, roi_count


def parse_args():
    parser = argparse.ArgumentParser(description="Generate valid nvdspreprocess config for SAHI-style tiling")
    parser.add_argument("--source-width", type=int, required=True)
    parser.add_argument("--source-height", type=int, required=True)
    parser.add_argument("--slice-width", type=int, default=640)
    parser.add_argument("--slice-height", type=int, default=640)
    parser.add_argument("--overlap-width-ratio", type=float, default=0.2)
    parser.add_argument("--overlap-height-ratio", type=float, default=0.2)
    parser.add_argument("--tensor-name", default="input", help="Model input tensor name from ONNX/engine")
    parser.add_argument("--target-unique-id", type=int, default=1, help="GIE unique id targeted by preprocess")
    parser.add_argument("--src-id", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--include-full-frame",
        action="store_true",
        help="Append one full-frame ROI to mimic SAHI standard prediction",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT_SAHI_PREPROCESS_CONFIG),
        help="Output config path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    content, roi_count = render_config(
        source_width=args.source_width,
        source_height=args.source_height,
        slice_width=args.slice_width,
        slice_height=args.slice_height,
        overlap_width_ratio=args.overlap_width_ratio,
        overlap_height_ratio=args.overlap_height_ratio,
        tensor_name=args.tensor_name,
        target_unique_id=args.target_unique_id,
        src_id=args.src_id,
        gpu_id=args.gpu_id,
        include_full_frame=args.include_full_frame,
    )

    output_path = pathlib.Path(args.output).expanduser().resolve()
    output_path.write_text(content, encoding="utf-8")

    print(f"Wrote: {output_path}")
    print(f"ROI count: {roi_count}")
    print("Set nvinfer batch-size to ROI count for best SAHI throughput")
    print("Rebuild engine with dynamic batch support for that setting")


if __name__ == "__main__":
    main()
