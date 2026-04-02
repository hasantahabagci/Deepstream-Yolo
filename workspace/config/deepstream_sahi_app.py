#!/usr/bin/env python3

import argparse
import configparser
import os
import pathlib
import signal
import sys
import time
from dataclasses import dataclass

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

import pyds
from project_paths import (
    ROOT_DEEPSTREAM_TRACKER_CONFIG,
    ROOT_SAHI_INFER_CONFIG,
    ROOT_SAHI_PREPROCESS_CONFIG,
)

MUXER_BATCH_TIMEOUT_USEC = 40000
GST_CAPS_FEATURES_NVMM = "memory:NVMM"


@dataclass
class MergeConfig:
    enabled: bool
    metric: str
    threshold: float
    class_agnostic: bool


def to_uri(path_or_uri: str) -> str:
    if "://" in path_or_uri:
        return path_or_uri
    return pathlib.Path(path_or_uri).expanduser().resolve().as_uri()


def bus_call(bus, message, loop):
    mtype = message.type
    if mtype == Gst.MessageType.EOS:
        print("End-of-stream")
        loop.quit()
    elif mtype == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}")
        if debug:
            print(f"Debug: {debug}")
        loop.quit()
    return True


def cb_newpad(decodebin, decoder_src_pad, data):
    caps = decoder_src_pad.get_current_caps()
    if not caps:
        return

    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    features = caps.get_features(0)

    if "video" in gstname and features.contains(GST_CAPS_FEATURES_NVMM):
        source_bin = data
        bin_ghost_pad = source_bin.get_static_pad("src")
        if not bin_ghost_pad.set_target(decoder_src_pad):
            sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
    elif "video" in gstname:
        sys.stderr.write("Decodebin did not pick an NVMM decoder.\n")


def decodebin_child_added(child_proxy, obj, name, user_data):
    if name.find("decodebin") != -1:
        obj.connect("child-added", decodebin_child_added, user_data)


def create_source_bin(index: int, uri: str):
    bin_name = f"source-bin-{index:02d}"
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        return None

    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", f"uri-decode-bin-{index}")
    if not uri_decode_bin:
        return None

    uri_decode_bin.set_property("uri", uri)
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    Gst.Bin.add(nbin, uri_decode_bin)
    if not nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)):
        return None
    return nbin


def bbox_to_xyxy(obj_meta):
    left = float(obj_meta.rect_params.left)
    top = float(obj_meta.rect_params.top)
    width = float(obj_meta.rect_params.width)
    height = float(obj_meta.rect_params.height)
    return [left, top, left + width, top + height]


def intersection_area(box_a, box_b):
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    return (x_right - x_left) * (y_bottom - y_top)


def box_area(box):
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    return width * height


def overlap_metric(box_a, box_b, metric: str):
    inter = intersection_area(box_a, box_b)
    if inter <= 0.0:
        return 0.0
    area_a = box_area(box_a)
    area_b = box_area(box_b)
    if metric == "iou":
        union = area_a + area_b - inter
        return 0.0 if union <= 0.0 else inter / union
    smaller = min(area_a, area_b)
    return 0.0 if smaller <= 0.0 else inter / smaller


def union_box(box_a, box_b):
    return [
        min(box_a[0], box_b[0]),
        min(box_a[1], box_b[1]),
        max(box_a[2], box_b[2]),
        max(box_a[3], box_b[3]),
    ]


def greedy_nmm_indices(detections, metric: str, threshold: float, class_agnostic: bool):
    if not detections:
        return {}

    order = sorted(
        range(len(detections)),
        key=lambda i: (
            -detections[i]["confidence"],
            detections[i]["bbox"][0],
            detections[i]["bbox"][1],
            detections[i]["bbox"][2],
            detections[i]["bbox"][3],
        ),
    )

    suppressed = set()
    keep_to_merge = {}

    for current_idx in order:
        if current_idx in suppressed:
            continue

        current = detections[current_idx]
        current_box = current["bbox"]
        merge_list = []

        for cand_idx in order:
            if cand_idx == current_idx or cand_idx in suppressed:
                continue

            candidate = detections[cand_idx]
            if not class_agnostic and candidate["class_id"] != current["class_id"]:
                continue
            if candidate["confidence"] > current["confidence"]:
                continue
            if candidate["confidence"] == current["confidence"]:
                if tuple(candidate["bbox"]) > tuple(current_box):
                    continue

            if overlap_metric(current_box, candidate["bbox"], metric) >= threshold:
                merge_list.append(cand_idx)
                suppressed.add(cand_idx)

        keep_to_merge[current_idx] = merge_list

    return keep_to_merge


def merge_frame_detections(frame_meta, merge_cfg: MergeConfig):
    detections = []
    l_obj = frame_meta.obj_meta_list
    while l_obj is not None:
        try:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
        except StopIteration:
            break

        detections.append(
            {
                "obj_meta": obj_meta,
                "class_id": int(obj_meta.class_id),
                "confidence": float(obj_meta.confidence),
                "bbox": bbox_to_xyxy(obj_meta),
            }
        )

        try:
            l_obj = l_obj.next
        except StopIteration:
            break

    if len(detections) <= 1:
        return

    keep_to_merge = greedy_nmm_indices(
        detections,
        metric=merge_cfg.metric,
        threshold=merge_cfg.threshold,
        class_agnostic=merge_cfg.class_agnostic,
    )

    if not keep_to_merge:
        return

    merged_indices = set()
    for keep_idx, merge_list in keep_to_merge.items():
        keep_det = detections[keep_idx]
        keep_bbox = keep_det["bbox"]
        keep_conf = keep_det["confidence"]
        keep_class = keep_det["class_id"]

        for merge_idx in merge_list:
            candidate = detections[merge_idx]
            if overlap_metric(keep_bbox, candidate["bbox"], merge_cfg.metric) < merge_cfg.threshold:
                continue
            keep_bbox = union_box(keep_bbox, candidate["bbox"])
            if candidate["confidence"] > keep_conf:
                keep_conf = candidate["confidence"]
                keep_class = candidate["class_id"]
            merged_indices.add(merge_idx)

        keep_det["bbox"] = keep_bbox
        keep_det["confidence"] = keep_conf
        keep_det["class_id"] = keep_class

    for keep_idx in keep_to_merge.keys():
        keep_det = detections[keep_idx]
        obj_meta = keep_det["obj_meta"]
        x1, y1, x2, y2 = keep_det["bbox"]
        obj_meta.rect_params.left = float(x1)
        obj_meta.rect_params.top = float(y1)
        obj_meta.rect_params.width = float(max(0.0, x2 - x1))
        obj_meta.rect_params.height = float(max(0.0, y2 - y1))
        obj_meta.class_id = int(keep_det["class_id"])
        obj_meta.confidence = float(keep_det["confidence"])

        obj_meta.detector_bbox_info.org_bbox_coords.left = obj_meta.rect_params.left
        obj_meta.detector_bbox_info.org_bbox_coords.top = obj_meta.rect_params.top
        obj_meta.detector_bbox_info.org_bbox_coords.width = obj_meta.rect_params.width
        obj_meta.detector_bbox_info.org_bbox_coords.height = obj_meta.rect_params.height

    for idx in merged_indices:
        pyds.nvds_remove_obj_meta_from_frame(frame_meta, detections[idx]["obj_meta"])


def pgie_src_pad_buffer_probe(pad, info, user_data):
    merge_cfg: MergeConfig = user_data["merge_cfg"]
    if not merge_cfg.enabled:
        return Gst.PadProbeReturn.OK

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    pyds.nvds_acquire_meta_lock(batch_meta)
    try:
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
            merge_frame_detections(frame_meta, merge_cfg)
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
    finally:
        pyds.nvds_release_meta_lock(batch_meta)

    return Gst.PadProbeReturn.OK


def configure_tracker(tracker, tracker_config_path: str):
    def _set_if_supported(prop_name, value):
        if tracker.find_property(prop_name) is not None:
            tracker.set_property(prop_name, value)

    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    if not config.read(tracker_config_path):
        raise FileNotFoundError(f"Unable to read tracker config: {tracker_config_path}")
    if "tracker" not in config:
        raise ValueError("Tracker config missing [tracker] section")

    tracker_section = config["tracker"]
    if "tracker-width" in tracker_section:
        _set_if_supported("tracker-width", tracker_section.getint("tracker-width"))
    if "tracker-height" in tracker_section:
        _set_if_supported("tracker-height", tracker_section.getint("tracker-height"))
    if "gpu-id" in tracker_section:
        _set_if_supported("gpu-id", tracker_section.getint("gpu-id"))
    if "ll-lib-file" in tracker_section:
        _set_if_supported("ll-lib-file", tracker_section.get("ll-lib-file"))
    if "ll-config-file" in tracker_section:
        _set_if_supported("ll-config-file", tracker_section.get("ll-config-file"))
    if "enable-batch-process" in tracker_section:
        _set_if_supported("enable-batch-process", tracker_section.getint("enable-batch-process"))
    if "enable-past-frame" in tracker_section:
        _set_if_supported("enable-past-frame", tracker_section.getint("enable-past-frame"))


def make_element(factory_name: str, element_name: str):
    element = Gst.ElementFactory.make(factory_name, element_name)
    if not element:
        raise RuntimeError(f"Unable to create element '{element_name}' ({factory_name})")
    return element


def link_elements(*elements):
    for upstream, downstream in zip(elements[:-1], elements[1:]):
        if not upstream.link(downstream):
            raise RuntimeError(f"Unable to link {upstream.get_name()} -> {downstream.get_name()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepStream SAHI pipeline for YOLO11 (nvdspreprocess + nvinfer + cross-tile merge)"
    )
    parser.add_argument("--source", required=True, help="Input file path, RTSP URI, or file URI")
    parser.add_argument(
        "--preprocess-config",
        default=str(ROOT_SAHI_PREPROCESS_CONFIG),
        help="nvdspreprocess config file",
    )
    parser.add_argument(
        "--infer-config",
        default=str(ROOT_SAHI_INFER_CONFIG),
        help="nvinfer config file",
    )
    parser.add_argument("--width", type=int, default=1920, help="streammux width")
    parser.add_argument("--height", type=int, default=1080, help="streammux height")
    parser.add_argument("--tiled-width", type=int, default=1280, help="tiler output width")
    parser.add_argument("--tiled-height", type=int, default=720, help="tiler output height")
    parser.add_argument("--no-display", action="store_true", help="Use fakesink instead of display sink")
    parser.add_argument("--enable-tracker", action="store_true", help="Enable nvtracker after PGIE")
    parser.add_argument(
        "--tracker-config",
        default=str(ROOT_DEEPSTREAM_TRACKER_CONFIG),
        help="Tracker config file path",
    )
    parser.add_argument(
        "--disable-merge",
        action="store_true",
        help="Disable SAHI-style post-merge. Not recommended for overlapping tiles.",
    )
    parser.add_argument(
        "--merge-metric",
        choices=["ios", "iou"],
        default="ios",
        help="Matching metric used in merge stage",
    )
    parser.add_argument("--merge-threshold", type=float, default=0.5, help="Merge threshold")
    parser.add_argument(
        "--merge-class-agnostic",
        action="store_true",
        help="Merge across classes too (default is class-aware, like SAHI)",
    )
    parser.add_argument(
        "--pgie-batch-size",
        type=int,
        default=0,
        help="Override PGIE batch-size property. Keep 0 to use infer config value.",
    )
    parser.add_argument(
        "--det-interval",
        type=int,
        default=-1,
        help="Override PGIE interval (detect every N+1 frames). -1 keeps config value.",
    )
    parser.add_argument(
        "--batched-push-timeout",
        type=int,
        default=MUXER_BATCH_TIMEOUT_USEC,
        help="nvstreammux batched-push-timeout in usec.",
    )
    parser.add_argument(
        "--leaky-queues",
        action="store_true",
        help="Set all queues leaky downstream to reduce latency in real-time mode.",
    )
    parser.add_argument(
        "--realtime-preset",
        action="store_true",
        help="Enable practical real-time preset: tracker ON, merge OFF, det-interval=2, leaky queues ON.",
    )
    return parser.parse_args()


def _read_ini(path: str):
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    if not parser.read(path):
        raise FileNotFoundError(f"Unable to read config file: {path}")
    return parser


def _resolve_path(base_file: str, maybe_relative: str) -> str:
    expanded = os.path.expanduser(maybe_relative)
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(os.path.dirname(base_file), expanded))


def validate_sahi_config_compat(preprocess_config: str, infer_config: str):
    pre_cfg = _read_ini(preprocess_config)
    if "property" not in pre_cfg:
        raise ValueError(f"{preprocess_config} missing [property] section")

    pre_prop = pre_cfg["property"]
    proc_w = pre_prop.getint("processing-width")
    proc_h = pre_prop.getint("processing-height")
    pre_tensor_name = pre_prop.get("tensor-name", fallback="").strip()

    inf_cfg = _read_ini(infer_config)
    if "property" not in inf_cfg:
        raise ValueError(f"{infer_config} missing [property] section")

    inf_prop = inf_cfg["property"]
    onnx_file = inf_prop.get("onnx-file", fallback="").strip()
    if not onnx_file:
        return

    onnx_abs = _resolve_path(infer_config, onnx_file)
    if not os.path.exists(onnx_abs):
        print(f"Warning: ONNX file not found for shape precheck: {onnx_abs}")
        return

    try:
        import onnx
    except Exception:
        print("Warning: python 'onnx' module not available, skipping shape precheck.")
        return

    model = onnx.load(onnx_abs)
    if not model.graph.input:
        print("Warning: ONNX has no graph inputs, skipping shape precheck.")
        return

    input_tensor = model.graph.input[0]
    onnx_name = input_tensor.name
    dims = [d.dim_value for d in input_tensor.type.tensor_type.shape.dim]
    if len(dims) < 4:
        print("Warning: ONNX input rank is <4, skipping shape precheck.")
        return

    onnx_h = dims[2]
    onnx_w = dims[3]
    if pre_tensor_name and pre_tensor_name != onnx_name:
        print(
            f"Warning: tensor-name mismatch (preprocess='{pre_tensor_name}', onnx='{onnx_name}'). "
            "Using mismatched tensor names will break tensor-meta inference."
        )

    if onnx_w > 0 and onnx_h > 0:
        if onnx_w != proc_w or onnx_h != proc_h:
            raise RuntimeError(
                "Shape mismatch between preprocess and ONNX. "
                f"preprocess={proc_w}x{proc_h}, onnx={onnx_w}x{onnx_h}. "
                "Regenerate preprocess config or switch ONNX."
            )

    print(f"Shape precheck OK: preprocess={proc_w}x{proc_h}, onnx={onnx_w}x{onnx_h}, tensor='{onnx_name}'")


def build_pipeline(args, merge_cfg: MergeConfig):
    Gst.init(None)

    pipeline = Gst.Pipeline.new("deepstream-sahi-pipeline")
    if not pipeline:
        raise RuntimeError("Unable to create GstPipeline")

    uri = to_uri(args.source)
    source_bin = create_source_bin(0, uri)
    if not source_bin:
        raise RuntimeError("Unable to create source bin")

    streammux = make_element("nvstreammux", "streammux")
    queue1 = make_element("queue", "queue-preprocess")
    preprocess = make_element("nvdspreprocess", "preprocess")
    queue2 = make_element("queue", "queue-pgie")
    pgie = make_element("nvinfer", "primary-gie")
    queue3 = make_element("queue", "queue-post-pgie")

    tracker = None
    queue_tracker = None
    if args.enable_tracker:
        tracker = make_element("nvtracker", "tracker")
        queue_tracker = make_element("queue", "queue-tracker")

    tiler = make_element("nvmultistreamtiler", "tiler")
    queue4 = make_element("queue", "queue-tiler")
    nvvidconv = make_element("nvvideoconvert", "nvvidconv")
    queue5 = make_element("queue", "queue-conv")
    nvosd = make_element("nvdsosd", "nvosd")
    queue6 = make_element("queue", "queue-sink")

    sink = make_element("fakesink", "fakesink") if args.no_display else make_element("nveglglessink", "sink")

    pipeline.add(source_bin)
    for element in [streammux, queue1, preprocess, queue2, pgie, queue3]:
        pipeline.add(element)
    if tracker:
        pipeline.add(tracker)
        pipeline.add(queue_tracker)
    for element in [tiler, queue4, nvvidconv, queue5, nvosd, queue6, sink]:
        pipeline.add(element)

    streammux.set_property("width", args.width)
    streammux.set_property("height", args.height)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", args.batched_push_timeout)
    streammux.set_property("live-source", 1 if uri.startswith("rtsp://") else 0)

    preprocess.set_property("config-file", args.preprocess_config)

    pgie.set_property("config-file-path", args.infer_config)
    pgie.set_property("input-tensor-meta", True)
    if args.pgie_batch_size > 0:
        pgie.set_property("batch-size", args.pgie_batch_size)
    if args.det_interval >= 0:
        pgie.set_property("interval", args.det_interval)

    if tracker:
        configure_tracker(tracker, args.tracker_config)

    tiler.set_property("rows", 1)
    tiler.set_property("columns", 1)
    tiler.set_property("width", args.tiled_width)
    tiler.set_property("height", args.tiled_height)

    sink.set_property("sync", False)
    sink.set_property("qos", False)

    if args.leaky_queues:
        queue_list = [queue1, queue2, queue3, queue4, queue5, queue6]
        if queue_tracker:
            queue_list.append(queue_tracker)
        for q in queue_list:
            q.set_property("max-size-buffers", 1)
            q.set_property("max-size-bytes", 0)
            q.set_property("max-size-time", 0)
            q.set_property("leaky", 2)

    sinkpad = streammux.request_pad_simple("sink_0")
    if not sinkpad:
        raise RuntimeError("Unable to request streammux sink pad")
    srcpad = source_bin.get_static_pad("src")
    if not srcpad:
        raise RuntimeError("Unable to get source bin src pad")
    if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
        raise RuntimeError("Unable to link source bin to streammux")

    link_elements(streammux, queue1, preprocess, queue2, pgie, queue3)

    if tracker:
        link_elements(queue3, tracker, queue_tracker, tiler, queue4, nvvidconv, queue5, nvosd, queue6, sink)
    else:
        link_elements(queue3, tiler, queue4, nvvidconv, queue5, nvosd, queue6, sink)

    pgie_src_pad = pgie.get_static_pad("src")
    if not pgie_src_pad:
        raise RuntimeError("Unable to get PGIE src pad")
    pgie_src_pad.add_probe(
        Gst.PadProbeType.BUFFER,
        pgie_src_pad_buffer_probe,
        {"merge_cfg": merge_cfg},
    )

    return pipeline


def main():
    args = parse_args()
    if args.realtime_preset:
        args.enable_tracker = True
        args.leaky_queues = True
        if args.det_interval < 0:
            args.det_interval = 2

    validate_sahi_config_compat(args.preprocess_config, args.infer_config)

    merge_cfg = MergeConfig(
        enabled=(not args.disable_merge) and (not args.realtime_preset),
        metric=args.merge_metric.lower(),
        threshold=args.merge_threshold,
        class_agnostic=args.merge_class_agnostic,
    )

    print("Starting DeepStream SAHI pipeline with settings:")
    print(f"  source={args.source}")
    print(f"  preprocess_config={args.preprocess_config}")
    print(f"  infer_config={args.infer_config}")
    print(f"  tracker={'enabled' if args.enable_tracker else 'disabled'}")
    print(f"  det_interval={args.det_interval if args.det_interval >= 0 else 'from infer config'}")
    print(f"  leaky_queues={args.leaky_queues}")
    print(f"  realtime_preset={args.realtime_preset}")
    print(
        "  merge="
        + (
            f"enabled(metric={merge_cfg.metric}, threshold={merge_cfg.threshold}, class_agnostic={merge_cfg.class_agnostic})"
            if merge_cfg.enabled
            else "disabled"
        )
    )

    loop = GLib.MainLoop()
    pipeline = build_pipeline(args, merge_cfg)
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    def _handle_signal(sig, frame):
        pipeline.send_event(Gst.Event.new_eos())

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    start = time.time()
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        elapsed = time.time() - start
        print(f"Pipeline stopped. Runtime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
