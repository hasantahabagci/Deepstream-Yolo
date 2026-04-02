#!/usr/bin/env python3

"""
ROS 2 publisher built directly on python_test.py pipeline logic.

Pipeline: nvarguscamerasrc -> capsfilter -> nvstreammux -> nvinfer -> nvvideoconvert
          -> tee (raw branch + annotated HUD branch) just like python_test.py.

Published topics (same as deepstream_ros2_publisher.py):
  /detection/bbox_center   geometry_msgs/Point (x=cx, y=cy, z=confidence)
  /detection/all_detections std_msgs/String (JSON payload with all bboxes)

Redis side-effects from python_test.py are preserved (bbox, frame_number, cam_fps, etc.).
"""

import sys
import time
import json
import threading
import os
from datetime import datetime

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GObject, Gst

import pyds

from geometry_msgs.msg import Point
from std_msgs.msg import String
import rclpy
from rclpy.node import Node

from redis_helper import RedisHelper
from moving_average import MovingAverage


# === Runtime settings (identical defaults to python_test.py) ===
RESOLUTION = (1920, 1080)
FPS = 40
DISPLAY = True

# === Logging paths (aligned with deepstream_ros2_publisher.py) ===
LOGS_DIR = "/home/ituarc/ros2_ws/src/thermal_guidance/logs"
VIDEO_LOG_DIR = os.path.join(LOGS_DIR, "videos")
LOG_ID_FILE = os.path.join(LOGS_DIR, "last_log_id.txt")

# === Globals shared with pad probe ===
ma = MovingAverage(60)
redis = RedisHelper()
frame_count = 0
start_time = time.time()
fps_interval = 1
ros_node = None  # set in main()


def calculate_fps():
    """Compute instantaneous FPS every `fps_interval` frames."""
    global frame_count, start_time
    frame_count += 1
    if frame_count % fps_interval == 0:
        now = time.time()
        elapsed = now - start_time
        fps = fps_interval / elapsed if elapsed > 0 else 0.0
        start_time = now
        return fps
    return None


class BBoxPublisherNode(Node):
    """ROS 2 publishers mirroring deepstream_ros2_publisher.py."""

    def __init__(self):
        super().__init__("python_test_bbox_publisher")
        self.center_pub = self.create_publisher(Point, "/detection/bbox_center", 40)
        self.all_pub = self.create_publisher(String, "/detection/all_detections", 10)

    def publish_frame(self, frame_number: int, fps: float, bboxes: list, best: dict | None):
        msg_time = time.time()

        # Best detection center
        pt = Point()
        if best:
            cx = best["left"] + best["width"] / 2.0
            cy = best["top"] + best["height"] / 2.0
            pt.x = float(cx)
            pt.y = float(cy)
            pt.z = float(best.get("confidence", 0.0))
        else:
            pt.x = float("nan")
            pt.y = float("nan")
            pt.z = 0.0
        self.center_pub.publish(pt)

        # All detections JSON
        payload = {
            "frame_number": int(frame_number),
            "timestamp": float(msg_time),
            "fps": float(fps) if fps is not None else 0.0,
            "num_objects": len(bboxes),
            "detections": [
                {
                    "class_id": int(bb["class_id"]),
                    "confidence": float(bb["confidence"]),
                    "bbox": {
                        "left": float(bb["left"]),
                        "top": float(bb["top"]),
                        "width": float(bb["width"]),
                        "height": float(bb["height"]),
                    },
                    "center": {
                        "x": float(bb["left"] + bb["width"] / 2.0),
                        "y": float(bb["top"] + bb["height"] / 2.0),
                    },
                }
                for bb in bboxes
            ] or None,
        }

        msg = String()
        msg.data = json.dumps(payload)
        self.all_pub.publish(msg)


def _safe_get(redis_conn, key, default="0"):
    try:
        val = redis_conn.r.get(key)
        return val.decode("utf-8") if val is not None else default
    except Exception:
        return default


def osd_sink_pad_buffer_probe(pad, info, u_data):
    """Probe that mirrors python_test.py and additionally publishes to ROS 2."""
    global ros_node

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK

    current_fps = calculate_fps()

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta

        bboxes = []
        max_conf = 0.0
        target_bbox = None

        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            bbox = {
                "class_id": obj_meta.class_id,
                "confidence": obj_meta.confidence,
                "left": obj_meta.rect_params.left,
                "top": obj_meta.rect_params.top,
                "width": obj_meta.rect_params.width,
                "height": obj_meta.rect_params.height,
            }
            bboxes.append(bbox)

            if bbox["confidence"] > max_conf:
                max_conf = bbox["confidence"]
                target_bbox = bbox

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Redis updates (unchanged from python_test.py spirit)
        if target_bbox:
            redis.r.set("bbox", json.dumps(target_bbox))
        else:
            redis.r.set("bbox", json.dumps({}))

        # HUD text sourced from Redis state
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]

        offset = _safe_get(redis, "offset")
        depth_virt = _safe_get(redis, "depth_virt")
        drone_mode = _safe_get(redis, "drone_mode")
        distance = _safe_get(redis, "distance")
        attitude = _safe_get(redis, "attitude")
        velocity = _safe_get(redis, "velocity")
        interceptor_location = _safe_get(redis, "interceptor_location")
        target_location = _safe_get(redis, "target_location")
        pixel_error = _safe_get(redis, "pixel_errors")
        filter_roll = _safe_get(redis, "filter_roll")
        kp_yaw = _safe_get(redis, "kp_yaw")
        cbf = _safe_get(redis, "cbf")
        roll_rate = _safe_get(redis, "roll_rate")
        pitch_rate = _safe_get(redis, "pitch_rate")
        yaw_rate = _safe_get(redis, "yaw_rate")
        throttle_command = _safe_get(redis, "throttle_command")
        redis.r.set("frame_number", frame_number)
        redis.r.set("cam_fps", current_fps if current_fps is not None else 0.0)
        fps_vis = ma.update(current_fps or 0.0)

        display_text2 = (
            f"Drone Mode: {drone_mode}\n"
            f"Distance: {distance} m\n"
            f"Attitude: {attitude}\n"
            f"Velocity: {velocity} m/s\n"
            f"Interceptor Location: {interceptor_location}\n"
            f"Target Location: {target_location}\n"
            f"Pixel Error: {pixel_error}\n"
            f"Filter Roll: {filter_roll}\n"
            f"Kp Yaw: {kp_yaw}\n"
            f"CBF: {cbf}\n"
            f"Roll Rate: {roll_rate} rad/s\n"
            f"Pitch Rate: {pitch_rate} rad/s\n"
            f"Yaw Rate: {yaw_rate} rad/s\n"
            f"Throttle Command: {throttle_command}\n"
            f"Offset: {offset}, Depth: {depth_virt}"
        )
        fps_text = f" FPS: {fps_vis:.1f}" if fps_vis is not None else ""
        display_text = (
            f"Frame Number={frame_number} Number of Objects={num_rects}{fps_text} CAM: 4mm\n"
            f"{display_text2}"
        )
        py_nvosd_text_params.display_text = display_text
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 3 if RESOLUTION[0] <= 1280 else 20
        py_nvosd_text_params.font_params.font_color.set(1.0, 0.0, 0.0, 1.0)
        py_nvosd_text_params.set_bg_clr = 1
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.1)

        # Draw line from frame center to virtual pixel (kept from python_test.py)
        display_meta.num_lines = 1
        line_params = display_meta.line_params[0]
        screen_center_x = RESOLUTION[0] // 2
        screen_center_y = RESOLUTION[1] // 2
        virt_x = float(_safe_get(redis, "pixel_x_virt", "0"))
        virt_y = float(_safe_get(redis, "pixel_y_virt", "0"))
        line_params.x1 = screen_center_x
        line_params.y1 = screen_center_y
        line_params.x2 = int(virt_x)
        line_params.y2 = int(virt_y)
        line_params.line_width = 2
        line_params.line_color.set(1.0, 1.0, 0.0, 0.8)

        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        # Publish to ROS 2
        if ros_node is not None:
            try:
                ros_node.publish_frame(frame_number, current_fps or 0.0, bboxes, target_bbox)
            except Exception as e:
                print(f"[ROS2 publish error] {e}")

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def create_pipeline():
    """Create the same DeepStream pipeline defined in python_test.py."""
    Gst.init(None)

    pipeline = Gst.Pipeline()

    source = Gst.ElementFactory.make("nvarguscamerasrc", "camera-source")
    caps_filter = Gst.ElementFactory.make("capsfilter", "caps-filter")
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinference-engine")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    tee = Gst.ElementFactory.make("tee", "tee-hud")
    queue1 = Gst.ElementFactory.make("queue", "queue-display")
    queue2 = Gst.ElementFactory.make("queue", "queue-hud-record")

    if DISPLAY:
        display_sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")

    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder-hud")
    parser = Gst.ElementFactory.make("h264parse", "parser-hud")
    container = Gst.ElementFactory.make("mp4mux", "container-hud")
    file_sink = Gst.ElementFactory.make("filesink", "file-sink-hud")

    tee_raw_split = Gst.ElementFactory.make("tee", "tee-raw-split")
    queue_raw = Gst.ElementFactory.make("queue", "queue-raw-record")
    encoder_raw = Gst.ElementFactory.make("nvv4l2h264enc", "encoder-raw")
    parser_raw = Gst.ElementFactory.make("h264parse", "parser-raw")
    container_raw = Gst.ElementFactory.make("mp4mux", "container-raw")
    file_sink_raw = Gst.ElementFactory.make("filesink", "file-sink-raw")

    if not all([source, streammux, pgie, nvvidconv, nvosd, tee]):
        raise RuntimeError("Unable to create essential elements")
    if not all([queue1, queue2, encoder, parser, container, file_sink]):
        raise RuntimeError("Unable to create HUD recording elements")
    if not all([tee_raw_split, queue_raw, encoder_raw, parser_raw, container_raw, file_sink_raw]):
        raise RuntimeError("Unable to create Raw recording elements")

    source.set_property("sensor-id", 0)
    source.set_property("tnr-mode", 0)
    source.set_property("tnr-strength", 0)
    source.set_property("ee-mode", 0)
    source.set_property("ee-strength", 0)
    caps = Gst.Caps.from_string(
        f"video/x-raw(memory:NVMM), width={RESOLUTION[0]}, height={RESOLUTION[1]}, format=NV12, framerate={FPS}/1"
    )
    caps_filter.set_property("caps", caps)
    nvvidconv.set_property("flip-method", 2)

    streammux.set_property("width", RESOLUTION[0])
    streammux.set_property("height", RESOLUTION[1])
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 4_000_000)

    pgie.set_property("config-file-path", "/home/ituarc/DeepStream-Yolo/config_infer_primary_yolo11.txt")

    if DISPLAY:
        display_sink.set_property("sync", False)

    queue1.set_property("max-size-buffers", 1)
    queue1.set_property("leaky", 2)
    queue2.set_property("max-size-buffers", 10)
    queue2.set_property("max-size-time", 0)
    queue2.set_property("max-size-bytes", 0)
    queue_raw.set_property("max-size-buffers", 10)
    encoder.set_property("bitrate", 4_000_000)
    encoder_raw.set_property("bitrate", 4_000_000)

    # Build output paths exactly like deepstream_ros2_publisher.py
    os.makedirs(VIDEO_LOG_DIR, exist_ok=True)
    log_id = 0
    try:
        with open(LOG_ID_FILE, "r") as f:
            log_id = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        pass

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    annotated_path = os.path.join(
        VIDEO_LOG_DIR,
        f"{log_id:03d}_deepstream_flight_video_annotated_{ts}.mp4",
    )
    raw_path = os.path.join(
        VIDEO_LOG_DIR,
        f"{log_id:03d}_deepstream_flight_video_annotated_{ts}_raw.mp4",
    )

    file_sink.set_property("location", annotated_path)
    file_sink_raw.set_property("location", raw_path)

    pipeline.add(source)
    pipeline.add(caps_filter)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(tee_raw_split)
    pipeline.add(nvosd)
    pipeline.add(tee)
    pipeline.add(queue1)
    pipeline.add(queue2)
    if DISPLAY:
        pipeline.add(display_sink)
    pipeline.add(encoder)
    pipeline.add(parser)
    pipeline.add(container)
    pipeline.add(file_sink)
    pipeline.add(queue_raw)
    pipeline.add(encoder_raw)
    pipeline.add(parser_raw)
    pipeline.add(container_raw)
    pipeline.add(file_sink_raw)

    source.link(caps_filter)
    sinkpad = streammux.get_request_pad("sink_0")
    srcpad = caps_filter.get_static_pad("src")
    srcpad.link(sinkpad)

    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(tee_raw_split)

    tee_raw_split.link(queue_raw)
    queue_raw.link(encoder_raw)
    encoder_raw.link(parser_raw)
    parser_raw.link(container_raw)
    container_raw.link(file_sink_raw)

    tee_raw_split.link(nvosd)
    nvosd.link(tee)

    tee.link(queue1)
    if DISPLAY:
        queue1.link(display_sink)

    tee.link(queue2)
    queue2.link(encoder)
    encoder.link(parser)
    parser.link(container)
    container.link(file_sink)

    osdsinkpad = nvosd.get_static_pad("sink")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    return pipeline


def main():
    global ros_node, start_time
    start_time = time.time()

    rclpy.init()
    ros_node = BBoxPublisherNode()
    ros_thread = threading.Thread(target=lambda: rclpy.spin(ros_node), daemon=True)
    ros_thread.start()

    pipeline = create_pipeline()
    loop = GObject.MainLoop()

    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(bus, message):
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, Debug: {debug}")
            loop.quit()
        elif message.type == Gst.MessageType.EOS:
            print("End of stream")
            loop.quit()
        elif _safe_get(redis, "stop", "False") == "True":
            print("Stopping pipeline as per Redis command")
            loop.quit()

    bus.connect("message", on_message)

    print("Starting pipeline...")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("Failed to start pipeline")
        rclpy.shutdown()
        return

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
    finally:
        print("Sending EOS event to the pipeline...")
        pipeline.send_event(Gst.Event.new_eos())
        time.sleep(2)

        total_time = time.time() - start_time
        if total_time > 0:
            print(f"Total frames processed: {frame_count}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Average FPS: {frame_count / total_time:.2f}")
        redis.r.set("stop", "False")

        print("Setting pipeline to NULL state...")
        pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped.")

        rclpy.shutdown()
        ros_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
