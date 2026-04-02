#!/usr/bin/env python3

import configparser
import os
import sys
import time
import threading
import signal
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
import pyds
import cv2
import numpy as np
import json
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy._rclpy_pybind11 import RCLError
from geometry_msgs.msg import Point, PointStamped
from mavros_msgs.msg import State
from sensor_msgs.msg import RegionOfInterest
from redis_helper import RedisHelper
from moving_average import MovingAverage
from project_paths import (
    PYTHON_TEST_OUTPUT_DIR,
    ROOT_DEEPSTREAM_APP_CONFIG,
    ROOT_PGIE_CONFIG_YOLO26,
    ROOT_TRACKER_FALLBACK_CONFIG,
)

ma = MovingAverage(60)

redis = RedisHelper()
ros_node = None
shutdown_event = threading.Event()

focal_x = 1238.10428
focal_y = 1238.78782  # Set focal length y
c_x = 960
c_y = 540

RESOLUTION = (1920, 1080)  # Set resolution as per your config

display = os.getenv("PYTHON_TEST_DISPLAY", "0") == "1"
ENABLE_HUD_RECORDING = os.getenv("PYTHON_TEST_RECORD_HUD", "1") == "1"
ENABLE_RAW_RECORDING = os.getenv("PYTHON_TEST_RECORD_RAW", "1") == "1"
ENABLE_FULL_HUD = os.getenv("PYTHON_TEST_FULL_HUD", "0") == "1"
DEBUG_PRINT_DETECTIONS = os.getenv("PYTHON_TEST_DEBUG_DETECTIONS", "0") == "1"
ENABLE_SMALL_TARGET_PREPROCESS = os.getenv("PYTHON_TEST_PREPROCESS", "0") == "1"
ENABLE_DYNAMIC_ROI_INFERENCE = os.getenv("PYTHON_TEST_DYNAMIC_ROI", "1") == "1"
PREPROCESS_MODE = os.getenv("PYTHON_TEST_PREPROCESS_MODE", "roi").strip().lower()
PREPROCESS_CLAHE_CLIP_LIMIT = float(os.getenv("PYTHON_TEST_PREPROCESS_CLAHE", "1.8"))
PREPROCESS_CLAHE_GRID = max(1, int(os.getenv("PYTHON_TEST_PREPROCESS_TILE", "8")))
PREPROCESS_GAMMA = max(0.1, float(os.getenv("PYTHON_TEST_PREPROCESS_GAMMA", "1.2")))
PREPROCESS_SHARPEN = max(0.0, float(os.getenv("PYTHON_TEST_PREPROCESS_SHARPEN", "0.35")))
PREPROCESS_INTERVAL = max(1, int(os.getenv("PYTHON_TEST_PREPROCESS_INTERVAL", "1")))
PREPROCESS_ROI_SIZE = max(32, int(os.getenv("PYTHON_TEST_PREPROCESS_ROI", "320")))
CAMERA_TNR_MODE = int(os.getenv("PYTHON_TEST_CAMERA_TNR_MODE", "1"))
CAMERA_TNR_STRENGTH = float(os.getenv("PYTHON_TEST_CAMERA_TNR_STRENGTH", "0.15"))
CAMERA_EE_MODE = int(os.getenv("PYTHON_TEST_CAMERA_EE_MODE", "1"))
CAMERA_EE_STRENGTH = float(os.getenv("PYTHON_TEST_CAMERA_EE_STRENGTH", "0.2"))
CAMERA_SATURATION = float(os.getenv("PYTHON_TEST_CAMERA_SATURATION", "1.0"))
CAMERA_EXPOSURE_COMPENSATION = float(os.getenv("PYTHON_TEST_CAMERA_EXPOSURE_COMPENSATION", "0.0"))
SOURCE_TYPE = os.getenv("PYTHON_TEST_SOURCE", "camera").strip().lower()
VIDEO_SOURCE_PATH = os.getenv("PYTHON_TEST_VIDEO_PATH", "").strip()

FPS = 40
INFERENCE_ROI_TOPIC = os.getenv(
    "PYTHON_TEST_ROI_TOPIC", "/interceptor/camera/inference_roi"
).strip() or "/interceptor/camera/inference_roi"
INFERENCE_ROI_TIMEOUT_SEC = max(
    0.1, float(os.getenv("PYTHON_TEST_ROI_TIMEOUT", "0.5"))
)
FRAME_INFERENCE_CONTEXT_LIMIT = max(32, FPS * 4)
APP_CONFIG_PATH = str(ROOT_DEEPSTREAM_APP_CONFIG)
TRACKER_CONFIG_PATH = APP_CONFIG_PATH
TRACKER_DEFAULT_WIDTH = 1920
TRACKER_DEFAULT_HEIGHT = 1080
TRACKER_DEFAULT_GPU_ID = 0
TRACKER_LIB_FILE = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
TRACKER_ACCURACY_CONFIG = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_accuracy.yml"
TRACKER_ACCURACY_FALLBACK_CONFIG = str(ROOT_TRACKER_FALLBACK_CONFIG)
TRACKER_ACCURACY_REID_MODEL = "/opt/nvidia/deepstream/deepstream/samples/models/Tracker/resnet50_market1501.etlt"
STREAMMUX_BATCHED_PUSH_TIMEOUT_USEC = 40_000
UNTRACKED_OBJECT_ID = (1 << 64) - 1
SELECTED_TRACK_MAX_MISSES = max(2, FPS // 8)

HUD_DEFAULTS = {
    "offset": "0",
    "depth_virt": "0",
    "drone_mode": "UNKNOWN",
    "distance": "0",
    "attitude": "0",
    "velocity": "0",
    "interceptor_location": "0",
    "target_location": "0",
    "pixel_errors": "0",
    "filter_roll": "0",
    "kp_yaw": "0",
    "cbf": "0",
    "roll_rate": "0",
    "pitch_rate": "0",
    "yaw_rate": "0",
    "throttle_command": "0",
}
HUD_KEYS = tuple(HUD_DEFAULTS.keys())

P_BAR_TOPIC = "/interceptor/camera/target_pbar" #########################################################################
P_BAR_FRAME_ID = "normalized_image_coordinates"

DKF_BAR_TOPIC = "/eskf_reduced/pbar"
DKF_TIMEOUT_SEC = 0.5
DKF_CROSS_HALF_SIZE = 12

dkf_lock = threading.Lock()
dkf_overlay_state = {
    "pixel_x": None,
    "pixel_y": None,
    "pbar_x": None,
    "pbar_y": None,
    "timestamp": 0.0,
}

roi_lock = threading.Lock()
inference_roi_state = {
    "x_offset": None,
    "y_offset": None,
    "width": None,
    "height": None,
    "timestamp": 0.0,
}

frame_inference_lock = threading.Lock()
frame_inference_context = {}

# Global variables for FPS calculation
frame_count = 0
start_time = time.time()
fps_interval = 1  # Calculate FPS every 30 frames
selected_track_id = None
selected_track_misses = 0
hud_cache = HUD_DEFAULTS.copy()
hud_cache_frame = -1
preprocess_clahe = cv2.createCLAHE(
    clipLimit=PREPROCESS_CLAHE_CLIP_LIMIT,
    tileGridSize=(PREPROCESS_CLAHE_GRID, PREPROCESS_CLAHE_GRID),
)
gamma_inverse = 1.0 / PREPROCESS_GAMMA
preprocess_gamma_lut = np.array(
    [((i / 255.0) ** gamma_inverse) * 255 for i in range(256)],
    dtype=np.uint8,
)
preprocess_warning_printed = False


class NormalizedTargetPublisher(Node):
    def __init__(self):
        super().__init__("python_test_normalized_target_publisher")
        self.target_pub = self.create_publisher(PointStamped, P_BAR_TOPIC, 10)
        self.dkf_sub = self.create_subscription(Point, DKF_BAR_TOPIC, self.dkf_callback, 10)
        self.roi_sub = self.create_subscription(
            RegionOfInterest, INFERENCE_ROI_TOPIC, self.roi_callback, 10
        )
        self.mavros_state_sub = self.create_subscription(
            State, "/mavros/state", self.mavros_state_callback, 10
        )
        self.was_armed = False
        self.shutdown_requested = False

    def publish_target(self, x_bar, y_bar):
        if shutdown_event.is_set() or not rclpy.ok():
            return

        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = P_BAR_FRAME_ID
        msg.point.x = float(x_bar)
        msg.point.y = float(y_bar)
        msg.point.z = 0.0
        try:
            self.target_pub.publish(msg)
        except RCLError:
            # The pipeline can still drain a few frames during shutdown.
            pass

    def dkf_callback(self, msg):
        if not np.isfinite(msg.x) or not np.isfinite(msg.y):
            with dkf_lock:
                dkf_overlay_state["pixel_x"] = None
                dkf_overlay_state["pixel_y"] = None
                dkf_overlay_state["pbar_x"] = None
                dkf_overlay_state["pbar_y"] = None
                dkf_overlay_state["timestamp"] = 0.0
            return

        pixel_x = int(round(msg.x * focal_x + c_x))
        pixel_y = int(round(msg.y * focal_y + c_y))

        pixel_x = max(0, min(RESOLUTION[0] - 1, pixel_x))
        pixel_y = max(0, min(RESOLUTION[1] - 1, pixel_y))

        with dkf_lock:
            dkf_overlay_state["pixel_x"] = pixel_x
            dkf_overlay_state["pixel_y"] = pixel_y
            dkf_overlay_state["pbar_x"] = float(msg.x)
            dkf_overlay_state["pbar_y"] = float(msg.y)
            dkf_overlay_state["timestamp"] = time.time()

        redis.r.set("dkf_pixel_x", pixel_x)
        redis.r.set("dkf_pixel_y", pixel_y)
        redis.r.set("dkf_pbar_x", float(msg.x))
        redis.r.set("dkf_pbar_y", float(msg.y))

    def roi_callback(self, msg):
        if msg.width <= 0 or msg.height <= 0:
            clear_latest_inference_roi()
            return

        update_latest_inference_roi(
            int(msg.x_offset),
            int(msg.y_offset),
            int(msg.width),
            int(msg.height),
        )

    def mavros_state_callback(self, msg):
        if shutdown_event.is_set() or self.shutdown_requested:
            self.was_armed = msg.armed
            return

        if msg.armed:
            self.was_armed = True
            return

        if self.was_armed and not msg.armed:
            self.shutdown_requested = True
            redis.r.set("stop", "True")
            self.get_logger().info(
                "Vehicle disarmed after arming. Sending SIGINT to stop python_test.py."
            )
            os.kill(os.getpid(), signal.SIGINT)


def get_latest_dkf_pixel():
    with dkf_lock:
        timestamp = dkf_overlay_state["timestamp"]
        if timestamp <= 0.0 or (time.time() - timestamp) > DKF_TIMEOUT_SEC:
            return None

        return (
            dkf_overlay_state["pixel_x"],
            dkf_overlay_state["pixel_y"],
        )


def clear_latest_inference_roi():
    with roi_lock:
        inference_roi_state["x_offset"] = None
        inference_roi_state["y_offset"] = None
        inference_roi_state["width"] = None
        inference_roi_state["height"] = None
        inference_roi_state["timestamp"] = 0.0


def update_latest_inference_roi(x_offset, y_offset, width, height):
    with roi_lock:
        inference_roi_state["x_offset"] = int(x_offset)
        inference_roi_state["y_offset"] = int(y_offset)
        inference_roi_state["width"] = int(width)
        inference_roi_state["height"] = int(height)
        inference_roi_state["timestamp"] = time.time()


def _clamp_roi_bounds(left, top, width, height, frame_width, frame_height):
    if width <= 0 or height <= 0 or frame_width <= 0 or frame_height <= 0:
        return None

    left = int(round(left))
    top = int(round(top))
    width = int(round(width))
    height = int(round(height))

    left = max(0, min(frame_width - 1, left))
    top = max(0, min(frame_height - 1, top))
    right = min(frame_width, left + width)
    bottom = min(frame_height, top + height)

    if right <= left or bottom <= top:
        return None

    return left, top, right, bottom


def get_latest_inference_roi_bounds(frame_width, frame_height):
    if not ENABLE_DYNAMIC_ROI_INFERENCE:
        return None, None

    with roi_lock:
        timestamp = inference_roi_state["timestamp"]
        x_offset = inference_roi_state["x_offset"]
        y_offset = inference_roi_state["y_offset"]
        width = inference_roi_state["width"]
        height = inference_roi_state["height"]

    if timestamp <= 0.0 or (time.time() - timestamp) > INFERENCE_ROI_TIMEOUT_SEC:
        return None, (
            f"ROI topic missing or stale on {INFERENCE_ROI_TOPIC}; full-frame inference"
        )

    bounds = _clamp_roi_bounds(
        x_offset, y_offset, width, height, frame_width, frame_height
    )
    if bounds is None:
        return None, (
            f"ROI topic invalid on {INFERENCE_ROI_TOPIC}; full-frame inference"
        )

    return bounds, None


def _store_frame_inference_context(frame_number, context):
    with frame_inference_lock:
        frame_inference_context[frame_number] = context

        while len(frame_inference_context) > FRAME_INFERENCE_CONTEXT_LIMIT:
            oldest_frame = min(frame_inference_context)
            frame_inference_context.pop(oldest_frame, None)


def _get_frame_inference_context(frame_number):
    with frame_inference_lock:
        return frame_inference_context.get(frame_number)


def _pop_restore_frame(frame_number):
    with frame_inference_lock:
        context = frame_inference_context.get(frame_number)
        if context is None:
            return None
        return context.pop("restore_frame", None)


def _discard_frame_inference_context(frame_number):
    with frame_inference_lock:
        frame_inference_context.pop(frame_number, None)
        stale_frames = [
            stored_frame
            for stored_frame in frame_inference_context
            if stored_frame < (frame_number - FRAME_INFERENCE_CONTEXT_LIMIT)
        ]
        for stale_frame in stale_frames:
            frame_inference_context.pop(stale_frame, None)


def get_log_number(path="/home/ituarc/ros2_ws/src/thermal_guidance/logs/last_log_id.txt"):
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except Exception as e:
        print(f"Error reading log number: {e}")
        return 0


def decode_redis_value(value, default="0"):
    if value is None:
        return default

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return default

    return str(value)


def safe_redis_get_float(key, default=0.0):
    try:
        return float(decode_redis_value(redis.r.get(key), str(default)))
    except (TypeError, ValueError):
        return float(default)
    except Exception:
        return float(default)


def refresh_hud_cache(frame_number):
    global hud_cache, hud_cache_frame

    if not ENABLE_FULL_HUD:
        return

    if hud_cache_frame >= 0 and (frame_number - hud_cache_frame) < max(1, FPS // 10):
        return

    try:
        values = redis.r.mget(HUD_KEYS)
    except Exception:
        values = [None] * len(HUD_KEYS)

    hud_cache = {
        key: decode_redis_value(value, HUD_DEFAULTS[key])
        for key, value in zip(HUD_KEYS, values)
    }
    hud_cache_frame = frame_number


def _get_preprocess_roi_bounds(frame_width, frame_height):
    roi_size = min(PREPROCESS_ROI_SIZE, frame_width, frame_height)
    half_roi = roi_size // 2

    roi_center = get_latest_dkf_pixel()
    if roi_center is None:
        virt_x = safe_redis_get_float("pixel_x_virt", float("nan"))
        virt_y = safe_redis_get_float("pixel_y_virt", float("nan"))
        if np.isfinite(virt_x) and np.isfinite(virt_y):
            roi_center = (int(round(virt_x)), int(round(virt_y)))

    if roi_center is None:
        return None

    center_x = int(max(0, min(frame_width - 1, roi_center[0])))
    center_y = int(max(0, min(frame_height - 1, roi_center[1])))
    left = max(0, center_x - half_roi)
    top = max(0, center_y - half_roi)
    right = min(frame_width, left + roi_size)
    bottom = min(frame_height, top + roi_size)
    left = max(0, right - roi_size)
    top = max(0, bottom - roi_size)
    return left, top, right, bottom


def enhance_small_target_rgba_inplace(rgba_frame, force_full_frame=False):
    """
    Boost local contrast for tiny airborne targets before primary inference.
    This is intentionally mild so it helps the drone stand out without
    destabilizing the detector with an overly synthetic image.
    """
    if PREPROCESS_MODE == "roi" and not force_full_frame:
        roi_bounds = _get_preprocess_roi_bounds(rgba_frame.shape[1], rgba_frame.shape[0])
        if roi_bounds is None:
            return
        left, top, right, bottom = roi_bounds
        target_region = rgba_frame[top:bottom, left:right, :3]
    else:
        target_region = rgba_frame[:, :, :3]

    rgb_frame = np.ascontiguousarray(target_region)
    lab_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_frame)
    l_channel = preprocess_clahe.apply(l_channel)
    l_channel = cv2.LUT(l_channel, preprocess_gamma_lut)
    lab_frame = cv2.merge((l_channel, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(lab_frame, cv2.COLOR_LAB2RGB)

    if PREPROCESS_SHARPEN > 0.0:
        blurred = cv2.GaussianBlur(enhanced_rgb, (0, 0), sigmaX=1.1, sigmaY=1.1)
        enhanced_rgb = cv2.addWeighted(
            enhanced_rgb,
            1.0 + PREPROCESS_SHARPEN,
            blurred,
            -PREPROCESS_SHARPEN,
            0,
        )

    np.copyto(target_region, enhanced_rgb)


def preprocess_sink_pad_buffer_probe(pad, info, u_data):
    global ENABLE_DYNAMIC_ROI_INFERENCE, ENABLE_SMALL_TARGET_PREPROCESS
    global preprocess_warning_printed

    if shutdown_event.is_set():
        return Gst.PadProbeReturn.OK

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_context = {
            "frame_width": 0,
            "frame_height": 0,
            "roi_bounds": None,
            "used_roi_inference": False,
            "warning_text": None,
            "restore_frame": None,
        }

        try:
            rgba_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            frame_height, frame_width = rgba_frame.shape[:2]
            frame_context["frame_width"] = frame_width
            frame_context["frame_height"] = frame_height

            roi_bounds, warning_text = get_latest_inference_roi_bounds(
                frame_width, frame_height
            )
            frame_context["roi_bounds"] = roi_bounds
            frame_context["warning_text"] = warning_text

            if roi_bounds is not None:
                left, top, right, bottom = roi_bounds
                roi_width = right - left
                roi_height = bottom - top
                frame_context["used_roi_inference"] = True

                if left != 0 or top != 0 or roi_width != frame_width or roi_height != frame_height:
                    frame_context["restore_frame"] = np.array(rgba_frame, copy=True)
                    roi_frame = np.ascontiguousarray(rgba_frame[top:bottom, left:right, :])
                    resized_roi = cv2.resize(
                        roi_frame,
                        (frame_width, frame_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    np.copyto(rgba_frame, resized_roi)

            should_run_small_target_preprocess = (
                ENABLE_SMALL_TARGET_PREPROCESS
                and (
                    PREPROCESS_INTERVAL <= 1
                    or frame_meta.frame_num % PREPROCESS_INTERVAL == 0
                )
            )
            if should_run_small_target_preprocess:
                enhance_small_target_rgba_inplace(
                    rgba_frame, force_full_frame=frame_context["used_roi_inference"]
                )
        except Exception as exc:
            if not preprocess_warning_printed:
                print(f"Inference preprocess path disabled after error: {exc}")
                preprocess_warning_printed = True
            frame_context["warning_text"] = (
                "Preprocess error; full-frame inference fallback active"
            )
            frame_context["roi_bounds"] = None
            frame_context["used_roi_inference"] = False
            frame_context["restore_frame"] = None
            ENABLE_DYNAMIC_ROI_INFERENCE = False
            ENABLE_SMALL_TARGET_PREPROCESS = False
        finally:
            _store_frame_inference_context(frame_meta.frame_num, frame_context)
            try:
                pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            except Exception:
                pass

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def _apply_bbox_to_object_meta(obj_meta, left, top, width, height):
    obj_meta.rect_params.left = float(left)
    obj_meta.rect_params.top = float(top)
    obj_meta.rect_params.width = float(width)
    obj_meta.rect_params.height = float(height)

    for bbox_info_name in ("detector_bbox_info", "tracker_bbox_info"):
        try:
            bbox_info = getattr(obj_meta, bbox_info_name)
            bbox_coords = bbox_info.org_bbox_coords
            bbox_coords.left = float(left)
            bbox_coords.top = float(top)
            bbox_coords.width = float(width)
            bbox_coords.height = float(height)
        except Exception:
            continue


def _remap_bbox_from_roi(obj_meta, roi_bounds, frame_width, frame_height):
    left, top, right, bottom = roi_bounds
    roi_width = max(1.0, float(right - left))
    roi_height = max(1.0, float(bottom - top))
    scale_x = roi_width / max(1.0, float(frame_width))
    scale_y = roi_height / max(1.0, float(frame_height))

    new_left = left + (obj_meta.rect_params.left * scale_x)
    new_top = top + (obj_meta.rect_params.top * scale_y)
    new_width = obj_meta.rect_params.width * scale_x
    new_height = obj_meta.rect_params.height * scale_y

    new_left = max(0.0, min(float(frame_width - 1), new_left))
    new_top = max(0.0, min(float(frame_height - 1), new_top))
    new_width = max(1.0, min(float(frame_width) - new_left, new_width))
    new_height = max(1.0, min(float(frame_height) - new_top, new_height))

    _apply_bbox_to_object_meta(obj_meta, new_left, new_top, new_width, new_height)


def pgie_src_pad_buffer_probe(pad, info, u_data):
    if shutdown_event.is_set():
        return Gst.PadProbeReturn.OK

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_context = _get_frame_inference_context(frame_meta.frame_num)
        if frame_context and frame_context.get("used_roi_inference"):
            roi_bounds = frame_context.get("roi_bounds")
            frame_width = frame_context.get("frame_width", RESOLUTION[0])
            frame_height = frame_context.get("frame_height", RESOLUTION[1])

            if roi_bounds is not None:
                l_obj = frame_meta.obj_meta_list
                while l_obj is not None:
                    try:
                        obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    except StopIteration:
                        break

                    _remap_bbox_from_roi(
                        obj_meta, roi_bounds, frame_width, frame_height
                    )

                    try:
                        l_obj = l_obj.next
                    except StopIteration:
                        break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def tracker_src_restore_buffer_probe(pad, info, u_data):
    if shutdown_event.is_set():
        return Gst.PadProbeReturn.OK

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        restore_frame = _pop_restore_frame(frame_meta.frame_num)
        if restore_frame is not None:
            try:
                rgba_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                np.copyto(rgba_frame, restore_frame)
            finally:
                try:
                    pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                except Exception:
                    pass

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def _resolve_video_uri(video_path):
    if not video_path:
        return ""

    if "://" in video_path:
        return video_path

    return GLib.filename_to_uri(os.path.abspath(video_path), None)


def _on_video_pad_added(decodebin, pad, user_data):
    source_ctx = user_data
    if source_ctx.get("linked"):
        return

    caps = pad.get_current_caps()
    if not caps:
        return

    structure = caps.get_structure(0)
    if not structure:
        return

    media_type = structure.get_name()
    if not media_type.startswith("video/"):
        return

    queue = Gst.ElementFactory.make("queue", "file-source-queue")
    conv = Gst.ElementFactory.make("nvvideoconvert", "file-source-convert")
    caps_filter = Gst.ElementFactory.make("capsfilter", "file-source-caps")
    if not all([queue, conv, caps_filter]):
        print("Unable to create video source decode chain")
        return

    conv.set_property("compute-hw", 1)
    caps_filter.set_property(
        "caps",
        Gst.Caps.from_string(
            f"video/x-raw(memory:NVMM), format=NV12, width={RESOLUTION[0]}, height={RESOLUTION[1]}"
        ),
    )

    pipeline = source_ctx["pipeline"]
    streammux = source_ctx["streammux"]
    pipeline.add(queue)
    pipeline.add(conv)
    pipeline.add(caps_filter)
    queue.sync_state_with_parent()
    conv.sync_state_with_parent()
    caps_filter.sync_state_with_parent()

    if pad.link(queue.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
        print("Failed to link video decodebin pad to queue")
        return

    if not queue.link(conv) or not conv.link(caps_filter):
        print("Failed to link video conversion chain")
        return

    if hasattr(streammux, "request_pad_simple"):
        sinkpad = streammux.request_pad_simple("sink_0")
    else:
        sinkpad = streammux.get_request_pad("sink_0")

    if caps_filter.get_static_pad("src").link(sinkpad) != Gst.PadLinkReturn.OK:
        print("Failed to link video source into streammux")
        return

    source_ctx["linked"] = True


def select_target_bbox(bboxes):
    global selected_track_id, selected_track_misses

    if not bboxes:
        selected_track_id = None
        selected_track_misses = 0
        return None

    tracked_bboxes = {
        bbox["object_id"]: bbox for bbox in bboxes if bbox["object_id"] >= 0
    }

    if selected_track_id is not None:
        selected_bbox = tracked_bboxes.get(selected_track_id)
        if selected_bbox is not None:
            selected_track_misses = 0
            return selected_bbox

        selected_track_misses += 1
        if selected_track_misses > SELECTED_TRACK_MAX_MISSES:
            selected_track_id = None

    candidates = list(tracked_bboxes.values()) if tracked_bboxes else bboxes
    target_bbox = max(candidates, key=lambda bbox: bbox["confidence"], default=None)

    if target_bbox is None or target_bbox["object_id"] < 0:
        selected_track_id = None
        selected_track_misses = 0
    else:
        selected_track_id = target_bbox["object_id"]
        selected_track_misses = 0

    return target_bbox


def calculate_fps():
    """Calculate and return current FPS"""
    global frame_count, start_time
    
    frame_count += 1
    
    if frame_count % fps_interval == 0:
        current_time = time.time()
        elapsed_time = current_time - start_time
        fps = fps_interval / elapsed_time
        start_time = current_time
        return fps


def configure_tracker(tracker, tracker_config_path=TRACKER_CONFIG_PATH):
    def _set_if_supported(prop_name, value):
        if tracker.find_property(prop_name) is not None:
            tracker.set_property(prop_name, value)

    def _resolve_ll_config_file(configured_path=None):
        if configured_path and configured_path != TRACKER_ACCURACY_CONFIG:
            return configured_path

        if os.path.exists(TRACKER_ACCURACY_REID_MODEL):
            return TRACKER_ACCURACY_CONFIG

        print(
            "Tracker ReID sample model is missing at "
            f"{TRACKER_ACCURACY_REID_MODEL}. "
            f"Using {TRACKER_ACCURACY_FALLBACK_CONFIG} instead."
        )
        return TRACKER_ACCURACY_FALLBACK_CONFIG

    tracker_width = TRACKER_DEFAULT_WIDTH
    tracker_height = TRACKER_DEFAULT_HEIGHT
    tracker_gpu_id = TRACKER_DEFAULT_GPU_ID
    tracker_lib_file = TRACKER_LIB_FILE
    tracker_ll_config_file = TRACKER_ACCURACY_CONFIG
    enable_batch_process = 1
    enable_past_frame = None
    compute_hw = None
    display_tracking_id = None

    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    if tracker_config_path and config.read(tracker_config_path) and "tracker" in config:
        tracker_section = config["tracker"]
        tracker_width = tracker_section.getint("tracker-width", fallback=tracker_width)
        tracker_height = tracker_section.getint("tracker-height", fallback=tracker_height)
        tracker_gpu_id = tracker_section.getint("gpu-id", fallback=tracker_gpu_id)
        tracker_lib_file = tracker_section.get("ll-lib-file", fallback=tracker_lib_file)
        tracker_ll_config_file = tracker_section.get(
            "ll-config-file", fallback=tracker_ll_config_file
        )
        enable_batch_process = tracker_section.getint(
            "enable-batch-process", fallback=enable_batch_process
        )
        if "enable-past-frame" in tracker_section:
            enable_past_frame = tracker_section.getint("enable-past-frame")
        if "compute-hw" in tracker_section:
            compute_hw = tracker_section.getint("compute-hw")
        if "display-tracking-id" in tracker_section:
            display_tracking_id = tracker_section.getint("display-tracking-id")

    _set_if_supported("tracker-width", tracker_width)
    _set_if_supported("tracker-height", tracker_height)
    _set_if_supported("gpu-id", tracker_gpu_id)
    _set_if_supported("ll-lib-file", tracker_lib_file)
    _set_if_supported("ll-config-file", _resolve_ll_config_file(tracker_ll_config_file))
    _set_if_supported("enable-batch-process", enable_batch_process)
    if enable_past_frame is not None:
        _set_if_supported("enable-past-frame", enable_past_frame)
    if compute_hw is not None:
        _set_if_supported("compute-hw", compute_hw)
    if display_tracking_id is not None:
        _set_if_supported("display-tracking-id", display_tracking_id)

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """
    Probe function to extract bounding box data, draw a line from center to
    the target bbox, and calculate FPS.
    """
    global ros_node

    if shutdown_event.is_set():
        return Gst.PadProbeReturn.OK

    frame_number = 0
    num_rects = 0
    
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return Gst.PadProbeReturn.OK
        
    current_fps = calculate_fps() or 0.0
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
            
        # 1. Adım: Tüm sınırlayıcı kutuları bir listede topla
        bboxes = []
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            object_id = int(obj_meta.object_id)
            if object_id == UNTRACKED_OBJECT_ID:
                object_id = -1
            
            bboxes.append({
                'object_id': object_id,
                'class_id': obj_meta.class_id,
                'confidence': obj_meta.confidence,
                'left': obj_meta.rect_params.left,
                'top': obj_meta.rect_params.top,
                'width': obj_meta.rect_params.width,
                'height': obj_meta.rect_params.height
            })
            
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        
        # 2. Adım: En yüksek güven skoruna sahip hedef kutusunu bul
        target_bbox = None
        x_bar = None
        y_bar = None
        if DEBUG_PRINT_DETECTIONS:
            for bbox in bboxes:
                print(f"Class={bbox['class_id']}, "
                      f"TrackID={bbox['object_id']}, "
                      f"Confidence={bbox['confidence']:.2f}, "
                      f"BBox=({bbox['left']:.0f},{bbox['top']:.0f},"
                      f"{bbox['width']:.0f},{bbox['height']:.0f})")

        target_bbox = select_target_bbox(bboxes)
        if target_bbox:
            center_x = target_bbox['left'] + target_bbox['width'] / 2
            center_y = target_bbox['top'] + target_bbox['height'] / 2
            # Normalized image coordinates with setting center as zero
            x_bar = (center_x - c_x) / focal_x
            y_bar = (center_y - c_y) / focal_y

        if target_bbox and ros_node is not None:
            ros_node.publish_target(x_bar, y_bar)

        # Redis'i en iyi bulunan kutu ile güncelle
        bbox_payload = json.dumps(target_bbox) if target_bbox else json.dumps({})
        try:
            redis.r.set("bbox", bbox_payload)
            redis.r.mset({
                "frame_number": frame_number,
                "cam_fps": current_fps,
            })
        except Exception:
            pass

        # Görüntü metadatasını al
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        
        # Metin bilgilerini yapılandır (Mevcut kodunuz)
        frame_context = _get_frame_inference_context(frame_number) or {}
        warning_text = frame_context.get("warning_text")
        roi_bounds = frame_context.get("roi_bounds")

        display_meta.num_labels = 2 if warning_text else 1
        py_nvosd_text_params = display_meta.text_params[0]
        refresh_hud_cache(frame_number)
        fps_vis = ma.update(current_fps)
        fps_text = f" FPS: {fps_vis:.1f}" if fps_vis is not None else ""
        display_text = f"Frame Number={frame_number} Number of Objects={num_rects}{fps_text} CAM: 4mm"
        if ENABLE_FULL_HUD:
            display_text += (
                f"\nDrone Mode: {hud_cache['drone_mode']}"
                f"\nDistance: {hud_cache['distance']} m"
                f"\nAttitude: {hud_cache['attitude']}"
                f"\nVelocity: {hud_cache['velocity']} m/s"
                f"\nInterceptor Location: {hud_cache['interceptor_location']}"
                f"\nTarget Location: {hud_cache['target_location']}"
                f"\nPixel Error: {hud_cache['pixel_errors']}"
                f"\nFilter Roll: {hud_cache['filter_roll']}"
                f"\nKp Yaw: {hud_cache['kp_yaw']}"
                f"\nCBF: {hud_cache['cbf']}"
                f"\nRoll Rate: {hud_cache['roll_rate']} rad/s"
                f"\nPitch Rate: {hud_cache['pitch_rate']} rad/s"
                f"\nYaw Rate: {hud_cache['yaw_rate']} rad/s"
                f"\nThrottle Command: {hud_cache['throttle_command']}"
                f"\nOffset: {hud_cache['offset']}, Depth: {hud_cache['depth_virt']}"
            )
        py_nvosd_text_params.display_text = display_text
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 3 if RESOLUTION[0] <= 1280 else 20
        py_nvosd_text_params.font_params.font_color.set(1.0, 0.0, 0.0, 1.0)
        py_nvosd_text_params.set_bg_clr = 1
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.1)

        if warning_text:
            warning_params = display_meta.text_params[1]
            warning_params.display_text = warning_text
            warning_params.x_offset = 10
            warning_params.y_offset = max(10, RESOLUTION[1] - 42)
            warning_params.font_params.font_name = "Serif"
            warning_params.font_params.font_size = 2 if RESOLUTION[0] <= 1280 else 16
            warning_params.font_params.font_color.set(1.0, 1.0, 0.0, 1.0)
            warning_params.set_bg_clr = 1
            warning_params.text_bg_clr.set(0.3, 0.0, 0.0, 0.45)

        dkf_pixel = get_latest_dkf_pixel()
        next_line_index = 0
        line_params = display_meta.line_params[next_line_index]
        next_line_index += 1
        
        # Çizginin başlangıç noktası: ekranın merkezi
        screen_center_x = RESOLUTION[0] // 2
        screen_center_y = RESOLUTION[1] // 2
        
        # Çizginin bitiş noktası: hedef kutusunun merkezi
        virt_x = safe_redis_get_float("pixel_x_virt", screen_center_x)
        virt_y = safe_redis_get_float("pixel_y_virt", screen_center_y)
        bbox_center_x = int(max(0, min(RESOLUTION[0] - 1, virt_x)))
        bbox_center_y = int(max(0, min(RESOLUTION[1] - 1, virt_y)))
        
        # Çizgi parametrelerini ayarla
        line_params.x1 = screen_center_x
        line_params.y1 = screen_center_y
        line_params.x2 = bbox_center_x
        line_params.y2 = bbox_center_y
        line_params.line_width = 2
        line_params.line_color.set(1.0, 1.0, 0.0, 0.8)  # Sarı, hafif şeffaf

        if roi_bounds is not None:
            left, top, right, bottom = roi_bounds
            roi_right = max(left, right - 1)
            roi_bottom = max(top, bottom - 1)
            roi_segments = (
                (left, top, roi_right, top),
                (roi_right, top, roi_right, roi_bottom),
                (roi_right, roi_bottom, left, roi_bottom),
                (left, roi_bottom, left, top),
            )
            for x1, y1, x2, y2 in roi_segments:
                roi_line = display_meta.line_params[next_line_index]
                roi_line.x1 = int(x1)
                roi_line.y1 = int(y1)
                roi_line.x2 = int(x2)
                roi_line.y2 = int(y2)
                roi_line.line_width = 2
                roi_line.line_color.set(0.0, 0.8, 1.0, 0.9)
                next_line_index += 1

        if dkf_pixel is not None:
            dkf_pixel_x, dkf_pixel_y = dkf_pixel

            horizontal = display_meta.line_params[next_line_index]
            horizontal.x1 = max(0, dkf_pixel_x - DKF_CROSS_HALF_SIZE)
            horizontal.y1 = dkf_pixel_y
            horizontal.x2 = min(RESOLUTION[0] - 1, dkf_pixel_x + DKF_CROSS_HALF_SIZE)
            horizontal.y2 = dkf_pixel_y
            horizontal.line_width = 3
            horizontal.line_color.set(0.0, 1.0, 0.0, 1.0)
            next_line_index += 1

            vertical = display_meta.line_params[next_line_index]
            vertical.x1 = dkf_pixel_x
            vertical.y1 = max(0, dkf_pixel_y - DKF_CROSS_HALF_SIZE)
            vertical.x2 = dkf_pixel_x
            vertical.y2 = min(RESOLUTION[1] - 1, dkf_pixel_y + DKF_CROSS_HALF_SIZE)
            vertical.line_width = 3
            vertical.line_color.set(0.0, 1.0, 0.0, 1.0)
            next_line_index += 1

        display_meta.num_lines = next_line_index

        # Metadatayı (metin ve çizgi) kareye ekle
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        _discard_frame_inference_context(frame_number)
            
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    
    return Gst.PadProbeReturn.OK

def create_pipeline():
    """
    Create a DeepStream pipeline similar to your deepstream-app configuration
    """
    # Initialize GStreamer
    Gst.init(None)
    
    # Create pipeline
    pipeline = Gst.Pipeline()
    
    # --- Original Elements ---
    source = None
    caps_filter = None
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    preprocess_enabled = ENABLE_DYNAMIC_ROI_INFERENCE or ENABLE_SMALL_TARGET_PREPROCESS
    preprocess_convert = (
        Gst.ElementFactory.make("nvvideoconvert", "preprocess-convert")
        if preprocess_enabled
        else None
    )
    preprocess_caps = (
        Gst.ElementFactory.make("capsfilter", "preprocess-caps")
        if preprocess_enabled
        else None
    )
    pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinference-engine")
    tracker = Gst.ElementFactory.make("nvtracker", "object-tracker")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    
    # Tee for splitting after OSD (for display and optional HUD recording)
    tee = Gst.ElementFactory.make("tee", "tee-hud") if (display and ENABLE_HUD_RECORDING) else None
    queue1 = Gst.ElementFactory.make("queue", "queue-display") if display else None
    queue2 = Gst.ElementFactory.make("queue", "queue-hud-record") if ENABLE_HUD_RECORDING else None
    
    # Display sink
    display_sink = None
    if display:
        print("Creating nv3dsink \n")
        display_sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")

    # HUD video sink elements
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder-hud") if ENABLE_HUD_RECORDING else None
    parser = Gst.ElementFactory.make("h264parse", "parser-hud") if ENABLE_HUD_RECORDING else None
    container = Gst.ElementFactory.make("mp4mux", "container-hud") if ENABLE_HUD_RECORDING else None
    file_sink = Gst.ElementFactory.make("filesink", "file-sink-hud") if ENABLE_HUD_RECORDING else None

    # --- YENİ: Ham Video Kaydı İçin Elemanlar ---
    tee_raw_split = Gst.ElementFactory.make("tee", "tee-raw-split") if ENABLE_RAW_RECORDING else None
    queue_raw = Gst.ElementFactory.make("queue", "queue-raw-record") if ENABLE_RAW_RECORDING else None
    encoder_raw = Gst.ElementFactory.make("nvv4l2h264enc", "encoder-raw") if ENABLE_RAW_RECORDING else None
    parser_raw = Gst.ElementFactory.make("h264parse", "parser-raw") if ENABLE_RAW_RECORDING else None
    container_raw = Gst.ElementFactory.make("mp4mux", "container-raw") if ENABLE_RAW_RECORDING else None
    file_sink_raw = Gst.ElementFactory.make("filesink", "file-sink-raw") if ENABLE_RAW_RECORDING else None
    null_sink = Gst.ElementFactory.make("fakesink", "null-sink") if not display and not ENABLE_HUD_RECORDING else None

    if SOURCE_TYPE == "camera":
        source = Gst.ElementFactory.make("nvarguscamerasrc", "camera-source")
        caps_filter = Gst.ElementFactory.make("capsfilter", "caps-filter")
    elif SOURCE_TYPE == "video":
        video_uri = _resolve_video_uri(VIDEO_SOURCE_PATH)
        if not video_uri:
            sys.stderr.write(
                "PYTHON_TEST_SOURCE=video requires PYTHON_TEST_VIDEO_PATH to be set\n"
            )
            sys.exit(1)

        source = Gst.ElementFactory.make("uridecodebin", "video-decodebin")
        if source is None:
            sys.stderr.write("Unable to create video source elements\n")
            sys.exit(1)

        source.set_property("uri", video_uri)
        source.connect(
            "pad-added",
            _on_video_pad_added,
            {"pipeline": pipeline, "streammux": streammux, "linked": False},
        )
    else:
        sys.stderr.write(
            f"Unsupported PYTHON_TEST_SOURCE='{SOURCE_TYPE}'. Use 'camera' or 'video'.\n"
        )
        sys.exit(1)

    if not all([source, streammux, pgie, tracker, nvvidconv, nvosd]):
        sys.stderr.write("Unable to create essential elements\n")
        sys.exit(1)

    if SOURCE_TYPE == "camera" and caps_filter is None:
        sys.stderr.write("Unable to create camera caps filter\n")
        sys.exit(1)

    if preprocess_enabled and not all([preprocess_convert, preprocess_caps]):
        sys.stderr.write("Unable to create preprocessing elements\n")
        sys.exit(1)
    
    if ENABLE_HUD_RECORDING and not all([queue2, encoder, parser, container, file_sink]):
        sys.stderr.write("Unable to create HUD recording elements\n")
        sys.exit(1)
    
    # --- YENİ: Ham Kayıt Elemanları için Kontrol ---
    if ENABLE_RAW_RECORDING and not all([tee_raw_split, queue_raw, encoder_raw, parser_raw, container_raw, file_sink_raw]):
        sys.stderr.write("Unable to create Raw recording elements\n")
        sys.exit(1)

    # --- Özellikleri Ayarlama ---
    if SOURCE_TYPE == "camera":
        source.set_property("sensor-id", 0)
        source.set_property("tnr-mode", CAMERA_TNR_MODE)
        source.set_property("tnr-strength", CAMERA_TNR_STRENGTH)
        source.set_property("ee-mode", CAMERA_EE_MODE)
        source.set_property("ee-strength", CAMERA_EE_STRENGTH)
        source.set_property("saturation", CAMERA_SATURATION)
        source.set_property("exposurecompensation", CAMERA_EXPOSURE_COMPENSATION)
        caps = Gst.Caps.from_string(
            f"video/x-raw(memory:NVMM), width={RESOLUTION[0]}, height={RESOLUTION[1]}, format=NV12, framerate={FPS}/1"
        )
        caps_filter.set_property("caps", caps)
    nvvidconv.set_property("flip-method", 2 if SOURCE_TYPE == "camera" else 0)
    if preprocess_caps is not None:
        preprocess_caps.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw(memory:NVMM), width={RESOLUTION[0]}, height={RESOLUTION[1]}, format=RGBA"
            ),
        )
    
    streammux.set_property("width", RESOLUTION[0])
    streammux.set_property("height", RESOLUTION[1])
    streammux.set_property("batch-size", 1)
    if streammux.find_property("live-source") is not None:
        streammux.set_property("live-source", 1 if SOURCE_TYPE == "camera" else 0)
    streammux.set_property("batched-push-timeout", STREAMMUX_BATCHED_PUSH_TIMEOUT_USEC)
    
    pgie.set_property("config-file-path", str(ROOT_PGIE_CONFIG_YOLO26))
    configure_tracker(tracker)
    if preprocess_enabled:
        print(
            "Inference preprocess path enabled "
            f"(dynamic-roi={ENABLE_DYNAMIC_ROI_INFERENCE}, topic={INFERENCE_ROI_TOPIC}, "
            f"timeout={INFERENCE_ROI_TIMEOUT_SEC}s)"
        )
        if ENABLE_SMALL_TARGET_PREPROCESS:
            print(
                "Small-target preprocess enabled "
                f"(mode={PREPROCESS_MODE}, CLAHE={PREPROCESS_CLAHE_CLIP_LIMIT}, "
                f"tile={PREPROCESS_CLAHE_GRID}, "
                f"gamma={PREPROCESS_GAMMA}, sharpen={PREPROCESS_SHARPEN}, "
                f"interval={PREPROCESS_INTERVAL}, roi={PREPROCESS_ROI_SIZE})"
            )
    if SOURCE_TYPE == "camera":
        print(
            "Camera ISP tuning "
            f"(tnr-mode={CAMERA_TNR_MODE}, tnr-strength={CAMERA_TNR_STRENGTH}, "
            f"ee-mode={CAMERA_EE_MODE}, ee-strength={CAMERA_EE_STRENGTH}, "
            f"saturation={CAMERA_SATURATION}, exposure-comp={CAMERA_EXPOSURE_COMPENSATION})"
        )
    else:
        print(f"Video file source: {video_uri}")
    
    if display:
        display_sink.set_property("sync", False)
    
    if null_sink is not None and null_sink.find_property("sync") is not None:
        null_sink.set_property("sync", False)
    
    if queue1 is not None:
        queue1.set_property("max-size-buffers", 1)
        queue1.set_property("max-size-time", 0)
        queue1.set_property("max-size-bytes", 0)
        queue1.set_property("leaky", 2)
    if queue2 is not None:
        queue2.set_property("max-size-buffers", 4)
        queue2.set_property("max-size-time", 0)
        queue2.set_property("max-size-bytes", 0)
        queue2.set_property("leaky", 2)
    
    # --- YENİ: Ham Kayıt Kuyruğu Özellikleri ---
    if queue_raw is not None:
        queue_raw.set_property("max-size-buffers", 4)
        queue_raw.set_property("max-size-time", 0)
        queue_raw.set_property("max-size-bytes", 0)
        queue_raw.set_property("leaky", 2)
    
    # HUD video kayıt özellikleri
    if encoder is not None:
        encoder.set_property("bitrate", 4000000)
    
    # --- YENİ: Ham Video Kodlayıcı Özellikleri ---
    if encoder_raw is not None:
        encoder_raw.set_property("bitrate", 4000000)
    
    if ENABLE_HUD_RECORDING or ENABLE_RAW_RECORDING:
        log_number = get_log_number()
        if file_sink is not None:
            file_sink.set_property(
                "location",
                str(PYTHON_TEST_OUTPUT_DIR / f"output_4mm_{log_number}.mp4"),
            )
        if file_sink_raw is not None:
            file_sink_raw.set_property(
                "location",
                str(PYTHON_TEST_OUTPUT_DIR / f"output_4mm_{log_number}_raw.mp4"),
            )

    # --- Boru Hattına Eleman Ekleme ---
    for element in [
        source,
        caps_filter,
        streammux,
        preprocess_convert,
        preprocess_caps,
        pgie,
        tracker,
        nvvidconv,
        tee_raw_split,
        nvosd,
        tee,
        queue1,
        queue2,
        display_sink,
        encoder,
        parser,
        container,
        file_sink,
        queue_raw,
        encoder_raw,
        parser_raw,
        container_raw,
        file_sink_raw,
        null_sink,
    ]:
        if element is not None:
            pipeline.add(element)

    # --- Elemanları Birbirine Bağlama ---
    if SOURCE_TYPE == "camera":
        source.link(caps_filter)

        if hasattr(streammux, "request_pad_simple"):
            sinkpad = streammux.request_pad_simple("sink_0")
        else:
            sinkpad = streammux.get_request_pad("sink_0")
        srcpad = caps_filter.get_static_pad("src")
        srcpad.link(sinkpad)
    
    if preprocess_enabled:
        streammux.link(preprocess_convert)
        preprocess_convert.link(preprocess_caps)
        preprocess_caps.link(pgie)
    else:
        streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvvidconv)
    
    if ENABLE_RAW_RECORDING:
        nvvidconv.link(tee_raw_split)
        tee_raw_split.link(queue_raw)
        queue_raw.link(encoder_raw)
        encoder_raw.link(parser_raw)
        parser_raw.link(container_raw)
        container_raw.link(file_sink_raw)
        tee_raw_split.link(nvosd)
    else:
        nvvidconv.link(nvosd)

    if display and ENABLE_HUD_RECORDING:
        nvosd.link(tee)
        tee.link(queue1)
        queue1.link(display_sink)
        tee.link(queue2)
        queue2.link(encoder)
        encoder.link(parser)
        parser.link(container)
        container.link(file_sink)
    elif display:
        nvosd.link(queue1)
        queue1.link(display_sink)
    elif ENABLE_HUD_RECORDING:
        nvosd.link(queue2)
        queue2.link(encoder)
        encoder.link(parser)
        parser.link(container)
        container.link(file_sink)
    else:
        nvosd.link(null_sink)
    
    # Sınırlayıcı kutuları çıkarmak için prob ekle
    if preprocess_caps is not None:
        preprocess_pad = preprocess_caps.get_static_pad("src")
        preprocess_pad.add_probe(
            Gst.PadProbeType.BUFFER, preprocess_sink_pad_buffer_probe, 0
        )
    pgie_src_pad = pgie.get_static_pad("src")
    pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)
    tracker_src_pad = tracker.get_static_pad("src")
    tracker_src_pad.add_probe(
        Gst.PadProbeType.BUFFER, tracker_src_restore_buffer_probe, 0
    )
    osdsinkpad = nvosd.get_static_pad("sink")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    
    return pipeline

def main():
    global ros_node

    # Initialize global timing variables
    global start_time
    start_time = time.time()

    shutdown_event.clear()
    rclpy.init()
    ros_node = NormalizedTargetPublisher()

    def spin_ros_node():
        try:
            rclpy.spin(ros_node)
        except (ExternalShutdownException, KeyboardInterrupt):
            pass

    # Create and start pipeline
    pipeline = create_pipeline()

    # Create loop
    loop = GLib.MainLoop()
    
    # Add bus message handler for proper cleanup
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def request_shutdown(signum, _frame):
        if shutdown_event.is_set():
            return

        shutdown_event.set()
        signal_name = signal.Signals(signum).name
        print(f"\nReceived {signal_name}, stopping pipeline...")
        GLib.idle_add(loop.quit)
    
    def on_message(bus, message):
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, Debug: {debug}")
            loop.quit()
        elif message.type == Gst.MessageType.EOS:
            print("End of stream")
            loop.quit()
            
        elif redis.r.get("stop").decode('utf-8') == "True":
            print("Stopping pipeline as per Redis command")
            loop.quit()
    
    bus.connect("message", on_message)

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    ros_thread = threading.Thread(target=spin_ros_node, daemon=True)
    ros_thread.start()
    
    # Start pipeline
    print("Starting pipeline...")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("Failed to start pipeline")
        pipeline.set_state(Gst.State.NULL)
        if ros_node is not None:
            ros_node.destroy_node()
            ros_node = None
        if rclpy.ok():
            rclpy.shutdown()
        return
    
    if redis.r.get("stop").decode('utf-8') == "True":
        print("Stopping pipeline as per Redis command")
        loop.quit()
    
    try:
        loop.run()
    finally:
        shutdown_event.set()
        # Send EOS event to properly close MP4 file
        print("Sending EOS event to the pipeline...")
        pipeline.send_event(Gst.Event.new_eos())
        # Wait a bit for EOS to be processed
        time.sleep(2)
        # Print final statistics
        total_time = time.time() - start_time
        if total_time > 0:
            print(f"Total frames processed: {frame_count}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Average FPS: {frame_count / total_time:.2f}")
        redis.r.set("stop", "False")
        
        # Cleanup
        print("Setting pipeline to NULL state...")
        pipeline.set_state(Gst.State.NULL)
        bus.remove_signal_watch()
        if ros_node is not None:
            ros_node.destroy_node()
            ros_node = None
        if rclpy.ok():
            rclpy.shutdown()
        ros_thread.join(timeout=2)
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        print("Pipeline stopped.")


if __name__ == '__main__':
    main()
