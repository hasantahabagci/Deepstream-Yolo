#!/usr/bin/env python3

import configparser
import os
import sys
import time
import threading
import queue as queue_module
import signal
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds
import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy._rclpy_pybind11 import RCLError
from geometry_msgs.msg import Point, PointStamped
from mavros_msgs.msg import State
from sensor_msgs.msg import RegionOfInterest
from std_msgs.msg import Bool
from moving_average import MovingAverage
from project_paths import (
    PYTHON_TEST_OUTPUT_DIR,
    ROOT_DEEPSTREAM_APP_CONFIG,
    ROOT_PGIE_CONFIG_YOLO26,
    ROOT_TRACKER_FALLBACK_CONFIG,
)

try:
    from jetson_power_logger import JetsonPowerLogger, plot_power_csv
except Exception:
    JetsonPowerLogger = None  # type: ignore
    plot_power_csv = None  # type: ignore

ma = MovingAverage(60)

ros_node = None
shutdown_event = threading.Event()

power_logger = None
power_logging_started = False

multiplier = 1.0

PRIMARY_FOCAL_X = float(
    os.getenv("PYTHON_TEST_PRIMARY_FOCAL_X", str(1238.10428 * multiplier))
)
PRIMARY_FOCAL_Y = float(
    os.getenv("PYTHON_TEST_PRIMARY_FOCAL_Y", str(1238.78782 * multiplier))
)
SECONDARY_FOCAL_X = float(
    os.getenv("PYTHON_TEST_SECONDARY_FOCAL_X", str(1988.47063182 * multiplier))
)
SECONDARY_FOCAL_Y = float(
    os.getenv("PYTHON_TEST_SECONDARY_FOCAL_Y", str(1988.20715551 * multiplier))
)
SHARED_CX = float(os.getenv("PYTHON_TEST_CX", str(960 * multiplier)))
SHARED_CY = float(os.getenv("PYTHON_TEST_CY", str(540 * multiplier)))

RESOLUTION = (int(1920*multiplier), int(1080*multiplier))  # Set resolution as per your config

display = os.getenv("PYTHON_TEST_DISPLAY", "0") == "1"
ENABLE_HUD_RECORDING = os.getenv("PYTHON_TEST_RECORD_HUD", "1") == "1"
ENABLE_RAW_RECORDING = os.getenv("PYTHON_TEST_RECORD_RAW", "1") == "1"
ENABLE_ROI_RECORDING = os.getenv("PYTHON_TEST_RECORD_ROI", "0") == "1"
LOW_LATENCY_MODE = os.getenv("PYTHON_TEST_LOW_LATENCY", "1") == "1"
ENABLE_FULL_HUD = os.getenv("PYTHON_TEST_FULL_HUD", "0") == "1"
DEBUG_PRINT_DETECTIONS = os.getenv("PYTHON_TEST_DEBUG_DETECTIONS", "0") == "1"
ENABLE_DYNAMIC_ROI_INFERENCE = os.getenv("PYTHON_TEST_DYNAMIC_ROI", "1") == "1"
CAMERA_TNR_MODE = int(os.getenv("PYTHON_TEST_CAMERA_TNR_MODE", "0"))
CAMERA_TNR_STRENGTH = float(os.getenv("PYTHON_TEST_CAMERA_TNR_STRENGTH", "0"))
CAMERA_EE_MODE = int(os.getenv("PYTHON_TEST_CAMERA_EE_MODE", "0"))
CAMERA_EE_STRENGTH = float(os.getenv("PYTHON_TEST_CAMERA_EE_STRENGTH", "0"))
CAMERA_SATURATION = float(os.getenv("PYTHON_TEST_CAMERA_SATURATION", "1.0"))
CAMERA_EXPOSURE_COMPENSATION = float(os.getenv("PYTHON_TEST_CAMERA_EXPOSURE_COMPENSATION", "0.0"))
SOURCE_TYPE = os.getenv("PYTHON_TEST_SOURCE", "camera").strip().lower()
VIDEO_SOURCE_PATH = os.getenv("PYTHON_TEST_VIDEO_PATH", "").strip()
ALLOW_INFERENCE_WHILE_DEBUGGING = (
    os.getenv("PYTHON_TEST_DEBUG_RUNS_INFERENCE", "1") == "1"
)
SKIP_ARMING_CHECK = os.getenv("PYTHON_TEST_SKIP_ARMING_CHECK", "0") == "1"
DEBUGGER_ATTACHED = False
DEBUG_INFERENCE_MODE = DEBUGGER_ATTACHED and ALLOW_INFERENCE_WHILE_DEBUGGING
REQUIRE_ARM_FOR_INFERENCE = (
    SOURCE_TYPE == "camera" and not (SKIP_ARMING_CHECK or DEBUG_INFERENCE_MODE)
)
PRIMARY_FLIP_METHOD = int(os.getenv("PYTHON_TEST_PRIMARY_FLIP_METHOD", "0"))
SECONDARY_FLIP_METHOD = int(os.getenv("PYTHON_TEST_SECONDARY_FLIP_METHOD", "2"))

FPS = 30
INFERENCE_ROI_TOPIC = os.getenv(
    "PYTHON_TEST_ROI_TOPIC", "/interceptor/camera/inference_roi"
).strip() or "/interceptor/camera/inference_roi"
INFERENCE_ROI_TIMEOUT_SEC = max(
    0.1, float(os.getenv("PYTHON_TEST_ROI_TIMEOUT", "0.5"))
)
FRAME_INFERENCE_CONTEXT_LIMIT = max(32, FPS * 4)
APP_CONFIG_PATH = str(ROOT_DEEPSTREAM_APP_CONFIG)
PGIE_CONFIG_PATH = (
    os.getenv("PYTHON_TEST_PGIE_CONFIG", str(ROOT_PGIE_CONFIG_YOLO26)).strip()
    or str(ROOT_PGIE_CONFIG_YOLO26)
)
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
LOW_LATENCY_QUEUE_SIZE = max(1, int(os.getenv("PYTHON_TEST_QUEUE_SIZE", "1")))
ROI_RECORD_QUEUE_SIZE = max(1, int(os.getenv("PYTHON_TEST_ROI_RECORD_QUEUE", "1")))
CAMERA_SWITCH_TOPIC = os.getenv(
    "PYTHON_TEST_CAMERA_SWITCH_TOPIC", "/interceptor/camera/select_secondary"
).strip() or "/interceptor/camera/select_secondary"
PRIMARY_SENSOR_ID = int(os.getenv("PYTHON_TEST_PRIMARY_SENSOR_ID", "0"))
SECONDARY_SENSOR_ID = int(os.getenv("PYTHON_TEST_SECONDARY_SENSOR_ID", "1"))
PRIMARY_CAMERA_LABEL = (
    os.getenv("PYTHON_TEST_PRIMARY_CAMERA_LABEL", f"CAM{PRIMARY_SENSOR_ID}").strip()
    or f"CAM{PRIMARY_SENSOR_ID}"
)
SECONDARY_CAMERA_LABEL = (
    os.getenv("PYTHON_TEST_SECONDARY_CAMERA_LABEL", f"CAM{SECONDARY_SENSOR_ID}").strip()
    or f"CAM{SECONDARY_SENSOR_ID}"
)

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
    "roi_enabled": ENABLE_DYNAMIC_ROI_INFERENCE,
    "timestamp": 0.0,
}

frame_inference_lock = threading.Lock()
frame_inference_context = {}

# Global variables for FPS calculation
frame_count = 0
start_time = time.time()
FPS_REPORT_INTERVAL_FRAMES = max(1, FPS)  # report about once per second
fps_window_start_time = time.time()
fps_last_reported = None
selected_track_id = None
selected_track_misses = 0
hud_cache = HUD_DEFAULTS.copy()
roi_inference_warning_printed = False
roi_recorder = None
roi_recording_output_path = None
roi_recording_failed = False
roi_recorder_lock = threading.Lock()
camera_switch_lock = threading.Lock()
camera_switch_state = {
    "requested_secondary": False,
    "topic_seen": False,
    "active_secondary": False,
    "idle_sync_pending": False,
}
camera_switch_runtime = {
    "source": None,
    "display_transform": None,
    "raw_transform": None,
    "pipeline_playing": False,
    "switch_in_progress": False,
}


def _reset_selected_track():
    global selected_track_id, selected_track_misses
    selected_track_id = None
    selected_track_misses = 0


def _camera_label(use_secondary):
    return SECONDARY_CAMERA_LABEL if use_secondary else PRIMARY_CAMERA_LABEL


def _camera_sensor_id(use_secondary):
    return SECONDARY_SENSOR_ID if use_secondary else PRIMARY_SENSOR_ID


def _camera_intrinsics(use_secondary):
    if use_secondary:
        return SECONDARY_FOCAL_X, SECONDARY_FOCAL_Y, SHARED_CX, SHARED_CY
    return PRIMARY_FOCAL_X, PRIMARY_FOCAL_Y, SHARED_CX, SHARED_CY


def _camera_flip_method(use_secondary):
    return SECONDARY_FLIP_METHOD if use_secondary else PRIMARY_FLIP_METHOD


def get_active_camera_label():
    with camera_switch_lock:
        return _camera_label(bool(camera_switch_state["active_secondary"]))


def get_active_camera_intrinsics():
    if SOURCE_TYPE != "camera":
        return _camera_intrinsics(False)

    with camera_switch_lock:
        use_secondary = bool(camera_switch_state["active_secondary"])
    return _camera_intrinsics(use_secondary)


def get_display_flip_method():
    if SOURCE_TYPE != "camera":
        return 0

    with camera_switch_lock:
        use_secondary = bool(camera_switch_state["active_secondary"])
    return _camera_flip_method(use_secondary)


def get_roi_inference_flip_method():
    return get_display_flip_method()


def get_roi_coordinate_flip_method():
    return get_display_flip_method()


def get_display_source_label():
    if SOURCE_TYPE == "camera":
        return get_active_camera_label()
    return "4mm"


def _compute_desired_secondary_locked():
    if not camera_switch_state["topic_seen"]:
        return bool(camera_switch_state["active_secondary"])

    return bool(camera_switch_state["requested_secondary"])


def _schedule_camera_switch_sync(reason="update"):
    if SOURCE_TYPE != "camera":
        return

    with camera_switch_lock:
        source = camera_switch_runtime["source"]
        if source is None or camera_switch_state["idle_sync_pending"]:
            return
        camera_switch_state["idle_sync_pending"] = True

    GLib.idle_add(_camera_switch_idle_sync, reason)


def _set_gst_element_state(element, state, timeout_sec=5.0):
    result = element.set_state(state)
    if result == Gst.StateChangeReturn.FAILURE:
        return False

    if result == Gst.StateChangeReturn.ASYNC:
        timeout_ns = int(timeout_sec * Gst.SECOND)
        state_change, _current, _pending = element.get_state(timeout_ns)
        return state_change != Gst.StateChangeReturn.FAILURE

    return True


def _apply_output_flip_method(use_secondary):
    flip_method = _camera_flip_method(use_secondary)

    with camera_switch_lock:
        display_transform = camera_switch_runtime["display_transform"]
        raw_transform = camera_switch_runtime["raw_transform"]

    for transform in (display_transform, raw_transform):
        if transform is None:
            continue
        if transform.find_property("flip-method") is not None:
            transform.set_property("flip-method", int(flip_method))


def _apply_camera_switch_if_needed(reason="update"):
    if SOURCE_TYPE != "camera":
        return False

    with camera_switch_lock:
        source = camera_switch_runtime["source"]
        if source is None:
            return False

        desired_secondary = _compute_desired_secondary_locked()
        active_secondary = bool(camera_switch_state["active_secondary"])
        if desired_secondary == active_secondary:
            return False

        if camera_switch_runtime["switch_in_progress"]:
            return False

        target_sensor_id = _camera_sensor_id(desired_secondary)
        active_sensor_id = _camera_sensor_id(active_secondary)
        pipeline_playing = bool(camera_switch_runtime["pipeline_playing"])
        camera_switch_runtime["switch_in_progress"] = True

    try:
        if not pipeline_playing:
            source.set_property("sensor-id", int(target_sensor_id))
        else:
            if not _set_gst_element_state(source, Gst.State.NULL, timeout_sec=5.0):
                raise RuntimeError("failed to stop camera source before sensor handover")

            source.set_property("sensor-id", int(target_sensor_id))

            if not _set_gst_element_state(source, Gst.State.PLAYING, timeout_sec=5.0):
                raise RuntimeError("failed to restart camera source after sensor handover")
    except Exception as exc:
        print(
            "Failed to hand over camera source "
            f"(from sensor-id={active_sensor_id} to sensor-id={target_sensor_id}, "
            f"reason={reason}): {exc}"
        )
        try:
            source.set_property("sensor-id", int(active_sensor_id))
            if pipeline_playing:
                _set_gst_element_state(source, Gst.State.PLAYING, timeout_sec=5.0)
        except Exception as restore_exc:
            print(f"Failed to restore previous camera source after handover error: {restore_exc}")
        finally:
            with camera_switch_lock:
                camera_switch_runtime["switch_in_progress"] = False
        return False

    with camera_switch_lock:
        camera_switch_state["active_secondary"] = desired_secondary
        camera_switch_runtime["switch_in_progress"] = False

    _apply_output_flip_method(desired_secondary)
    _reset_selected_track()
    clear_latest_inference_roi()
    print(
        "Camera switch: active source -> "
        f"{_camera_label(desired_secondary)} "
        f"(sensor-id={_camera_sensor_id(desired_secondary)}, reason={reason})"
    )

    with camera_switch_lock:
        needs_resync = (
            _compute_desired_secondary_locked()
            != bool(camera_switch_state["active_secondary"])
        )
    if needs_resync:
        _schedule_camera_switch_sync("post-switch")

    return True


def _camera_switch_idle_sync(reason):
    with camera_switch_lock:
        camera_switch_state["idle_sync_pending"] = False
    _apply_camera_switch_if_needed(reason)
    return False


def _reset_camera_switch_runtime():
    with camera_switch_lock:
        camera_switch_runtime["source"] = None
        camera_switch_runtime["display_transform"] = None
        camera_switch_runtime["raw_transform"] = None
        camera_switch_runtime["pipeline_playing"] = False
        camera_switch_runtime["switch_in_progress"] = False
        camera_switch_state["requested_secondary"] = False
        camera_switch_state["topic_seen"] = False
        camera_switch_state["active_secondary"] = False
        camera_switch_state["idle_sync_pending"] = False


class NormalizedTargetPublisher(Node):
    def __init__(self):
        super().__init__("python_test_normalized_target_publisher")
        self.target_pub = self.create_publisher(PointStamped, P_BAR_TOPIC, 10)
        self.dkf_sub = self.create_subscription(Point, DKF_BAR_TOPIC, self.dkf_callback, 10)
        self.roi_sub = self.create_subscription(
            RegionOfInterest, INFERENCE_ROI_TOPIC, self.roi_callback, 10
        )
        self.camera_switch_sub = self.create_subscription(
            Bool, CAMERA_SWITCH_TOPIC, self.camera_switch_callback, 10
        )
        self.mavros_state_sub = self.create_subscription(
            State, "/mavros/state", self.mavros_state_callback, 10
        )
        self.was_armed = False
        self.shutdown_requested = False
        self.inference_ready_event = threading.Event()
        if not REQUIRE_ARM_FOR_INFERENCE:
            self.inference_ready_event.set()

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
        focal_x, focal_y, c_x, c_y = get_active_camera_intrinsics()

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

    def roi_callback(self, msg):
        # Use RegionOfInterest.do_rectify as a runtime switch:
        #   False => don't use ROI for inference (full frame)
        #   True  => use ROI for inference
        update_roi_enabled(bool(msg.do_rectify))

        if not msg.do_rectify or msg.width <= 0 or msg.height <= 0:
            clear_latest_inference_roi()
            return

        # Convert incoming rectangle ROI to a square ROI that keeps the
        # incoming *rectangle* center and uses the larger edge length.
        in_x = int(msg.x_offset)
        in_y = int(msg.y_offset)
        in_w = int(msg.width)
        in_h = int(msg.height)

        center_x = float(in_x) + (float(in_w) / 2.0)
        center_y = float(in_y) + (float(in_h) / 2.0)
        square_edge = int(max(in_w, in_h))

        square_x = int(round(center_x - (float(square_edge) / 2.0)))
        square_y = int(round(center_y - (float(square_edge) / 2.0)))

        update_latest_inference_roi(square_x, square_y, square_edge, square_edge)

    def camera_switch_callback(self, msg):
        if shutdown_event.is_set():
            return

        with camera_switch_lock:
            camera_switch_state["requested_secondary"] = bool(msg.data)
            camera_switch_state["topic_seen"] = True

        _schedule_camera_switch_sync("topic")

    def mavros_state_callback(self, msg):
        if shutdown_event.is_set() or self.shutdown_requested:
            self.was_armed = msg.armed
            return

        if not REQUIRE_ARM_FOR_INFERENCE:
            self.was_armed = msg.armed
            self.inference_ready_event.set()
            return

        if msg.armed:
            if not self.inference_ready_event.is_set():
                self.get_logger().info(
                    "Vehicle armed. Enabling inference and starting pipeline."
                )
                _start_power_logging(self)
            self.was_armed = True
            self.inference_ready_event.set()
            return

        self.inference_ready_event.clear()

        if self.was_armed and not msg.armed:
            self.shutdown_requested = True
            self.get_logger().info(
                "Vehicle disarmed after arming. Sending SIGINT to stop python_test.py."
            )
            _stop_power_logging(self)
            os.kill(os.getpid(), signal.SIGINT)


def _start_power_logging(node: Node | None = None) -> None:
    """Start Jetson power sampling in the background.

    Best-effort: if telemetry files don't exist (non-Jetson), it logs a message
    and continues without failing the main pipeline.
    """

    global power_logger, power_logging_started
    if power_logging_started:
        return

    sample_hz = float(os.getenv("PYTHON_TEST_POWER_LOG_HZ", "5.0"))
    enabled = os.getenv("PYTHON_TEST_POWER_LOG", "1") == "1"
    if not enabled:
        return

    if JetsonPowerLogger is None:
        if node is not None:
            node.get_logger().warn(
                "Power logging requested but jetson_power_logger import failed."
            )
        return

    out_dir = str(PYTHON_TEST_OUTPUT_DIR)
    power_logger = JetsonPowerLogger(output_dir=out_dir, sample_hz=sample_hz)
    ok = power_logger.start()
    if not ok:
        power_logger = None
        if node is not None:
            node.get_logger().warn(
                "jtop couldn't start; skipping power logging."
            )
        return

    power_logging_started = True
    if node is not None:
        node.get_logger().info(
            f"Started Jetson power logging via jtop ({sample_hz:.1f} Hz)."
        )


def _stop_power_logging(node: Node | None = None) -> None:
    """Stop background power sampling, write CSV, and create plots."""

    global power_logger, power_logging_started
    if not power_logger or not power_logging_started:
        return

    try:
        power_logger.stop()
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = str(PYTHON_TEST_OUTPUT_DIR)
        csv_path = os.path.join(out_dir, f"jetson_power_{ts}.csv")
        power_logger.write_csv(csv_path)

        if plot_power_csv is not None:
            plot_dir = os.path.join(out_dir, f"jetson_power_{ts}_plots")
            written = plot_power_csv(csv_path, plot_dir)
        else:
            written = []

        if node is not None:
            node.get_logger().info(f"Power log saved: {csv_path}")
            if written:
                node.get_logger().info(
                    f"Power plots saved: {os.path.dirname(written[0])}"
                )
    except Exception as e:
        if node is not None:
            node.get_logger().warn(f"Failed to finalize power log: {e}")
    finally:
        power_logger = None
        power_logging_started = False


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


def update_roi_enabled(enabled: bool):
    with roi_lock:
        inference_roi_state["roi_enabled"] = bool(enabled)


def _transform_rect_for_flip(
    left, top, width, height, frame_width, frame_height, flip_method
):
    if flip_method == 2:
        return (
            frame_width - (left + width),
            frame_height - (top + height),
            width,
            height,
        )
    if flip_method == 4:
        return frame_width - (left + width), top, width, height
    if flip_method == 6:
        return left, frame_height - (top + height), width, height
    return left, top, width, height


def _transform_bounds_for_flip(bounds, frame_width, frame_height, flip_method):
    if bounds is None:
        return None

    left, top, right, bottom = bounds
    width = right - left
    height = bottom - top
    transformed_left, transformed_top, transformed_width, transformed_height = (
        _transform_rect_for_flip(
            left,
            top,
            width,
            height,
            frame_width,
            frame_height,
            flip_method,
        )
    )
    return _clamp_roi_bounds(
        transformed_left,
        transformed_top,
        transformed_width,
        transformed_height,
        frame_width,
        frame_height,
    )


def update_latest_inference_roi(x_offset, y_offset, width, height):
    # Inputs follow sensor_msgs/RegionOfInterest: (x_offset, y_offset, width, height)
    # Convert to bounds while preserving the *center* of the ROI.
    width = int(width)
    height = int(height)
    center_x = float(x_offset) + (float(width) / 2.0)
    center_y = float(y_offset) + (float(height) / 2.0)
    roi_width = width
    roi_height = height
    roi_left = int(round(center_x - (roi_width / 2.0)))
    roi_top = int(round(center_y - (roi_height / 2.0)))

    transformed_left, transformed_top, transformed_width, transformed_height = (
        _transform_rect_for_flip(
            roi_left,
            roi_top,
            roi_width,
            roi_height,
            RESOLUTION[0],
            RESOLUTION[1],
            get_roi_coordinate_flip_method(),
        )
    )

    with roi_lock:
        inference_roi_state["x_offset"] = int(transformed_left)
        inference_roi_state["y_offset"] = int(transformed_top)
        inference_roi_state["width"] = int(transformed_width)
        inference_roi_state["height"] = int(transformed_height)
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
    with roi_lock:
        roi_enabled = bool(inference_roi_state.get("roi_enabled", False))
        timestamp = inference_roi_state["timestamp"]
        x_offset = inference_roi_state["x_offset"]
        y_offset = inference_roi_state["y_offset"]
        width = inference_roi_state["width"]
        height = inference_roi_state["height"]

    if not roi_enabled:
        return None, None

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


def _flip_frame_for_inference(frame, flip_method):
    if flip_method == 2:
        return cv2.flip(frame, -1)
    if flip_method == 4:
        return cv2.flip(frame, 1)
    if flip_method == 6:
        return cv2.flip(frame, 0)
    return frame


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


def _configure_queue_for_low_latency(queue_element, leaky=True):
    if queue_element is None:
        return

    queue_element.set_property(
        "max-size-buffers", LOW_LATENCY_QUEUE_SIZE if LOW_LATENCY_MODE else 4
    )
    queue_element.set_property("max-size-time", 0)
    queue_element.set_property("max-size-bytes", 0)
    if leaky and LOW_LATENCY_MODE:
        queue_element.set_property("leaky", 2)


class InferenceRoiRecorder:
    def __init__(self, output_path, width, height, fps):
        self.output_path = output_path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(max(1, int(fps)))
        self.frame_index = 0
        self.dropped_frames = 0
        self.closed = False
        self._stop_sentinel = object()
        self._frame_queue = queue_module.Queue(maxsize=ROI_RECORD_QUEUE_SIZE)
        self.writer = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.width, self.height),
        )
        if not self.writer.isOpened():
            raise RuntimeError("Unable to open inference ROI video writer")
        self.worker_thread = threading.Thread(
            target=self._writer_loop,
            name="inference-roi-recorder",
            daemon=True,
        )
        self.worker_thread.start()

    def _writer_loop(self):
        while True:
            frame = self._frame_queue.get()
            if frame is self._stop_sentinel:
                self._frame_queue.task_done()
                break

            try:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                self.writer.write(bgr_frame)
                self.frame_index += 1
            finally:
                self._frame_queue.task_done()

    def write_rgba_frame(self, rgba_frame):
        if self.closed:
            return

        if rgba_frame is None or rgba_frame.ndim != 3 or rgba_frame.shape[2] < 4:
            return

        frame_copy = np.array(rgba_frame[:, :, :4], copy=True)
        try:
            self._frame_queue.put_nowait(frame_copy)
        except queue_module.Full:
            try:
                stale_frame = self._frame_queue.get_nowait()
                self._frame_queue.task_done()
                if stale_frame is not self._stop_sentinel:
                    self.dropped_frames += 1
            except queue_module.Empty:
                pass
            self._frame_queue.put_nowait(frame_copy)

    def close(self):
        if self.closed:
            return

        self.closed = True
        try:
            self._frame_queue.put(self._stop_sentinel)
            self.worker_thread.join(timeout=5.0)
            self.writer.release()
        except Exception:
            pass
        if self.frame_index > 0:
            dropped_text = (
                f" (dropped {self.dropped_frames} stale frames)"
                if self.dropped_frames > 0
                else ""
            )
            print(
                f"Inference ROI recording saved {self.frame_index} frames to "
                f"{self.output_path}{dropped_text}"
            )


def _ensure_inference_roi_recorder():
    global roi_recorder
    global roi_recording_failed

    if not ENABLE_ROI_RECORDING or roi_recording_output_path is None or roi_recording_failed:
        return None

    with roi_recorder_lock:
        if roi_recorder is not None:
            return roi_recorder

        try:
            roi_recorder = InferenceRoiRecorder(
                str(roi_recording_output_path),
                RESOLUTION[0],
                RESOLUTION[1],
                FPS,
            )
        except Exception as exc:
            print(f"Failed to start inference ROI recorder: {exc}")
            roi_recording_failed = True
            roi_recorder = None

        return roi_recorder


def _record_inference_roi_frame(rgba_frame):
    recorder = _ensure_inference_roi_recorder()
    if recorder is None:
        return

    try:
        recorder.write_rgba_frame(rgba_frame)
    except Exception as exc:
        print(f"Failed to write inference ROI frame: {exc}")
        _shutdown_inference_roi_recorder(disable=True)


def _shutdown_inference_roi_recorder(disable=False):
    global roi_recorder
    global roi_recording_failed

    with roi_recorder_lock:
        recorder = roi_recorder
        roi_recorder = None
        if disable:
            roi_recording_failed = True

    if recorder is not None:
        recorder.close()


def get_log_number(path="/home/ituarc/ros2_ws/src/thermal_guidance/logs/last_log_id.txt"):
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except Exception as e:
        print(f"Error reading log number: {e}")
        return 0


def refresh_hud_cache(frame_number):
    # Redis integration removed; leave cached defaults as-is.
    return


def roi_inference_sink_pad_buffer_probe(pad, info, u_data):
    global roi_inference_warning_printed

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
            "roi_flip_method": 0,
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
                roi_flip_method = get_roi_inference_flip_method()
                frame_context["used_roi_inference"] = True
                needs_crop_resize = (
                    left != 0
                    or top != 0
                    or roi_width != frame_width
                    or roi_height != frame_height
                )
                needs_roi_flip = roi_flip_method != 0

                if needs_crop_resize or needs_roi_flip:
                    frame_context["restore_frame"] = np.array(rgba_frame, copy=True)
                    roi_frame = np.ascontiguousarray(rgba_frame[top:bottom, left:right, :])

                    if needs_roi_flip:
                        roi_frame = _flip_frame_for_inference(roi_frame, roi_flip_method)
                        frame_context["roi_flip_method"] = roi_flip_method

                    if needs_crop_resize:
                        roi_frame = cv2.resize(
                            roi_frame,
                            (frame_width, frame_height),
                            interpolation=cv2.INTER_LINEAR,
                        )

                    np.copyto(rgba_frame, roi_frame)

            if frame_context["used_roi_inference"] and ENABLE_ROI_RECORDING:
                # Persist the exact ROI view that is fed into inference.
                _record_inference_roi_frame(rgba_frame)
        except Exception as exc:
            if not roi_inference_warning_printed:
                print(f"ROI inference path disabled after error: {exc}")
                roi_inference_warning_printed = True
            frame_context["warning_text"] = (
                "ROI inference error; full-frame inference fallback active"
            )
            frame_context["roi_bounds"] = None
            frame_context["roi_flip_method"] = 0
            frame_context["used_roi_inference"] = False
            frame_context["restore_frame"] = None
            clear_latest_inference_roi()
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


def _remap_bbox_from_roi(
    obj_meta, roi_bounds, frame_width, frame_height, roi_flip_method=0
):
    left, top, right, bottom = roi_bounds
    roi_width = max(1.0, float(right - left))
    roi_height = max(1.0, float(bottom - top))
    scale_x = roi_width / max(1.0, float(frame_width))
    scale_y = roi_height / max(1.0, float(frame_height))

    new_width = obj_meta.rect_params.width * scale_x
    new_height = obj_meta.rect_params.height * scale_y

    remapped_left = obj_meta.rect_params.left * scale_x
    remapped_top = obj_meta.rect_params.top * scale_y

    if roi_flip_method == 2:
        remapped_left = roi_width - (remapped_left + new_width)
        remapped_top = roi_height - (remapped_top + new_height)
    elif roi_flip_method == 4:
        remapped_left = roi_width - (remapped_left + new_width)
    elif roi_flip_method == 6:
        remapped_top = roi_height - (remapped_top + new_height)

    new_left = left + remapped_left
    new_top = top + remapped_top

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
            roi_flip_method = int(frame_context.get("roi_flip_method", 0))
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
                        obj_meta,
                        roi_bounds,
                        frame_width,
                        frame_height,
                        roi_flip_method=roi_flip_method,
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


def _request_pad(element, template_name):
    if hasattr(element, "request_pad_simple"):
        return element.request_pad_simple(template_name)
    return element.get_request_pad(template_name)


def _create_camera_source_chain(sensor_id):
    source = Gst.ElementFactory.make("nvarguscamerasrc", "camera-source")
    caps_filter = Gst.ElementFactory.make("capsfilter", "camera-caps-filter")
    queue = Gst.ElementFactory.make("queue", "camera-source-queue")

    if not all([source, caps_filter, queue]):
        return None, None, None

    source.set_property("sensor-id", int(sensor_id))
    source.set_property("tnr-mode", CAMERA_TNR_MODE)
    source.set_property("tnr-strength", CAMERA_TNR_STRENGTH)
    source.set_property("ee-mode", CAMERA_EE_MODE)
    source.set_property("ee-strength", CAMERA_EE_STRENGTH)
    source.set_property("saturation", CAMERA_SATURATION)
    source.set_property("exposurecompensation", CAMERA_EXPOSURE_COMPENSATION)
    caps_filter.set_property(
        "caps",
        Gst.Caps.from_string(
            f"video/x-raw(memory:NVMM), width={RESOLUTION[0]}, height={RESOLUTION[1]}, "
            f"format=NV12, framerate={FPS}/1"
        ),
    )
    _configure_queue_for_low_latency(queue, leaky=False)
    return source, caps_filter, queue


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
    global frame_count, fps_window_start_time, fps_last_reported
    
    frame_count += 1
    
    if frame_count % FPS_REPORT_INTERVAL_FRAMES == 0:
        current_time = time.time()
        elapsed_time = current_time - fps_window_start_time
        if elapsed_time <= 0:
            return None

        fps = FPS_REPORT_INTERVAL_FRAMES / elapsed_time
        fps_window_start_time = current_time
        fps_last_reported = fps
        return fps

    return None


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
        
    calculate_fps()
    current_fps = fps_last_reported
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
            focal_x, focal_y, c_x, c_y = get_active_camera_intrinsics()
            center_x = target_bbox['left'] + target_bbox['width'] / 2
            center_y = target_bbox['top'] + target_bbox['height'] / 2
            # Normalized image coordinates with setting center as zero
            x_bar = (center_x - c_x) / focal_x
            y_bar = (center_y - c_y) / focal_y

        # Publish only when we have a valid detection.
        if (
            target_bbox
            and ros_node is not None
            and x_bar is not None
            and y_bar is not None
            and np.isfinite(x_bar)
            and np.isfinite(y_bar)
        ):
            ros_node.publish_target(x_bar, y_bar)

        # Görüntü metadatasını al
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        
        # Metin bilgilerini yapılandır (Mevcut kodunuz)
        frame_context = _get_frame_inference_context(frame_number) or {}
        warning_text = frame_context.get("warning_text")
        roi_bounds = frame_context.get("roi_bounds")
        display_frame_width = int(frame_context.get("frame_width", RESOLUTION[0]))
        display_frame_height = int(frame_context.get("frame_height", RESOLUTION[1]))
        display_roi_bounds = _transform_bounds_for_flip(
            roi_bounds,
            display_frame_width,
            display_frame_height,
            get_display_flip_method(),
        )

        display_meta.num_labels = 2 if warning_text else 1
        py_nvosd_text_params = display_meta.text_params[0]
        refresh_hud_cache(frame_number)
        fps_vis = ma.update(current_fps) if current_fps is not None else None
        fps_text = f" FPS: {fps_vis:.1f}" if fps_vis is not None else ""
        display_text = (
            f"Frame Number={frame_number} Number of Objects={num_rects}"
            f"{fps_text} CAM: {get_display_source_label()}"
        )
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
        next_line_index += 1
        

        if display_roi_bounds is not None:
            left, top, right, bottom = display_roi_bounds
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
                roi_line.line_width = 4
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
    global roi_recording_failed
    global roi_recording_output_path

    # Initialize GStreamer
    Gst.init(None)
    
    # Create pipeline
    pipeline = Gst.Pipeline()
    _reset_camera_switch_runtime()
    
    # --- Original Elements ---
    source = None
    caps_filter = None
    queue_source = None
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    queue_mux_output = Gst.ElementFactory.make("queue", "queue-mux-output")
    preprocess_enabled = ENABLE_DYNAMIC_ROI_INFERENCE
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
    queue_tracker_output = Gst.ElementFactory.make("queue", "queue-tracker-output")
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
    queue_osd_input = Gst.ElementFactory.make("queue", "queue-osd-input") if ENABLE_RAW_RECORDING else None
    nvvidconv_raw = Gst.ElementFactory.make("nvvideoconvert", "convertor-raw") if ENABLE_RAW_RECORDING else None
    encoder_raw = Gst.ElementFactory.make("nvv4l2h264enc", "encoder-raw") if ENABLE_RAW_RECORDING else None
    parser_raw = Gst.ElementFactory.make("h264parse", "parser-raw") if ENABLE_RAW_RECORDING else None
    container_raw = Gst.ElementFactory.make("mp4mux", "container-raw") if ENABLE_RAW_RECORDING else None
    file_sink_raw = Gst.ElementFactory.make("filesink", "file-sink-raw") if ENABLE_RAW_RECORDING else None
    null_sink = Gst.ElementFactory.make("fakesink", "null-sink") if not display and not ENABLE_HUD_RECORDING else None

    if SOURCE_TYPE == "camera":
        if PRIMARY_SENSOR_ID == SECONDARY_SENSOR_ID:
            sys.stderr.write(
                "PYTHON_TEST_PRIMARY_SENSOR_ID and PYTHON_TEST_SECONDARY_SENSOR_ID "
                "must refer to different CSI sensors\n"
            )
            sys.exit(1)
        with camera_switch_lock:
            initial_secondary = _compute_desired_secondary_locked()
        initial_sensor_id = _camera_sensor_id(initial_secondary)
        source, caps_filter, queue_source = _create_camera_source_chain(initial_sensor_id)
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

    if SOURCE_TYPE == "camera" and not all(
        [source, caps_filter, queue_source, streammux, queue_mux_output, pgie, tracker, queue_tracker_output, nvvidconv, nvosd]
    ):
        sys.stderr.write("Unable to create camera source elements\n")
        sys.exit(1)
    elif SOURCE_TYPE == "video" and not all(
        [source, streammux, queue_mux_output, pgie, tracker, queue_tracker_output, nvvidconv, nvosd]
    ):
        sys.stderr.write("Unable to create essential elements\n")
        sys.exit(1)

    if preprocess_enabled and not all([preprocess_convert, preprocess_caps]):
        sys.stderr.write("Unable to create ROI inference elements\n")
        sys.exit(1)
    
    if ENABLE_HUD_RECORDING and not all([queue2, encoder, parser, container, file_sink]):
        sys.stderr.write("Unable to create HUD recording elements\n")
        sys.exit(1)
    
    # --- YENİ: Ham Kayıt Elemanları için Kontrol ---
    if ENABLE_RAW_RECORDING and not all([
        tee_raw_split,
        queue_raw,
        queue_osd_input,
        nvvidconv_raw,
        encoder_raw,
        parser_raw,
        container_raw,
        file_sink_raw,
    ]):
        sys.stderr.write("Unable to create Raw recording elements\n")
        sys.exit(1)

    # --- Özellikleri Ayarlama ---
    initial_flip_method = 0
    if SOURCE_TYPE == "camera":
        initial_flip_method = _camera_flip_method(bool(initial_secondary))
    nvvidconv.set_property("flip-method", initial_flip_method)
    if nvvidconv_raw is not None:
        nvvidconv_raw.set_property("flip-method", initial_flip_method)
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
    
    pgie.set_property("config-file-path", PGIE_CONFIG_PATH)
    configure_tracker(tracker)
    if LOW_LATENCY_MODE:
        print(
            "Low-latency mode enabled "
            f"(queue-size={LOW_LATENCY_QUEUE_SIZE}, roi-record-queue={ROI_RECORD_QUEUE_SIZE})"
        )
    if preprocess_enabled:
        print(
            "ROI inference path enabled "
            f"(dynamic-roi={ENABLE_DYNAMIC_ROI_INFERENCE}, topic={INFERENCE_ROI_TOPIC}, "
            f"timeout={INFERENCE_ROI_TIMEOUT_SEC}s)"
        )
    if SOURCE_TYPE == "camera":
        print(
            "Camera ISP tuning "
            f"(tnr-mode={CAMERA_TNR_MODE}, tnr-strength={CAMERA_TNR_STRENGTH}, "
            f"ee-mode={CAMERA_EE_MODE}, ee-strength={CAMERA_EE_STRENGTH}, "
            f"saturation={CAMERA_SATURATION}, exposure-comp={CAMERA_EXPOSURE_COMPENSATION})"
        )
        print(
            "Camera calibration "
            f"(primary: fx={PRIMARY_FOCAL_X:.2f}, fy={PRIMARY_FOCAL_Y:.2f}, flip={PRIMARY_FLIP_METHOD}; "
            f"secondary: fx={SECONDARY_FOCAL_X:.2f}, fy={SECONDARY_FOCAL_Y:.2f}, flip={SECONDARY_FLIP_METHOD}; "
            f"cx={SHARED_CX:.2f}, cy={SHARED_CY:.2f})"
        )
        print(
            "Camera switch enabled "
            f"(topic={CAMERA_SWITCH_TOPIC}, latched-level, "
            f"primary={PRIMARY_CAMERA_LABEL}/sensor-id={PRIMARY_SENSOR_ID}, "
            f"secondary={SECONDARY_CAMERA_LABEL}/sensor-id={SECONDARY_SENSOR_ID}, "
            "mode=exclusive-handover)"
        )
    else:
        print(f"Video file source: {video_uri}")
    
    if display:
        display_sink.set_property("sync", False)
        if LOW_LATENCY_MODE:
            if display_sink.find_property("processing-deadline") is not None:
                display_sink.set_property("processing-deadline", 0)
            if display_sink.find_property("max-lateness") is not None:
                display_sink.set_property("max-lateness", 0)
    
    if null_sink is not None and null_sink.find_property("sync") is not None:
        null_sink.set_property("sync", False)
    
    _configure_queue_for_low_latency(queue_mux_output, leaky=LOW_LATENCY_MODE)
    _configure_queue_for_low_latency(queue_tracker_output, leaky=LOW_LATENCY_MODE)
    if queue1 is not None:
        _configure_queue_for_low_latency(queue1, leaky=True)
    if queue2 is not None:
        _configure_queue_for_low_latency(queue2, leaky=True)
    
    # --- YENİ: Ham Kayıt Kuyruğu Özellikleri ---
    if queue_raw is not None:
        _configure_queue_for_low_latency(queue_raw, leaky=True)
    if queue_osd_input is not None:
        _configure_queue_for_low_latency(queue_osd_input, leaky=True)
    
    # HUD video kayıt özellikleri
    if encoder is not None:
        encoder.set_property("bitrate", 4000000)
    
    # --- YENİ: Ham Video Kodlayıcı Özellikleri ---
    if encoder_raw is not None:
        encoder_raw.set_property("bitrate", 4000000)
    
    roi_recording_output_path = None
    roi_recording_failed = False
    if ENABLE_HUD_RECORDING or ENABLE_RAW_RECORDING or ENABLE_ROI_RECORDING:
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
        if ENABLE_ROI_RECORDING:
            roi_recording_output_path = (
                PYTHON_TEST_OUTPUT_DIR / f"output_4mm_{log_number}_roi.mp4"
            )
            print(f"Inference ROI recording enabled: {roi_recording_output_path}")

    # --- Boru Hattına Eleman Ekleme ---
    for element in [
        source,
        caps_filter,
        queue_source,
        streammux,
        queue_mux_output,
        preprocess_convert,
        preprocess_caps,
        pgie,
        tracker,
        queue_tracker_output,
        nvvidconv,
        tee_raw_split,
        queue_osd_input,
        nvvidconv_raw,
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
        if not source.link(caps_filter):
            sys.stderr.write("Failed to link camera source -> capsfilter\n")
            sys.exit(1)
        if not caps_filter.link(queue_source):
            sys.stderr.write("Failed to link camera capsfilter -> queue\n")
            sys.exit(1)
        sinkpad = _request_pad(streammux, "sink_0")
        if sinkpad is None:
            sys.stderr.write("Failed to request streammux sink pad\n")
            sys.exit(1)
        if (
            queue_source.get_static_pad("src").link(sinkpad)
            != Gst.PadLinkReturn.OK
        ):
            sys.stderr.write("Failed to link camera queue -> streammux\n")
            sys.exit(1)

        with camera_switch_lock:
            camera_switch_runtime["source"] = source
            camera_switch_runtime["display_transform"] = nvvidconv
            camera_switch_runtime["raw_transform"] = nvvidconv_raw
            camera_switch_runtime["pipeline_playing"] = False
            camera_switch_runtime["switch_in_progress"] = False
            camera_switch_state["active_secondary"] = bool(initial_secondary)
        _apply_output_flip_method(bool(initial_secondary))
        print(
            "Camera switch: active source -> "
            f"{_camera_label(bool(initial_secondary))} "
            f"(sensor-id={_camera_sensor_id(bool(initial_secondary))}, reason=startup)"
        )
    else:
        pass

    streammux.link(queue_mux_output)

    if preprocess_enabled:
        queue_mux_output.link(preprocess_convert)
        preprocess_convert.link(preprocess_caps)
        preprocess_caps.link(pgie)
    else:
        queue_mux_output.link(pgie)
    pgie.link(tracker)
    tracker.link(queue_tracker_output)
    
    if ENABLE_RAW_RECORDING:
        queue_tracker_output.link(tee_raw_split)
        tee_raw_split.link(queue_raw)
        queue_raw.link(nvvidconv_raw)
        nvvidconv_raw.link(encoder_raw)
        encoder_raw.link(parser_raw)
        parser_raw.link(container_raw)
        container_raw.link(file_sink_raw)
        tee_raw_split.link(queue_osd_input)
        queue_osd_input.link(nvvidconv)
        nvvidconv.link(nvosd)
    else:
        queue_tracker_output.link(nvvidconv)
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
            Gst.PadProbeType.BUFFER, roi_inference_sink_pad_buffer_probe, 0
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
    global start_time, frame_count, fps_window_start_time, fps_last_reported
    start_time = time.time()
    frame_count = 0
    fps_window_start_time = start_time
    fps_last_reported = None

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
        try:
            _stop_power_logging(ros_node)
        except Exception:
            pass
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
            
    # Redis stop flag removed.
    
    bus.connect("message", on_message)

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    ros_thread = threading.Thread(target=spin_ros_node, daemon=True)
    pipeline_started = False

    try:
        ros_thread.start()

        if REQUIRE_ARM_FOR_INFERENCE:
            print(
                "Waiting for /mavros/state armed=True before starting inference..."
            )
            while (
                not shutdown_event.is_set()
                and not ros_node.inference_ready_event.wait(timeout=0.2)
            ):
                pass

            if shutdown_event.is_set():
                return
        elif DEBUG_INFERENCE_MODE:
            print("Debugger detected; starting inference without arming.")
        elif SKIP_ARMING_CHECK:
            print(
                "PYTHON_TEST_SKIP_ARMING_CHECK=1; starting inference without arming."
            )
        elif SOURCE_TYPE != "camera":
            print("Non-camera source selected; starting inference without arming.")

        # Start pipeline
        print("Starting pipeline...")
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Failed to start pipeline")
            return

        with camera_switch_lock:
            camera_switch_runtime["pipeline_playing"] = True
        pipeline_started = True
        loop.run()
    finally:
        shutdown_event.set()
        try:
            _stop_power_logging(ros_node)
        except Exception:
            pass
        if pipeline_started:
            # Send EOS event to properly close MP4 file.
            print("Sending EOS event to the pipeline...")
            pipeline.send_event(Gst.Event.new_eos())
            # Wait a bit for EOS to be processed.
            time.sleep(2)
            # Print final statistics.
            total_time = time.time() - start_time
            if total_time > 0:
                print(f"Total frames processed: {frame_count}")
                print(f"Total time: {total_time:.2f} seconds")
                print(f"Average FPS: {frame_count / total_time:.2f}")
        # Redis stop flag removed.
        
        # Cleanup
        print("Setting pipeline to NULL state...")
        with camera_switch_lock:
            camera_switch_runtime["pipeline_playing"] = False
        pipeline.set_state(Gst.State.NULL)
        _reset_camera_switch_runtime()
        _shutdown_inference_roi_recorder()
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
