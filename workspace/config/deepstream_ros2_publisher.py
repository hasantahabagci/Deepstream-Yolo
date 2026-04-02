#!/usr/bin/env python3
"""
DeepStream + ROS 2 Publisher
============================
Runs the DeepStream pipeline and publishes the best detected object's
bounding-box center as a geometry_msgs/msg/Point on the ROS 2 topic
  /detection/bbox_center   (x=cx, y=cy, z=confidence)

Optionally also publishes ALL detections as a JSON string on
  /detection/all_detections

Usage (inside a sourced ROS 2 workspace):
    python3 deepstream_ros2_publisher.py [--source file] [--path video.mp4]
    python3 deepstream_ros2_publisher.py [--source argus]
    python3 deepstream_ros2_publisher.py --help
"""

import sys
import os
import time
import signal
import argparse
import threading
import json
from datetime import datetime

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepstream_runner import DeepStreamRunner, DeepStreamConfig, FrameDetections
from project_paths import DEFAULT_VIDEO_INPUT, ROOT_PGIE_CONFIG_YOLO11


class BBoxPublisherNode(Node):
    """
    Minimal ROS 2 node that exposes two publishers:
      - /detection/bbox_center  (geometry_msgs/Point)
          x = horizontal center in pixels
          y = vertical center in pixels
          z = detection confidence [0..1]
      - /detection/all_detections  (std_msgs/String, JSON array)
    """

    def __init__(self):
        super().__init__("deepstream_bbox_publisher")

        self._center_pub = self.create_publisher(
            Point, "/detection/bbox_center", 40
        )
        self._all_pub = self.create_publisher(
            String, "/detection/all_detections", 10
        )

        self.get_logger().info("BBoxPublisherNode started.")
        self.get_logger().info(
            "Publishing best-detection center -> /detection/bbox_center"
        )
        self.get_logger().info(
            "Publishing all detections (JSON)  -> /detection/all_detections"
        )

    def publish_frame(self, frame: FrameDetections):
        """
        Called from the DeepStream detection callback (non-ROS thread).
        Publishes bbox center of the best detection and all detections.
        """
        if frame.best_detection is not None:
            det = frame.best_detection
            cx = det.left + det.width / 2.0
            cy = det.top + det.height / 2.0

            pt = Point()
            pt.x = float(cx)
            pt.y = float(cy)
            pt.z = float(det.confidence)
            self._center_pub.publish(pt)
        else:
            pt = Point()
            pt.x = float("nan")
            pt.y = float("nan")
            pt.z = 0.0
            self._center_pub.publish(pt)

        msg = String()
        if frame.detections:
            detections_list = []
            for d in frame.detections:
                cx = d.left + d.width / 2.0
                cy = d.top + d.height / 2.0
                detections_list.append(
                    {
                        "object_id": int(d.object_id),
                        "class_id": int(d.class_id),
                        "confidence": float(d.confidence),
                        "bbox": {
                            "left": float(d.left),
                            "top": float(d.top),
                            "width": float(d.width),
                            "height": float(d.height),
                        },
                        "center": {"x": float(cx), "y": float(cy)},
                    }
                )

            msg.data = json.dumps(
                {
                    "frame_number": int(frame.frame_number),
                    "timestamp": float(frame.timestamp),
                    "fps": float(frame.fps),
                    "num_objects": int(frame.num_objects),
                    "detections": detections_list,
                }
            )
        else:
            msg.data = json.dumps(
                {
                    "frame_number": int(frame.frame_number),
                    "timestamp": float(frame.timestamp),
                    "fps": float(frame.fps),
                    "num_objects": 0,
                    "detections": None,
                }
            )
        self._all_pub.publish(msg)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DeepStream pipeline and publish bbox centers to ROS 2"
    )
    parser.add_argument(
        "--source",
        default="argus",
        choices=["argus", "v4l2", "file", "appsrc"],
        help="Video source type (default: file)",
    )
    parser.add_argument(
        "--path",
        default=str(DEFAULT_VIDEO_INPUT),
        help="Source path for 'file' or 'v4l2' source types",
    )
    parser.add_argument(
        "--width", type=int, default=1920, help="Frame width (default: 1920)"
    )
    parser.add_argument(
        "--height", type=int, default=1080, help="Frame height (default: 1080)"
    )
    parser.add_argument(
        "--fps", type=int, default=40, help="Frame rate (default: 40)"
    )
    parser.add_argument(
        "--pgie-config",
        default=str(ROOT_PGIE_CONFIG_YOLO11),
        help="Path to primary inference config file",
    )
    parser.add_argument(
        "--onnx",
        default="",
        help="Path to ONNX model file (overrides pgie config)",
    )
    parser.add_argument(
        "--output",
        default="auto",
        help="Output MP4 path for annotated recording. "
             "'auto' = timestamped file in logs/videos, '' = disabled",
    )
    parser.add_argument(
        "--no-tracker",
        action="store_true",
        help="Disable NvDCF object tracker",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        default=False,
        help="Enable live display window (default: disabled)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=-1.0,
        help="Detection confidence threshold (default: use config value)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    ros_node = BBoxPublisherNode()

    ros_thread = threading.Thread(
        target=lambda: rclpy.spin(ros_node), daemon=True
    )
    ros_thread.start()

    LOGS_DIR = "/home/ituarc/ros2_ws/src/thermal_guidance/logs"
    VIDEO_LOG_DIR = os.path.join(LOGS_DIR, "videos")
    LOG_ID_FILE = os.path.join(LOGS_DIR, "last_log_id.txt")

    output_path = args.output
    if output_path == "auto":
        os.makedirs(VIDEO_LOG_DIR, exist_ok=True)

        log_id = 0
        try:
            with open(LOG_ID_FILE, "r") as f:
                log_id = int(f.read().strip())
        except (FileNotFoundError, ValueError):
            print(f"Warning: Could not read {LOG_ID_FILE}, starting from 000")

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(
            VIDEO_LOG_DIR,
            f"{log_id:03d}_deepstream_flight_video_annotated_{ts}.mp4",
        )
    elif not output_path.strip():
        output_path = ""

    ds_config = DeepStreamConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        source_type=args.source,
        source_path=args.path,
        pgie_config_path=args.pgie_config,
        onnx_file=args.onnx,
        enable_tracker=not args.no_tracker,
        enable_display=args.display,
        enable_recording=bool(output_path),
        output_path=output_path,
        pre_cluster_threshold=args.threshold,
    )

    runner = DeepStreamRunner(ds_config)
    runner.register_detection_callback(ros_node.publish_frame)

    stop_event = threading.Event()

    def _handle_shutdown(signum, _frame):
        ros_node.get_logger().info(
            f"Received signal {signum}, stopping DeepStream publisher."
        )
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    try:
        if not runner.start():
            raise RuntimeError("Failed to start DeepStream pipeline")

        while rclpy.ok() and runner.is_running() and not stop_event.is_set():
            time.sleep(0.1)
    finally:
        runner.stop()
        ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        ros_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
