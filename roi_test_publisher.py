#!/usr/bin/env python3

import math
import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import RegionOfInterest


ROI_TOPIC = os.getenv("ROI_TEST_TOPIC", "/interceptor/camera/inference_roi").strip()
FRAME_WIDTH = max(1, int(os.getenv("ROI_TEST_FRAME_WIDTH", "1920")))
FRAME_HEIGHT = max(1, int(os.getenv("ROI_TEST_FRAME_HEIGHT", "1080")))
ROI_WIDTH = max(32, int(os.getenv("ROI_TEST_WIDTH", "640")))
ROI_HEIGHT = max(32, int(os.getenv("ROI_TEST_HEIGHT", "384")))
PUBLISH_HZ = max(1.0, float(os.getenv("ROI_TEST_HZ", "10.0")))
SWEEP_PERIOD_SEC = max(1.0, float(os.getenv("ROI_TEST_SWEEP_PERIOD", "8.0")))
VERTICAL_SWING_PX = max(0, int(os.getenv("ROI_TEST_VERTICAL_SWING", "120")))


def clamp(value, low, high):
    return max(low, min(high, value))


class RoiTestPublisher(Node):
    def __init__(self):
        super().__init__("roi_test_publisher")
        self.publisher = self.create_publisher(RegionOfInterest, ROI_TOPIC, 10)
        self.start_time = self.get_clock().now()
        self.publish_count = 0
        self.timer = self.create_timer(1.0 / PUBLISH_HZ, self.publish_roi)

        self.get_logger().info(
            f"Publishing test ROI to {ROI_TOPIC} at {PUBLISH_HZ:.1f} Hz "
            f"(frame={FRAME_WIDTH}x{FRAME_HEIGHT}, roi={ROI_WIDTH}x{ROI_HEIGHT})"
        )

    def publish_roi(self):
        now = self.get_clock().now()
        elapsed_sec = (now - self.start_time).nanoseconds / 1e9

        max_x = max(0, FRAME_WIDTH - ROI_WIDTH)
        max_y = max(0, FRAME_HEIGHT - ROI_HEIGHT)
        sweep_phase = (elapsed_sec % SWEEP_PERIOD_SEC) / SWEEP_PERIOD_SEC

        x_offset = int(round(max_x * 0.5 * (1.0 + math.sin((2.0 * math.pi * sweep_phase) - (math.pi / 2.0)))))
        center_y = (FRAME_HEIGHT // 2) + int(
            round(VERTICAL_SWING_PX * math.sin(2.0 * math.pi * elapsed_sec / SWEEP_PERIOD_SEC))
        )
        y_offset = clamp(center_y - (ROI_HEIGHT // 2), 0, max_y)

        msg = RegionOfInterest()
        msg.x_offset = clamp(x_offset, 0, max_x)
        msg.y_offset = y_offset
        msg.width = min(ROI_WIDTH, FRAME_WIDTH)
        msg.height = min(ROI_HEIGHT, FRAME_HEIGHT)
        msg.do_rectify = False
        self.publisher.publish(msg)

        self.publish_count += 1
        if self.publish_count % max(1, int(PUBLISH_HZ)) == 0:
            self.get_logger().info(
                f"ROI x={msg.x_offset} y={msg.y_offset} w={msg.width} h={msg.height}"
            )


def main():
    rclpy.init()
    node = RoiTestPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
