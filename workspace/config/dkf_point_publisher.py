#!/usr/bin/env python3

import argparse

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point


DEFAULT_TOPIC = "/eskf_reduced/pbar"
FOCAL_X = 1238.10428
FOCAL_Y = 1238.78782
C_X = 960.0
C_Y = 540.0


def pbar_to_pixel(x_bar: float, y_bar: float) -> tuple[float, float]:
    pixel_x = x_bar * FOCAL_X + C_X
    pixel_y = y_bar * FOCAL_Y + C_Y
    return pixel_x, pixel_y


class DKFPointPublisher(Node):
    def __init__(self, topic: str, x_bar: float, y_bar: float, z_value: float, rate_hz: float):
        super().__init__("dkf_point_publisher")
        self.publisher = self.create_publisher(Point, topic, 10)
        self.topic = topic
        self.rate_hz = rate_hz

        self.msg = Point()
        self.msg.x = float(x_bar)
        self.msg.y = float(y_bar)
        self.msg.z = float(z_value)

        self.publish_count = 0
        self.timer = self.create_timer(1.0 / rate_hz, self.publish_point)

        pixel_x, pixel_y = pbar_to_pixel(self.msg.x, self.msg.y)
        self.get_logger().info(
            f"Publishing to {self.topic}: "
            f"pbar=({self.msg.x:.4f}, {self.msg.y:.4f}, {self.msg.z:.4f}) "
            f"-> pixel=({pixel_x:.1f}, {pixel_y:.1f})"
        )

    def publish_point(self):
        self.publisher.publish(self.msg)
        self.publish_count += 1

        if self.publish_count == 1 or self.publish_count % max(1, round(self.rate_hz)) == 0:
            pixel_x, pixel_y = pbar_to_pixel(self.msg.x, self.msg.y)
            self.get_logger().info(
                f"sent #{self.publish_count}: "
                f"pbar=({self.msg.x:.4f}, {self.msg.y:.4f}) "
                f"pixel=({pixel_x:.1f}, {pixel_y:.1f})"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publish a DKF pbar test point and print its pixel coordinates."
    )
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help=f"ROS topic (default: {DEFAULT_TOPIC})")
    parser.add_argument("--x", type=float, required=True, help="pbar x value")
    parser.add_argument("--y", type=float, required=True, help="pbar y value")
    parser.add_argument("--z", type=float, default=0.0, help="Point.z value (default: 0.0)")
    parser.add_argument("--rate", type=float, default=10.0, help="Publish rate in Hz (default: 10)")
    parser.add_argument("--once", action="store_true", help="Publish once and exit")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.rate <= 0.0:
        raise SystemExit("--rate must be > 0")

    rclpy.init()
    node = DKFPointPublisher(args.topic, args.x, args.y, args.z, args.rate)

    try:
        if args.once:
            node.publish_point()
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
