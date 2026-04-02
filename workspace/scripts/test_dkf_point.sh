#!/usr/bin/env bash
set -euo pipefail

if ! command -v ros2 >/dev/null 2>&1; then
    echo "ros2 command not found. Source your ROS 2 environment first." >&2
    exit 1
fi

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: $0 <x_bar> <y_bar> [rate_hz] [topic]" >&2
    echo "Example: $0 0.10 -0.05 10 /eskf_reduced/pbar" >&2
    exit 1
fi

X_BAR="$1"
Y_BAR="$2"
RATE_HZ="${3:-10}"
TOPIC="${4:-/eskf_reduced/pbar}"
Z_VALUE="0.0"

read -r PIXEL_X PIXEL_Y <<EOF
$(python3 - "$X_BAR" "$Y_BAR" <<'PY'
import sys
x_bar = float(sys.argv[1])
y_bar = float(sys.argv[2])
focal_x = 1238.10428
focal_y = 1238.78782
c_x = 960.0
c_y = 540.0
pixel_x = x_bar * focal_x + c_x
pixel_y = y_bar * focal_y + c_y
print(f"{pixel_x:.1f} {pixel_y:.1f}")
PY
)
EOF

echo "Publishing DKF test point"
echo "  topic : ${TOPIC}"
echo "  pbar  : (${X_BAR}, ${Y_BAR}, ${Z_VALUE})"
echo "  pixel : (${PIXEL_X}, ${PIXEL_Y})"
echo "Press Ctrl+C to stop."

exec ros2 topic pub "${TOPIC}" geometry_msgs/msg/Point "{x: ${X_BAR}, y: ${Y_BAR}, z: ${Z_VALUE}}" -r "${RATE_HZ}"
