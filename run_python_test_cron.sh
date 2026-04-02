#!/bin/bash

set -eo pipefail

export HOME=/home/ituarc
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/humble/setup.bash
source /home/ituarc/ros2_ws/install/setup.bash

cd "$SCRIPT_DIR"
exec /usr/bin/python python_test.py
