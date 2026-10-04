#!/usr/bin/env bash
# Build and run link_check against the system ROS 2 Humble Fast DDS.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
R=/opt/ros/humble
g++ -std=c++17 -O1 -I"$HERE/../app/src/main/cpp" -I$R/include -I$R/include/fastcdr -I$R/include/fastrtps \
    "$HERE/link_check.cpp" "$HERE/../app/src/main/cpp/ros_link.cpp" \
    -L$R/lib -Wl,-rpath,$R/lib -lfastrtps -lfastcdr -lpthread -o "$HERE/link_check"
exec "$HERE/link_check" "$@"
