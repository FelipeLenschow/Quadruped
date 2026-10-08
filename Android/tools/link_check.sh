#!/usr/bin/env bash
# Build and run link_check against the system ROS 2 Humble Fast DDS.
#   link_check.sh domain [server_ip:port|""] [local_ip]   console topics, with ros_side.py
#   link_check.sh viz domain [local_ip]                   map, 3D and LIO topics, with viz_side.py
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
R=/opt/ros/humble
TOOL=link_check
if [ "$1" = viz ]; then TOOL=viz_check; shift; fi
g++ -std=c++17 -O1 -I"$HERE/../app/src/main/cpp" -I$R/include -I$R/include/fastcdr -I$R/include/fastrtps \
    "$HERE/$TOOL.cpp" "$HERE/../app/src/main/cpp/ros_link.cpp" \
    -L$R/lib -Wl,-rpath,$R/lib -lfastrtps -lfastcdr -lpthread -o "$HERE/$TOOL"
exec "$HERE/$TOOL" "$@"
