# Android console

Phone replacement for `quadruped_operator/console.py`. The phone is a real ROS 2
participant: Fast DDS 2.6.11 (the version ROS 2 Humble uses) cross-compiled for
Android, no bridge on the robot.

- `app/src/main/cpp/`: `ros_link` (one DDS participant, hand-written CDR types for
  `std_msgs` Float32/Bool/String/Float32MultiArray and `geometry_msgs/Vector3`) and its JNI wrapper.
- `ConsoleCore.kt`: the console logic (heartbeat, safety params, e-stop latch, policy pre-flight).
- `ConsoleService.kt`: foreground service with wake + low-latency Wi-Fi locks, runs the heartbeat.
  Disconnect or swiping the app away sends the e-stop, like Ctrl-C on the console.
- Drive tab: two touch sticks publish `geometry_msgs/Twist` on `/cmd_vel/phone` (twist_mux
  priority 80: above Nav2 and skills, below the F710 and keyboard) at 20 Hz while the
  "Phone drive" switch is on, zero when centred, so it holds Nav2 off. Off sends one zero and
  stops, and twist_mux hands back to Nav2 after 0.5 s. E-stop or leaving the app turns it off.

- Robot tab, policy mode: the Go2 URDF drawn with OpenGL ES 3, posed from `/tf` (odom -> base)
  and `/sensors/joint_states`. The URDF and its meshes (decimated, ~1 MB) live in
  `app/src/main/assets/go2/`; regenerate after changing the URDF with
  `unset PYTHONPATH; ~/env_isaacsim/bin/python tools/export_urdf.py`.
- Robot tab: "Robot computer" card from `/system_stats` (CPU per core, GPU, RAM, temperatures),
  published once a second by `quadruped_drivers/system_monitor`, which the real launch starts.
- Shell tab: an SSH terminal on the robot (JSch + Termux's xterm emulator), to start the driver and
  Nav2 without a laptop. The app's ECDSA key lives in its private storage; log in once with the
  password and the app adds the key to `~/.ssh/authorized_keys`. Shortcut chips type common commands
  (editable, no Enter). Start launches inside `tmux new -A -s robot`: a dropped SSH session kills
  whatever runs outside tmux. Heartbeat and e-stop stay on DDS, independent of the shell.
- Map tab: `/map`, `/scan`, `/plan` and the robot (TF map -> base_footprint), RViz-style;
  "Goal" mode sends a drag as `/goal_pose`. These streams are subscribed only while shown,
  best-effort, so they never compete with the heartbeat for retransmits.

Network settings follow `config.yaml`: domain 42, discovery server `10.42.0.1:11811`
used when the phone is on that subnet (`auto`), else multicast.

## Build

Toolchain (once): JDK 17, Android SDK 34, NDK 26.3.11579264, CMake 3.22.1 in `~/Android`.

    export JAVA_HOME=~/Android/jdk17 ANDROID_HOME=~/Android/Sdk
    PATH=$ANDROID_HOME/cmake/3.22.1/bin:$PATH ./build_fastdds.sh   # once, ~5 min per ABI
    ./gradlew testDebugUnitTest assembleDebug
    $ANDROID_HOME/platform-tools/adb install -r app/build/outputs/apk/debug/app-debug.apk

## Checking the DDS side on a PC

`tools/link_check.sh` builds the same `ros_link.cpp` against `/opt/ros/humble` and runs it;
`tools/ros_side.py` plays the robot (`viz_check.cpp` / `viz_side.py` do the same for the map and 3D topics) (fake `/robot_state` etc.) and reports what arrived and
the heartbeat gaps. Use a domain nothing else is on:

    tools/link_check.sh 77 <server_ip:port> <local_ip> &
    ROS_DOMAIN_ID=77 ROS_DISCOVERY_SERVER=<server_ip:port> ROS_SUPER_CLIENT=TRUE python3 tools/ros_side.py 10

The SSH tests run against a throwaway sshd when `SSH_TEST_PORT` and `SSH_TEST_KEYS` are set
(an unprivileged `sshd -f <config>` on 127.0.0.1 with its own host key and `AuthorizedKeysFile`
holding the key in `SSH_TEST_KEYS`); otherwise they are skipped.

## Before trusting it with torque

On the robot, with the phone connected: `ros2 topic hz /safety/heartbeat` and watch the max
gap while walking around with the phone. It must stay well under `watchdog_timeout` (0.35 s).
