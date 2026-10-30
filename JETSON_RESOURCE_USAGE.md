# Go2 Jetson: resource usage and on-board LLM

Measurements taken on 2026-10-30/31 on the Go2's Jetson while adding the launcher's stacks one at a time, then
trying a vision-language model (Qwen3-VL) on board for the [A] LLM agent.

## Hardware

| | |
|---|---|
| Board | NVIDIA Orin Nano 8 GB (7337 MB usable, shared by CPU and GPU) |
| CPUs | 6 cores; the real driver reserves core 5 (SCHED_FIFO 80) |
| Software | JetPack 5.1.1 (L4T R35.3.1), CUDA 11.4, power mode 15 W |
| Swap | 3.7 GB |

## Method

Each run is 15 s, sampled every 0.5 s on the robot:

- `tegrastats`: CPU load per core, GPU (GR3D) load, RAM.
- `top -b`: CPU and resident memory per process.
- `/proc/net/dev`: traffic on `wlan0` (laptop/app) and `eth0` (Go2 internal network: lidar, camera, LowState).

Process CPU is a percentage of **one** core. The script and raw logs are in `~/usage_logs/` on the robot.

## Results

Each column adds to the one before it, unless the header says otherwise. LIO = Point-LIO ([L]).

| | Idle | Policy [6] moving | Policy [6] still | + [N] stack | + LIO | + LIO steady | + Camera [I] (YOLO) | Camera, no YOLO | No YOLO, no LIO, + Qwen3-VL-2B |
|---|---|---|---|---|---|---|---|---|---|
| **CPU (6-core average)** | 0.7% | 12.1% | 10.3% | 32.4% | 36.0% | 37.6% | 46.0% | 42.3% | 44.2% |
| CPU busiest core (peak) | 7% | 19% | 25% | 55% | 90% | 60% | 93% | 76% | 92% |
| **GPU** | 0% | 0% | 0% | 0% | 0% | 0% | 29% (bursts to 99%) | 0% | 58% (86–99% while answering) |
| **RAM used** (of 7337 MB) | 1011 | 1250 | 1256 | 1504 | 1730 | 1780 | **5640** | 2920 | **5830** + 460 MB swap |
| WiFi in / out (KB/s) | – | 19 / 27 | 20 / 28 | 17 / 85 | 20 / 127 | 20 / 116 | 20 / 356 | 20 / 335 | 20 / 275 |
| `eth0` in / out (KB/s) | – | 2834 / 185 | 2839 / 185 | 2845 / 184 | 2852 / 184 | 2843 / 183 | 5576 / 185 | 5562 / 185 | 4693 / 187 |
| `real_driver` (policy) | – | 20% | 19% | 21% | 20% | 20% | 20% | 20% | 18% |
| `real_sensors` | – | 11% | 11% | 11% | 13% | 13% | 14% | 14% | 10% |
| `mcp_server` | – | 18% | 18% | 45% | 27% | 28% | 28% | 28% | 25% |
| `lidar_filter` | – | – | – | 33% | 35% | 32% | 34% | 34% | 34% |
| nav2 `controller_server` | – | – | – | 21% | 7% | 7% | 7% | 7% | 7% |
| nav2 `bt_navigator` | – | – | – | 16% | 9% | 9% | 9% | 9% | 8% |
| `pointlio_mapping` | – | – | – | – | 38% / 162 MB | 37% / 208 MB | 37% / 313 MB | 38% / 349 MB | off |
| `lio_map_stream` | – | – | – | – | 22% | 21% | 20% | 22% | off |
| `camera_bridge` | – | – | – | – | – | – | 23% / 84 MB | 24% / 78 MB | 22% |
| `object_detector` (YOLO) | – | – | – | – | – | – | 21% / **3789 MB** | – | – |
| `llama-server` (Ollama) | – | – | – | – | – | – | – | – | **84%** / 3.5 GB GPU memory |

## What costs what

| Component | CPU (6-core average) | RAM | GPU | Network |
|---|---|---|---|---|
| Policy [6] | +11% | +240 MB | – | `eth0` 2.8 MB/s in (LowState + lidar) |
| [N] Nav2 + SLAM | +20% | +250 MB | – | WiFi out +60 KB/s |
| [L] Point-LIO | +4–5% | +230 MB, **growing** | – | WiFi out +40 KB/s (map stream) |
| [I] camera | +5% | +100 MB | – | `eth0` +2.7 MB/s, WiFi out +220 KB/s |
| YOLO (`object_detector`) | +4% | **+2.7 to 3.8 GB** | about 30%, in bursts | – |
| Qwen3-VL-2B (Ollama, fully on GPU) | +14% (one core busy-waiting) | **about 3.5 GB** | 90–99% while answering | – |

Whether the robot moves or stands still makes no difference (runs 2 and 3): the policy and drivers run at a fixed rate.

## Qwen3-VL on the Jetson

| Model | GPU memory (real) | Image prompt | Generation | Fits next to the stack? |
|---|---|---|---|---|
| Qwen3-VL-2B instruct, 4-bit, 4k context | 3.5 GB | about 1100 tokens, 6.1 s (cached: 0.1 s) | 22.6 tokens/s | only without YOLO |
| Qwen3-VL-4B, 4-bit | about 5 GB (estimated) | – | – | no |

- `ollama ps` reports 1.8 GB for the 2B model. The real figure is in
  `sudo cat /sys/kernel/debug/nvmap/iovmm/clients`, which also counts the CUDA libraries and the vision encoder's buffers.
- The model must be created with `PARAMETER num_gpu 99`. Otherwise Ollama counts only free memory, not the file
  cache, decides the model does not fit, and puts half its layers on the CPU.
- `llama-server` keeps about one CPU core busy while the GPU works.
- The 2B model is weak at choosing tools. Given "walk forward a bit", it never called `move`; it looped on
  `get_state`, `frontiers` and `list_objects`, and called `look(boxes=true)` with the detector off.
  Since then `mcp_server` hides the object tools (`list_objects`, `go_to_object`, `visit`, `look`'s `boxes`) while
  `object_mapper` is not publishing `/objects` (`mcp.objects: auto` in config.yaml; `true`/`false` or `--objects on|off`
  force it). Without them, `llm_host`'s default task is exploring and describing what it sees, instead of cone visiting.

## Problems found

### 1. Robot twitched, tripped and went numb with YOLO and Qwen running together (open)

With YOLO and Qwen-2B running and the robot walking, the app showed 6.2/7.2 GB RAM, 99% GPU and 34% CPU. The robot
twitched, moved forward a little, tripped the joint-position safety, and went **numb** instead of lying down with the
safety policy.

- The logs were lost: the container runs with `--rm` (`~/.ros/log` went with it), and the system log is not kept
  across reboots.
- Most likely cause: memory ran out and the control loop stalled on swap. `real_driver` did not lock its memory, so
  its pages could be swapped out under pressure even though its core and priority are reserved. A second possible
  cause is the GPU competing with the CPU for memory bandwidth (CPU and GPU share the RAM).
- Ways a trip ends in numb rather than lying down (`controller.cpp`): no safety policy loaded; the safety policy runs
  10 s without lying still; or `control step failed: …` (an error in the policy step, for example a failed memory
  allocation). Which one happened is unknown.
- **Fix pushed:** `6587036` locks the driver's memory (`mlockall(MCL_CURRENT | MCL_FUTURE | MCL_ONFAULT)`) and logs
  `Memory locked in RAM` or the reason it failed. Not yet tested on the robot.
- **To confirm:** repeat the test and watch `[RealDriver] Loop, last 10 s: … ms max, … late, … skipped`. If these stay
  near 1 ms under full load, swapping was the cause. If they still jump, look at memory-bandwidth contention.
- **Until then:** do not run YOLO and an LLM together on the robot.

### 2. Point-LIO memory grows without limit

162 → 208 → 313 → 349 MB in about 10 minutes, and 629 MB later in the same session. The map has no size limit.
On long runs this alone will exhaust the RAM. Check the map/cube size settings in the Point-LIO config.

### 3. YOLO uses 3.8 GB

That is very large for a detector. A TensorRT-exported YOLO usually needs a few hundred MB. Slimming it down would
make room for the LLM.

### 4. `mcp_server` uses 18–45% of a core while idle

It uses about as much CPU as the policy driver, even with no client connected. This suggests a polling loop.

### 5. [A] asked Ollama for a 32k context

`llm_host.py` defaults to `--ctx 32768` and the launcher did not pass `--ctx`, so Ollama reloaded the 2B model with
32k context and left 480 MB of RAM. **Fixed in `4b6475d`:** on the robot, [A] defaults to `qwen3vl-2b-gpu`, 8192 tokens
and 12 kept messages, and asks for the context size.

### 6. Smaller pitfalls

- **Logs lost on reboot.** Run `sudo mkdir -p /var/log/journal && sudo systemctl restart systemd-journald` once on the
  robot, and add `-e ROS_LOG_DIR=/app/ros_logs` to the `quaddocker` alias so ROS logs land in the repo.
- **`docker exec` shells cannot see the stack's topics.** They default to ROS domain 1 without the discovery server.
  Even with `Tools/ros_env.sh` sourced, a new process never received camera frames. Workaround: get a frame through
  the MCP server's `look` tool over HTTP (port 8765). Not investigated further.
- **Ctrl+C does not stop `docker exec` commands without `-t`.** The process inside the container keeps running.
- **`ollama run` image paths.** The CLI only attaches an image when the prompt has no other `/`: "left/center/right"
  in the prompt stopped it attaching. Look for the `Added image …` line. The API and `llm_host` are unaffected.
- **The Ollama service already runs** under systemd, so a manual `ollama serve` fails with "address already in use".
  Settings are in `/etc/systemd/system/ollama.service.d/override.conf`.

## Open next steps

1. Rebuild `quadruped_real` on the robot with `6587036` and check for the `Memory locked in RAM` log line.
2. Repeat the YOLO + Qwen + walking test with the loop statistics and the usage recorder running.
3. Limit Point-LIO's map size.
4. Pin `llama-server` away from core 5 (`CPUAffinity=0-4` in the Ollama service override).
5. To keep both the LLM and object grounding: a lighter YOLO (TensorRT), or drop YOLO and add an MCP tool that turns
   a pixel or bearing from the photo into a map point using the lidar.
