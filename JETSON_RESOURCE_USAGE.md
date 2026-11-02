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

The last four columns (2026-11-02/03, prompted YOLOE with a TensorRT engine, [6] + [N] + [I] with detection and the
object mapper) were sampled more briefly: per-core load from `/proc/stat` over 5 s, `top` over 3-4 s, `free -m`;
the network was not measured. The CPU average includes core 5. n/m: not measured.

## Results

Each column adds to the one before it, unless the header says otherwise. LIO = Point-LIO ([L]).

| | Idle | Policy [6] moving | Policy [6] still | + [N] stack | + LIO | + LIO steady | + Camera [I] (YOLO) | Camera, no YOLO | No YOLO, no LIO, + Qwen3-VL-2B | [I] YOLOE engine + mapper + lidar check | Same, no lidar check | + OpenBLAS one thread | + detector 2 Hz, half-size decode |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **CPU (6-core average)** | 0.7% | 12.1% | 10.3% | 32.4% | 36.0% | 37.6% | 46.0% | 42.3% | 44.2% | 69% | 60% | **41%** | **25%** |
| CPU busiest core (peak) | 7% | 19% | 25% | 55% | 90% | 60% | 93% | 76% | 92% | 87% | 77% | 60% | 35% |
| **GPU** | 0% | 0% | 0% | 0% | 0% | 0% | 29% (bursts to 99%) | 0% | 58% (86–99% while answering) | 0% | n/m | 0% | n/m |
| **RAM used** (of 7337 MB) | 1011 | 1250 | 1256 | 1504 | 1730 | 1780 | **5640** | 2920 | **5830** + 460 MB swap | 5107 | n/m | 5077-5188, 18 MB swap | 4646, 656 MB swap |
| WiFi in / out (KB/s) | – | 19 / 27 | 20 / 28 | 17 / 85 | 20 / 127 | 20 / 116 | 20 / 356 | 20 / 335 | 20 / 275 | n/m | n/m | n/m | n/m |
| `eth0` in / out (KB/s) | – | 2834 / 185 | 2839 / 185 | 2845 / 184 | 2852 / 184 | 2843 / 183 | 5576 / 185 | 5562 / 185 | 4693 / 187 | n/m | n/m | n/m | n/m |
| `real_driver` (policy) | – | 20% | 19% | 21% | 20% | 20% | 20% | 20% | 18% | 25% | 18% | 20% | 22% |
| `real_sensors` | – | 11% | 11% | 11% | 13% | 13% | 14% | 14% | 10% | 10% | 9% | 11% | 12% |
| `mcp_server` | – | 18% | 18% | 45% | 27% | 28% | 28% | 28% | 25% | n/m | n/m | under 5% | 4% |
| `lidar_filter` | – | – | – | 33% | 35% | 32% | 34% | 34% | 34% | 44% | 30% | 42% | 35% |
| nav2 `controller_server` | – | – | – | 21% | 7% | 7% | 7% | 7% | 7% | 7% | 7% | 8% | 8% |
| nav2 `bt_navigator` | – | – | – | 16% | 9% | 9% | 9% | 9% | 8% | 7% | 8% | 8% | 9% |
| `pointlio_mapping` | – | – | – | – | 38% / 162 MB | 37% / 208 MB | 37% / 313 MB | 38% / 349 MB | off | off | off | off | off |
| `lio_map_stream` | – | – | – | – | 22% | 21% | 20% | 22% | off | off | off | off | off |
| `camera_bridge` | – | – | – | – | – | – | 23% / 84 MB | 24% / 78 MB | 22% | 8% | 14% | 9% / 79 MB | 8% |
| `object_detector` (YOLO) | – | – | – | – | – | – | 21% / **3789 MB** | – | – | 22% / 3.6 GB | 20% | 20% / **3716 MB** | **9%** |
| `llama-server` (Ollama) | – | – | – | – | – | – | – | – | **84%** / 3.5 GB GPU memory | – | – | – | – |
| `object_mapper` | – | – | – | – | – | – | n/m | – | – | **148%** | **144%** | 39% / 104 MB | 32% |
| `camera_lidar_check` | – | – | – | – | – | – | n/m | – | – | **116%** | off | off | off |

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

**The TensorRT engine did not change it** (2026-11-03): with the engine, `object_detector` holds 3716 MB, against
3789 MB before. The engine makes the detector faster, not smaller. Most of the memory is the CUDA libraries, not the
model: in a new process, `import torch` takes 257 MB and the first CUDA tensor raises it to **1218 MB** (JetPack 5's
CUDA 11.4 loads every kernel at start; lazy loading needs CUDA 11.7). The detector also loads ultralytics (which
imports torch) and TensorRT's cuBLAS/cuDNN. To make it small, run the engine with TensorRT's own Python API, without
torch or ultralytics in the process.

### 4. `mcp_server` uses 18–45% of a core while idle

It uses about as much CPU as the policy driver, even with no client connected. This suggests a polling loop.

**Cause and fix** (`b47de0c`): not a loop. Its Python TF listener decodes every `/tf` message (100 per second on the
robot). A test node with only a TF listener used 20% of a core (28% with the `MultiThreadedExecutor`). `mcp_server`
now listens to TF only around tool calls. `lidar_filter`, `object_mapper` and `camera_lidar_check` pay the same
20% for their own listeners.

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

### 7. numpy's OpenBLAS threads spin (fixed)

`object_mapper` used 144-148% of a core for about 36% of real work: numpy's OpenBLAS ran a worker thread per core,
and the workers spin while they wait between products. The new class list made it worse (a detection in nearly
every frame, so the products run at 5 Hz). The same work (100k points at 5 Hz) took 151% of a core and 41 ms per
frame with the default threads, and **5% and 12 ms** with `OPENBLAS_NUM_THREADS=1`. The launcher sets it on the
robot, and the Jetson image sets it for shells. The cores went from 68-77% to 40-60%.

### 8. `camera_lidar_check` costs more than a core

116% of a core: it projects the last 20 raw clouds onto a full-size image and draws each point with a Python call.
It is only for checking the camera mount (checked on 2026-11-03: correct to about 0.5 degrees). Answer N to it
in [I] at other times.

### 9. Detector rate and image size: a small gain

`detector.rate` 5 to 2 Hz, and the JPEG decoded at half size (960 x 540, still above the model's 640): the detector
went from 20% to 9% of a core and `object_mapper` from 39% to 32%, about 18% of a core in all. The cores fell from
40-60% to 26-35%, but the earlier sample was taken soon after a restart, with pages still coming back from swap, so
most of that drop is noise. The mapper hardly changes because most of its cost does not depend on the detection
rate: its TF listener (about 20%) and its cloud callback, which moves every filtered cloud into the map frame at the
lidar's rate. The model input (`imgsz` 640) was kept: the GPU is not the limit (32 ms per frame), and smaller
inputs make far objects too small.

After this the detection is cheap, and most of the load is the rest of the stack: `lidar_filter` 35%, the mapper
32%, `real_driver` 22%, Nav2 idle about 40% in all, `real_sensors` 12%. The largest avoidable part is the two
Python TF listeners, about 40% of a core together (problem 4).

## Plan A: the detector without torch (not started)

Goal: keep the YOLOE detection and the object map, and make room for Qwen3-VL-2B (3.5 GB) next to the stack.
The detector holds 3716 MB, almost all of it libraries: the engine file is 23 MB, but the process loads torch
and its CUDA context (1218 MB measured in a new process), ultralytics, and cuDNN/cuBLAS for TensorRT. JetPack 5's
CUDA 11.4 loads every kernel of a library at start (lazy loading needs CUDA 11.7).

1. **Measure first:** a process that loads only `tensorrt` and the engine, with `[I]` stopped (a second detector-sized
   process next to the stack pushed 2.6 GB into swap). If it is not well under 1 GB, stop here.
2. **Engine without cuDNN and cuBLAS:** build it with those tactic sources off (`IBuilderConfig.set_tactic_sources`,
   or ultralytics' export with a TensorRT config), so TensorRT does not load those libraries at run time. Check that
   the GPU time stays near the current 32 ms per frame.
3. **GPU buffers without torch:** the container has neither `pycuda` nor `cuda-python`. Either install
   `cuda-python` (a version that works with the CUDA 11.4 driver) in `Dockerfile.jetson`, or call `cudaMalloc` /
   `cudaMemcpy` from `libcudart.so` with `ctypes`, which needs no new package.
4. **Pre-processing in numpy/cv2:** letterbox to 640, BGR to RGB, scale to 0-1, HWC to CHW, as ultralytics does.
5. **Post-processing in numpy:** YOLO26 is NMS-free, so this is the top-scoring boxes over `conf`, the class index,
   and the masks (the 32 mask coefficients times the prototype, sigmoid, crop to the box, scale back). The output
   must stay the same JSON, so `object_mapper` and the app need no change.
6. **Check against ultralytics:** the same frames through both, with the same boxes and scores within a small tolerance.
7. **Class list:** the text embeddings are baked into the engine at export (`set_classes` runs CLIP), so a new class
   list still needs `--export`, which can keep using ultralytics and torch (offline, once).

Expected result (an estimate, not measured): the detector under 1 GB, the stack with detection about 2.5-3 GB,
and Qwen3-VL-2B fitting with about 1 GB to spare. The alternative is plan B: no YOLO, Qwen grounds objects itself
(it can return a box for a description), and an MCP tool turns that box into a map point with the lidar.

## Open next steps

0. Python TF listeners (about 20% of a core each, problem 4), from small to large: lower the TF rate (`slam.yaml`
   `transform_publish_period` 0.02 to 0.05 takes `/tf` from 100 to 70 messages per second; `real_driver` sends
   `odom -> base` at 50 Hz); keep `object_mapper`'s cloud in the base frame and look up one transform per detection
   frame; port `lidar_filter` to C++ like `real_sensors`.
1. Rebuild `quadruped_real` on the robot with `6587036` and check for the `Memory locked in RAM` log line.
2. Repeat the YOLO + Qwen + walking test with the loop statistics and the usage recorder running.
3. Limit Point-LIO's map size.
4. Pin `llama-server` away from core 5 (`CPUAffinity=0-4` in the Ollama service override).
5. To keep both the LLM and object grounding: plan A (the detector without torch, above), or plan B (no YOLO; Qwen
   grounds objects and an MCP tool turns its box into a map point using the lidar).
