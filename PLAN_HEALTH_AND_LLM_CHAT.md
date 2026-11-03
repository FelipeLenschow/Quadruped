# Plan: robot health telemetry and the app's LLM chat

Two upgrades. Do upgrade 1 first: it is about safety, and it is smaller.

## Upgrade 1: battery and motor temperature

### Why

On 2026-11-03 at about 07:39, the robot stood still under the `main` policy and fell on its chin. The logs show:

| Time | Event |
|---|---|
| 07:38:33 | Pee → `main`. Then the robot stood still. |
| 07:38:33 – 07:40:17 | `real_driver` loop normal: 0 skipped ticks, `main` sent commands, no trip, no numb. |
| In this time | The motors went limp (damping only), and the robot fell forward. The Go2's light blinked red. |
| 07:40:19 | ROM trip (`RL_calf_joint -2.547 < -2.545`): the operator moved the limp legs. `Lie2`, then numb. |

Our code sent correct commands, so the Go2's own firmware stopped the motors. The probable cause is motor
temperature: the front legs failed first, and the knee motors were still warm some minutes later. Low battery
voltage is the other possible cause. We cannot prove which one, because our stack reads neither value.

### Robot side

1. `real_driver` reads the values from `LowState` in `ControlLoop::raw_state`
   (`src/quadruped_real/src/control_loop.cpp`), and puts them in `Snapshot` (`messages.hpp`):
   - `bms_state().soc()` (battery %), `power_v()`, `power_a()`.
   - `motor_state()[i].temperature()` for the 12 motors, in ROS joint order (`kSdkToRos`).
2. `DriverNode::publish_robot_state` (`driver_node.cpp`) adds these to `/robot_state`:
   `"battery": {"soc": 78, "v": 28.1, "a": 3.2}`, `"motor_temp_c": [12 values]`,
   `"hottest": {"joint": "FR_calf_joint", "c": 61}`.
3. `mcp_server.get_state` adds `battery_pct` and `hottest`, so the agent and Claude Code see them.
4. Protection, in `config.yaml` `safety:`:
   - `motor_temp_warn_c`: an event (log line and app banner) when a motor gets to this temperature.
   - `motor_temp_lie_c`: the safety policy (`Lie2`) lies the robot down, as a trip with the reason
     "FR_calf motor 85 C". The robot then lies down under control. It does not fall when the firmware cuts out.
   - `battery_low_pct`: an event, and the same lie-down at a lower value.
   - The thresholds are not known yet. First, record the temperatures during usual use (step 5). Then set
     values below the highest that the robot reached without a cut-out.
5. Measure the heat:
   - The temperature rise per minute while the robot stands under `main`, sits and lies.
   - If a stand heats the knees quickly, the agent must not leave the robot standing while it waits. Possible
     fix: `mcp_server` makes the robot `sit` after `idle_sit_s` with no tool call (config), and `walk` again
     before the next motion.

### App

- Top bar: battery % (red under 20%) and the hottest motor in °C (amber at `warn`, red at `lie`).
- System tab, new card "Robot body": battery %, V and A, and the 12 motor temperatures as a grid of 4 legs by
  3 joints, with colours.
- A banner on the Drive tab at the warn temperature or a low battery, like the numb banner.
- The values come from `/robot_state`, which the app already reads. No new topic or DDS type.

### Test

- `Android/tools/ros_side.py` sends a fake `/robot_state` with the new fields: check the top bar, the card and
  the banner.
- On the robot: the battery % agrees with the Go2's battery LEDs (and the Unitree app). Touch a hot motor: its
  value is the highest.

## Upgrade 2: LLM chat tab in the app

### How it works for the operator

1. `[A]` starts the agent in app mode. It waits for a task from the app and shows "waiting for a task from the
   app" in the terminal.
2. The app's new tab "Chat" shows the agent's status: no agent, waiting, thinking, calling a tool.
3. The first message that the operator types starts a chat, with that message as the task. Later messages are
   answers to the agent (they replace the terminal's `you>` prompt).
4. "New chat": stops the current chat (the `stop` tool, so the robot stops), clears the screen, and the next
   message starts a new task. The model stays loaded in `[A]`, so the new chat starts with no wait.
5. "Stop": stops the current chat and the robot, and does not start a new one.
6. The chat shows the agent's text, each tool call on one short line, the results, and the photos:
   the photo from `look`, and the photo from `find` with the box drawn on it.
7. Ctrl+C in the terminal ends `[A]`.

### Topics

All are `std_msgs/String` (JSON) or `sensor_msgs/CompressedImage`. The app already has both types.

| Topic | From → to | Content |
|---|---|---|
| `/llm/chat/in` | app → agent | `{"chat": id, "text": "..."}` or `{"chat": id, "cmd": "new" \| "stop"}` |
| `/llm/chat/out` | agent → app | `{"chat", "seq", "kind": "status" \| "said" \| "call" \| "result" \| "error", "text", "tool"}` |
| `/llm/chat/image/compressed` | agent → app | JPEG, 640 x 360 (about 40 kB). `header.frame_id` = `"<chat>:<seq>"` of its event. |

`/llm/chat/out` is reliable and transient-local, with a depth of 200. Thus the app gets the current chat again
after a reconnect.

### The photo and box from `find`

- `find` returns its photo with the box drawn on it (`mcp_server` already has OpenCV), as an MCP image content
  with `"annotations": {"audience": ["user"]}`. This is the MCP standard mark for content that is for the person
  and not for the model.
- `llm_host` sends images with `audience: user` to the app only. It never gives them to the model: each image
  costs the vision encoder some seconds and fills the 8192-token context, and the model does not need the
  photo, because it has x and y.
- The photo from `look` goes to the model, as now, and also to the app.

### Robot side changes

- `Tools/llm_host.py`:
  - New `--app` mode. Only this mode imports `rclpy` (the robot's Docker image has it). The terminal mode stays
    standard library only.
  - The agent loop becomes a function for one chat. Its operator input comes from a queue: the terminal's
    `input()`, or `/llm/chat/in` in app mode.
  - Each printed line also goes to `/llm/chat/out` in app mode.
- `launcher.py` `[A]`: no task prompt; it starts `llm_host.py --app`.
- `mcp_server.find`: the photo with the box, with `audience: user`.

### App changes

- `ros_link`: no new type. Check that a `String` subscription can be reliable and transient-local with a
  depth of 200.
- New tab `Tab.CHAT` and a new file `ui/ChatTab.kt`:
  - The list of messages. Tool calls are short lines that open on a tap. Photos show in the list, and a tap
    makes them full size.
  - A text field with Send, and the buttons "New chat" and "Stop".
  - The status line at the top.
  - Offline: disabled, like the other tabs.

### Test

- `Android/tools/llm_side.py` (new, like `ros_side.py`) plays the agent: it answers on `/llm/chat/out` and sends
  a test photo. Check the tab with no robot.
- On the robot: `[A]` in app mode, then from the phone "find the shoe and pee on it". Then "New chat" with
  another task.

## Decisions

1. **The `[A]` task prompt:** removed. `[A]` always starts in app mode, and the task comes from the app.
   `llm_host.py` keeps its `task` argument for runs from a PC terminal.
2. **Unload Qwen:** the model stays loaded while `[A]` waits in app mode, so a new chat starts with no wait.
   `[A]` unloads it when it ends. The cost is about 3.9 GB of RAM while `[A]` runs.
3. **Temperature and battery thresholds:** high values for now, so that they do not trip in usual use. Set the
   real values after the measurements in upgrade 1, step 5.
