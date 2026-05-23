import os
import sys
import time
import argparse
import importlib
from datetime import datetime

import rclpy
from rclpy.node import Node
from mcap_ros2.writer import Writer as McapWriter
from mcap_ros2.reader import read_ros2_messages

import yaml

def load_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "..", "Configs", "config.yaml")
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}

def get_msg_class(type_str: str):
    """Dynamically imports a ROS 2 message class from its type string."""
    parts = type_str.split('/')
    if len(parts) == 3:
        pkg_name = f"{parts[0]}.{parts[1]}"
        class_name = parts[2]
    elif len(parts) == 2:
        pkg_name = f"{parts[0]}.msg"
        class_name = parts[1]
    else:
        raise ValueError(f"Invalid message type string: {type_str}")
    
    module = importlib.import_module(pkg_name)
    return getattr(module, class_name)

PRIMITIVES = {
    "bool", "byte", "char", "float32", "float64", 
    "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
    "string", "wstring"
}

def map_field_type(f_type: str) -> str:
    if f_type.startswith("sequence<") and f_type.endswith(">"):
        inner = f_type[len("sequence<"):-1]
        return f"{map_field_type(inner)}[]"
    suffix = ""
    if "[" in f_type and f_type.endswith("]"):
        parts = f_type.split("[")
        f_type = parts[0]
        suffix = "[" + parts[1]
    mapping = {
        "double": "float64",
        "float": "float32",
        "boolean": "bool",
        "octet": "byte",
        "char": "char",
        "string": "string"
    }
    base = mapping.get(f_type, f_type)
    return base + suffix

def get_msg_def_recursive(msg_class, seen_types=None):
    if seen_types is None:
        seen_types = set()
        
    inst = msg_class()
    fields_dict = inst.get_fields_and_field_types()
    
    main_lines = []
    nested_defs = []
    
    for field, f_type in fields_dict.items():
        mapped_type = map_field_type(f_type)
        main_lines.append(f"{mapped_type} {field}")
        
        # Check if nested
        base_type = mapped_type.split('[')[0]
        if base_type not in PRIMITIVES and "/" in base_type:
            if base_type not in seen_types:
                seen_types.add(base_type)
                nested_class = get_msg_class(base_type)
                nested_main, nested_children = get_msg_def_recursive(nested_class, seen_types)
                nested_defs.append((base_type, nested_main))
                nested_defs.extend(nested_children)
                
    return "\n".join(main_lines), nested_defs

def get_msg_type_and_def(msg_class):
    """Safely retrieves ROS 2 type and full schema definition recursively."""
    # Type extraction
    if hasattr(msg_class, '_type'):
        msg_type = msg_class._type
    else:
        pkg = msg_class.__module__.split('.')[0]
        msg_type = f"{pkg}/msg/{msg_class.__name__}"

    # Definition text extraction
    if hasattr(msg_class, '_full_text'):
        msg_def = msg_class._full_text
    elif hasattr(msg_class.__class__, '_full_text'):
        msg_def = msg_class.__class__._full_text
    else:
        try:
            main_def, nested = get_msg_def_recursive(msg_class)
            full_def = main_def
            for nested_type, nested_text in nested:
                full_def += f"\n\n================================================================================\nMSG: {nested_type}\n{nested_text}"
            msg_def = full_def
        except Exception:
            msg_def = "# Fallback empty schema"

    return msg_type, msg_def


class McapRecorder(Node):
    def __init__(self, filepath: str):
        super().__init__("mcap_recorder")
        self.filepath = filepath
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        self.file = open(self.filepath, "wb")
        self.writer = McapWriter(self.file)
        self.registered_types = {}
        self.msg_count = 0
        self.start_time = time.time()
        
        self.subscriptions_dict = {}
        
        # Run an immediate manual scan to capture topics present at launch
        self._scan_topics()

        # Start a fast dynamic discovery timer to rapidly catch any delayed topics
        self.discovery_timer = self.create_timer(0.5, self._scan_topics)

        self.get_logger().info(f"MCAP Recorder Node started -> Recording to {self.filepath} (Auto-discovering all topics)")

    def _subscribe_topic(self, topic: str, msg_type: str):
        if topic in self.subscriptions_dict:
            return
        try:
            msg_class = get_msg_class(msg_type)
            cb = self._make_callback(topic, msg_class)
            self.subscriptions_dict[topic] = self.create_subscription(
                msg_class, topic, cb, 10
            )
        except Exception:
            pass

    def _scan_topics(self):
        for topic, types in self.get_topic_names_and_types():
            if not types: continue
            self._subscribe_topic(topic, types[0])

    def _make_callback(self, topic: str, msg_class):
        def callback(msg):
            msg_type, msg_def = get_msg_type_and_def(msg_class)
            if msg_type not in self.registered_types:
                schema = self.writer.register_msgdef(msg_type, msg_def)
                self.registered_types[msg_type] = schema
                
            schema = self.registered_types[msg_type]
            from rosidl_runtime_py.convert import message_to_ordereddict
            self.writer.write_message(
                topic=topic,
                schema=schema,
                message=message_to_ordereddict(msg),
                log_time=time.time_ns(),
                publish_time=time.time_ns()
            )
            self.file.flush()
            self.msg_count += 1
            elapsed = time.time() - self.start_time
            # print(f"\r[Recorder] Active | Time: {elapsed:7.2f}s | Messages written: {self.msg_count:<6}", end="", flush=True)
        return callback

    def close(self):
        print() # New line after the status reports
        self.writer.finish()
        self.file.close()
        self.get_logger().info(f"MCAP file successfully closed: {self.filepath}")


def dynamic_msg_to_dict(obj):
    if hasattr(obj, "__slots__"):
        d = {}
        for k in obj.__slots__:
            d[k] = dynamic_msg_to_dict(getattr(obj, k))
        return d
    elif hasattr(obj, "__dict__"):
        d = {}
        for k, v in vars(obj).items():
            if k.startswith("_"):
                continue
            d[k] = dynamic_msg_to_dict(v)
        return d
    elif isinstance(obj, list):
        return [dynamic_msg_to_dict(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(dynamic_msg_to_dict(x) for x in obj)
    else:
        return obj


class PlaybackFrame:
    def __init__(self, start_time_ns):
        self.start_time_ns = start_time_ns
        self.messages = []

class McapReplayer(Node):
    def __init__(self, filepath: str):
        super().__init__("mcap_replayer")
        self.filepath = filepath
        self.publishers_dict = {}

    def replay(self):
        import sys, tty, termios, select
        self.get_logger().info(f"Loading playback messages into memory: {self.filepath}")
        try:
            messages = list(read_ros2_messages(self.filepath))
        except Exception as e:
            self.get_logger().error(f"Failed to read MCAP file: {e}")
            return
            
        if not messages:
            self.get_logger().warn("No messages found in MCAP file.")
            return
            
        self.get_logger().info(f"Loaded {len(messages)} messages successfully.")

        config = load_config()
        cluster_window_ms = config.get("logging", {}).get("cluster_window_ms", 10)
        cluster_window_ns = cluster_window_ms * 1_000_000
        
        self.get_logger().info(f"Clustering messages using a {cluster_window_ms}ms gap threshold...")
        frames = []
        current_frame = None
        last_msg_time = 0
        for msg in messages:
            if current_frame is None or (msg.log_time_ns - last_msg_time) > cluster_window_ns:
                current_frame = PlaybackFrame(msg.log_time_ns)
                frames.append(current_frame)
            current_frame.messages.append(msg)
            last_msg_time = msg.log_time_ns
            
        self.get_logger().info(f"Clustered {len(messages)} messages into {len(frames)} physical frames.")

        # 0. Pre-scan all messages to create dynamic publishers upfront (prevents discovery dropped messages)
        self.get_logger().info("Scanning topics and establishing ROS 2 network connections...")
        for msg in messages:
            topic = msg.channel.topic
            msg_type = msg.schema.name if msg.schema else ""
            if topic not in self.publishers_dict:
                try:
                    msg_class = get_msg_class(msg_type)
                    self.publishers_dict[topic] = (
                        self.create_publisher(msg_class, topic, 10),
                        msg_class
                    )
                except Exception as e:
                    pass

        self.get_logger().info("Waiting 1.5s for Twin / DDS discovery to connect fully...")
        import time
        time.sleep(1.5)

        print("\n" + "=" * 80)
        print(" INTERACTIVE PLAYBACK CONTROLS:")
        print("   [Spacebar] : Toggle Play / Pause")
        print("   [Right Arrow] or [d] : Step 1 Frame Forward (when paused)")
        print("   [Left Arrow]  or [a] : Step 1 Frame Backward (when paused)")
        print("   [Up/Down Arrow]      : Increase/Decrease Playback Speed")
        print("   [r] / [R]  : Restart playback from the beginning")
        print("   [q]          : Quit playback")
        print("=" * 80 + "\n")

        # Set terminal to cbreak mode to capture keys instantly without blocking or dropping them
        fd = sys.stdin.fileno()
        old_settings = None
        try:
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception:
            pass

        key_buffer = b""
        import os

        def get_key_nonblock():
            nonlocal key_buffer
            try:
                rlist, _, _ = select.select([fd], [], [], 0.0)
                if rlist:
                    # Use raw os.read to completely bypass Python's text io buffer!
                    key_buffer += os.read(fd, 1024)
                    
                if not key_buffer:
                    return None
                    
                if key_buffer.startswith(b'\x1b['):
                    if len(key_buffer) >= 3:
                        ch3 = chr(key_buffer[2])
                        key_buffer = key_buffer[3:] # consume
                        if ch3 == 'A': return 'up'
                        elif ch3 == 'B': return 'down'
                        elif ch3 == 'C': return 'right'
                        elif ch3 == 'D': return 'left'
                        return '\x1b'
                    else:
                        # Wait for next OS flush if sequence is incomplete
                        return None
                else:
                    # Normal character
                    ch = chr(key_buffer[0])
                    key_buffer = key_buffer[1:]
                    return ch
            except Exception:
                pass
            return None

        def draw_progress(idx, total, paused, speed):
            bar_len = 30
            filled = int(round(bar_len * (idx + 1) / float(total)))
            bar = '█' * filled + '-' * (bar_len - filled)
            state = "PAUSED " if paused else "PLAYING"
            print(f"\r[Replayer] [{state}] |{bar}| Frame: {idx + 1:>5}/{total:<5} | Speed: {speed:.2f}x ", end="", flush=True)

        paused = False
        idx = 0
        t0 = None
        start_real_time = time.time()
        playback_speed = 1.0
        
        try:
            while rclpy.ok() and idx < len(frames):
                # 0. Check for keyboard input (throttled) ONLY if playing
                if not paused and (idx % max(1, int(20 * playback_speed)) == 0):
                    key = get_key_nonblock()
                    if key == ' ':
                        paused = not paused
                        if paused:
                            print("\n\n[Replayer] PAUSED. Press [Space] to resume, [Right/Left Arrow] to step, [R] to restart.")
                        else:
                            t0 = frames[idx].start_time_ns
                            start_real_time = time.time()
                            print("\n\n[Replayer] RESUMED (Playing)...")
                    elif key == 'up':
                        playback_speed = min(playback_speed * 2.0, 16.0)
                        t0 = frames[idx].start_time_ns
                        start_real_time = time.time()
                    elif key == 'down':
                        playback_speed = max(playback_speed / 2.0, 0.125)
                        t0 = frames[idx].start_time_ns
                        start_real_time = time.time()
                    elif key == 'q':
                        print("\n\n[Replayer] Exiting.")
                        break
                    elif key in ('r', 'R'):
                        idx = 0
                        t0 = None
                        print("\n\n[Replayer] RESTARTED from beginning.")
                        continue

                # 1. Non-blocking sleep delay if playing
                if not paused:
                    frame = frames[idx]
                    log_time_ns = frame.start_time_ns
                    if t0 is None:
                        t0 = log_time_ns
                        start_real_time = time.time()
                        
                    dt = (log_time_ns - t0) / 1e9
                    dt /= playback_speed
                    
                    while rclpy.ok() and not paused:
                        elapsed = time.time() - start_real_time
                        sleep_needed = dt - elapsed
                        if sleep_needed <= 0:
                            break
                            
                        key = get_key_nonblock()
                        if key == ' ':
                            paused = True
                            print("\n\n[Replayer] PAUSED. Press [Space] to resume, [Right/Left Arrow] to step, [R] to restart.")
                            break
                        elif key == 'up':
                            playback_speed = min(playback_speed * 2.0, 16.0)
                            t0 = frames[idx].start_time_ns
                            start_real_time = time.time()
                            break
                        elif key == 'down':
                            playback_speed = max(playback_speed / 2.0, 0.125)
                            t0 = frames[idx].start_time_ns
                            start_real_time = time.time()
                            break
                        elif key == 'q':
                            print("\n\n[Replayer] Exiting.")
                            idx = len(frames) # Trigger exit
                            break
                        time.sleep(min(sleep_needed, 0.005))
                        
                # 2. Wait-for-keystroke loop if paused
                if paused:
                    while rclpy.ok() and paused:
                        key = get_key_nonblock()
                        if key == ' ':
                            paused = False
                            t0 = frames[idx].start_time_ns
                            start_real_time = time.time()
                            print("\n\n[Replayer] RESUMED (Playing)...")
                            break
                        elif key in ('right', 'd'):
                            idx = min(idx + 1, len(frames) - 1)
                            break
                        elif key in ('left', 'a'):
                            idx = max(idx - 1, 0)
                            break
                        elif key == 'up':
                            playback_speed = min(playback_speed * 2.0, 16.0)
                            break
                        elif key == 'down':
                            playback_speed = max(playback_speed / 2.0, 0.125)
                            break
                        elif key in ('r', 'R'):
                            idx = 0
                            t0 = None
                            print("\n\n[Replayer] RESTARTED from beginning.")
                            break
                        elif key == 'q':
                            print("\n\n[Replayer] Exiting.")
                            idx = len(frames) # Trigger exit
                            break
                        time.sleep(0.01)

                if idx >= len(frames):
                    break

                # 3. Publish the current frame's messages
                frame = frames[idx]
                for msg in frame.messages:
                    topic = msg.channel.topic
                    if topic in self.publishers_dict:
                        pub, msg_class = self.publishers_dict[topic]
                        try:
                            from rosidl_runtime_py import set_message_fields
                            native_msg = msg_class()
                            set_message_fields(native_msg, dynamic_msg_to_dict(msg.ros_msg))
                            pub.publish(native_msg)
                        except Exception as e:
                            pass

                draw_progress(idx, len(frames), paused, playback_speed)
                
                # Advance frame if not paused
                if not paused:
                    idx += 1
                    
            print("\n[Replayer] Replay completed successfully.")
        except KeyboardInterrupt:
            print("\n[Replayer] Playback interrupted by user.")
        except Exception as e:
            self.get_logger().error(f"Error during playback: {e}")
        finally:
            if old_settings:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(description="MCAP Recorder & Replay Tool")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--record", type=str, help="Path to write the recorded MCAP file")
    group.add_argument("--replay", type=str, help="Path to read and replay the MCAP file")
    args = parser.parse_args()

    rclpy.init()
    if args.record:
        node = McapRecorder(args.record)
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            node.close()
            node.destroy_node()
    else:
        node = McapReplayer(args.replay)
        node.replay()
        node.destroy_node()
        
    if rclpy.ok():
        rclpy.shutdown()

if __name__ == "__main__":
    main()
