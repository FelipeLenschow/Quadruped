"""LLM host: a local model on Ollama drives the robot through the MCP server's tools.

Lists the server's tools (mcp_server in quadruped_operator), hands them to the model, runs each tool call
it makes and gives it the result, until it answers without one. Then the operator can answer it. On exit the
model is unloaded from Ollama to free its memory (--keep_loaded keeps it; loading again takes ~5 s on the robot).
Standard library only, so it runs on any machine that reaches both Ollama and the robot.

In the terminal (Enter: it keeps going until --minutes are up; --auto never waits; Ctrl+C sends stop):

    python3 Tools/llm_host.py                                  # the "having fun" loop, 10 minutes
    python3 Tools/llm_host.py "walk to the nearest cone and sniff it" --minutes 0
    python3 Tools/llm_host.py --mcp http://<robot>:8765/mcp --model qwen3.5:9b

With --app (launcher [A]), the operator is the phone app's Chat tab: it waits for a task from the app, and
serves the chat over HTTP on --port (8766), open to anyone on the network, like the MCP server without a token:
    GET  /chat?after=<seq>&wait=<s>&chat=<id>  the chat's events after seq: {"chat", "status", "events": [{"seq",
                                     "kind", "text", "tool", "image"}]}; waits up to wait s (at most 25) for a new
                                     one or a new status, not when chat (the one the app shows) is another
    GET  /image/<n>.jpg              a photo an event names (look's, find_something's with its box,
                                     go_to_place's with its point)
    POST /chat {"text"}              the task when no chat runs, else an answer ("" = keep going)
    POST /chat/stop                  stops the chat and the robot
    POST /chat/new                   stops the chat and clears it; the next text is a new task
Ctrl+C ends it.

Each run writes logs/agent/<date_time>/log.md (--log_dir; "" for none): every task, message, tool call and full
result with its time, and the photos beside it as files that the log links, so a Markdown preview shows them.

QUADRUPED_MCP_TOKEN (or --token) is sent as the bearer token when the MCP server requires one.
"""

import argparse
import base64
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FUN = """You are the robot dog. Have fun with the traffic cones in this room, on your own:
1. list_objects to see the cones you know about. Pick one whose "done" list is empty, or the one done longest ago.
2. visit it with trick sniff. Sometimes (about half the time) then visit it again with pee_left or pee_right.
3. If no cone is left to visit, look for more. The camera sees only ahead and the map shows walls, not cones:
   call frontiers and go_to the nearest one; if none is left, go_to the middle of a part of the room you have not
   stood in yet. There, move turn_left 90 four times, with list_objects after each turn.
Call skill walk first if you are not walking. If a call fails, read why and try something else.
Keep each message short: one line on what you do next, then the tool call."""

# FUN without object detection (no list_objects): the model picks things in its photos and find_something maps them.
FUN_LOOK = """You are the robot dog. Have fun with the things on the floor of this room (bins, chairs, boxes, cones), on
your own:
1. look. Pick a thing on the floor in the photo that you have not visited yet.
2. find_something by its name, then visit the x, y that it gives with trick sniff. Sometimes (about
   half the time) then visit the same x, y with pee_left or pee_right.
3. Then move turn_left 90 and look again. After four turns with nothing new, call frontiers and go_to the nearest.
Call skill walk first if you are not walking. If a call fails, read why and try something else.
Keep each message short: one line on what you do next, then the tool call."""

# Without object detection or find_something: explore and describe what the camera sees.
EXPLORE = """You are the robot dog. Explore this room on your own:
1. call frontiers and go_to the nearest one; if none is left, go_to the middle of a part of the room you have not
   stood in yet.
2. There, move turn_left 90 four times, with look after each turn, and say in one line what you see.
Call skill walk first if you are not walking. If a call fails, read why and try something else.
Keep each message short: one line on what you do next, then the tool call."""

KEEP_GOING = "Keep going: pick the next thing to do and call a tool."
OPERATOR = ("The operator reads your messages and may answer when you reply without a tool call. Ask only when you "
            "cannot go on without them. Keep each message short: one line on what you do next, then the tool call.")


class Mcp:
    def __init__(self, url, token):
        self.url, self.token, self.session, self.next_id = url, token, None, 1
        self.meta = {}  # the server's initialize _meta: mcp_server's vision model

    def _post(self, msg, on_note=None):
        """The reply; with an event stream (mcp_server's tool calls), each notification before it goes to
        on_note(its data) as it comes."""
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        req = urllib.request.Request(self.url, json.dumps(msg).encode(), headers)
        with urllib.request.urlopen(req, timeout=600) as r:
            self.session = r.headers.get("Mcp-Session-Id") or self.session
            if "event-stream" not in r.headers.get("Content-Type", ""):
                body = r.read().decode()
                return json.loads(body) if body.strip() else None
            reply = None
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:"):
                    continue
                body = json.loads(line[5:])
                if "id" in body:
                    reply = body
                elif on_note and body.get("method") == "notifications/message":
                    on_note(body["params"].get("data"))
            return reply

    def request(self, method, params=None, on_note=None):
        msg = {"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params or {}}
        self.next_id += 1
        reply = self._post(msg, on_note)
        if "error" in reply:
            raise RuntimeError(f"{method}: {reply['error'].get('message')}")
        return reply["result"]

    def start(self):
        init = self.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                           "clientInfo": {"name": "llm_host", "version": "0.1.0"}})
        self.meta = init.get("_meta") or {}
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return init.get("instructions", ""), self.request("tools/list")["tools"]

    def call(self, name, args, on_note=None):
        return self.request("tools/call", {"name": name, "arguments": args}, on_note)


class OllamaDown(RuntimeError):
    pass


def chat(url, model, messages, tools, think, ctx):
    body = {"model": model, "messages": messages, "tools": tools, "stream": False, "think": think,
            "options": {"num_ctx": ctx}}
    req = urllib.request.Request(f"{url}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["message"]
    except (urllib.error.URLError, OSError) as e:
        raise OllamaDown(f"Ollama at {url} did not answer ({e}); on the robot: sudo systemctl start ollama")


def unload(url, model):
    """Ollama keeps the model (KEEP_ALIVE=-1 on the robot) and its memory grows with use: 1.3 GB free on the
    robot fell to 0.4 GB and 0.9 GB went to swap in one run."""
    body = {"model": model, "keep_alive": 0}
    req = urllib.request.Request(f"{url}/api/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()


def trim(messages, keep):
    """System prompt, the task, and the last `keep` messages; photos only on the newest one."""
    if len(messages) > keep + 2:
        tail = messages[-keep:]
        while tail and tail[0]["role"] == "tool":
            tail = tail[1:]
        messages[:] = messages[:2] + tail
    with_images = [m for m in messages if m.get("images")]
    for m in with_images[:-1]:
        m.pop("images")
        m["content"] += " (photo dropped)"


def for_model(content):
    """An image content is for the model unless its annotations say it is for the user only (find_something's photo with
    its box: the model has x and y, and a photo costs the vision encoder seconds and fills the context)."""
    audience = (content.get("annotations") or {}).get("audience")
    return not audience or "assistant" in audience


class RunLog:
    """The run as Markdown, with its photos (the app's and the model's) as files beside it: so a run can be read
    afterwards, not copied from the terminal, where results are cut at 300 characters and photos are missing."""

    def __init__(self, root):
        self.dir = os.path.join(root, time.strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(self.dir, exist_ok=True)
        self.file = open(os.path.join(self.dir, "log.md"), "a", buffering=1)
        self.lock = threading.Lock()
        self.photos = 0

    def header(self, args, meta, tools):
        vision = meta.get("vision_model")
        vision = f"{vision}, context {meta.get('vision_ctx')}" if vision else "not given by the server"
        think = "on" if args.think else "off"
        with self.lock:
            self.file.write(f"# Agent run {os.path.basename(self.dir)}\n\n"
                            f"- model: {args.model}, context {args.ctx}, think {think}, keep {args.keep} messages\n"
                            f"- vision model (find_something, go_to_place): {vision}\n"
                            f"- Ollama {args.ollama}, MCP {args.mcp}\n- tools: {', '.join(tools)}\n")

    def write(self, kind, text, tool="", images=()):
        with self.lock:
            stamp = time.strftime("%H:%M:%S")
            text = str(text).replace("\n", "\n  ")
            if kind == "you":
                line = f"\n## {stamp} you: {text}\n"
            elif kind == "call":
                line = f"- `{stamp}` → **{text}**"
            elif kind == "said":
                line = f"- `{stamp}` agent: {text}"
            else:
                line = f"- `{stamp}` {kind}{' ' + tool if tool else ''}: {text}"
            for data in images:
                self.photos += 1
                raw = base64.b64decode(data)
                ext = "png" if raw[:4] == b"\x89PNG" else "jpg"  # not in the f-string: Python 3.8 on the robot
                name = f"{self.photos:03d}_{tool or kind}.{ext}"
                with open(os.path.join(self.dir, name), "wb") as f:
                    f.write(raw)
                line += f"\n\n  ![{name}]({name})\n"
            self.file.write(line + "\n")


LOG = None


def log(kind, text, tool="", images=()):
    if LOG is not None:
        try:
            LOG.write(kind, text, tool, images)
        except OSError as e:
            print(f"log failed: {e}")


class Terminal:
    """The operator at this terminal."""

    def __init__(self, ask):
        self.ask = ask

    def show(self, kind, text, tool="", images=()):
        log(kind, text, tool, images)
        if kind == "said":
            print(f"\n{text}")
        elif kind == "call":
            print(f"  -> {text}", flush=True)
        elif kind in ("result", "error", "note"):
            print(f"     {'ERROR ' if kind == 'error' else ''}{text[:300]}")
        else:
            print(text)

    def status(self, text):
        pass

    def answer(self):
        """The operator's answer, "" to keep going, None to end."""
        if not self.ask:
            return ""
        answer = input("\nyou (Enter: keep going, q: quit)> ").strip()
        if answer.lower() != "q":
            log("you", answer or "(keep going)")
        return None if answer.lower() == "q" else answer

    def interjection(self):
        return None

    def stopped(self):
        return False


class AppChat:
    """The operator in the app's Chat tab, over HTTP (see the module doc). Also prints what it sends."""

    MAX_IMAGES = 50

    def __init__(self, stop_robot):
        self.stop_robot = stop_robot
        self.cond = threading.Condition()
        # Not 1: the app keeps a chat while the id stays, so a restarted agent's chat 1 showed the old one, and
        # hid the new events until their numbers passed the old chat's last.
        self.chat = int(time.time())
        self.events, self.images = [], {}
        self.seq = 0
        # Photo numbers from the clock in ms, as the chat id: the app keeps its photos by path, so a restarted
        # agent's /image/1.jpg showed the old run's photo. A run never takes a photo a ms, so they never meet.
        self.image_n = int(time.time() * 1000)
        self.state = "waiting for a task"
        self.running = self.stop_flag = False
        self.inbox = []
        self.terminal = Terminal(False)

    # --- the agent's side ---
    def show(self, kind, text, tool="", images=()):
        self.terminal.show(kind, text, tool, images)
        with self.cond:
            image = None
            for data in images:
                self.image_n += 1
                self.images[self.image_n] = data
                image = f"/image/{self.image_n}.jpg"
                self.images.pop(self.image_n - self.MAX_IMAGES, None)
            self._add(kind, text, tool, image)

    def status(self, text):
        with self.cond:
            self.state = text
            self.cond.notify_all()

    def next_task(self):
        with self.cond:
            self.state = "waiting for a task"
            self.running = self.stop_flag = False
            self.cond.notify_all()
            while not self.inbox:
                self.cond.wait()
            task = self.inbox.pop(0)
            self.running = True
            return task

    def answer(self):
        with self.cond:
            self.state = "waiting for you"
            self.cond.notify_all()
            while not self.inbox and not self.stop_flag:
                self.cond.wait()
            return None if self.stop_flag else self.inbox.pop(0)

    def interjection(self):
        """A message the operator sent while the agent was busy, given to it before its next step."""
        with self.cond:
            return self.inbox.pop(0) if self.inbox and not self.stop_flag else None

    def stopped(self):
        return self.stop_flag

    # --- the app's side ---
    def _add(self, kind, text, tool="", image=None):
        self.seq += 1
        e = {"seq": self.seq, "kind": kind, "text": text}
        if tool:
            e["tool"] = tool
        if image:
            e["image"] = image
        self.events.append(e)
        self.cond.notify_all()

    def get(self, after, wait, chat=None):
        end = time.monotonic() + min(max(wait, 0.0), 25.0)
        with self.cond:
            if chat is not None and chat != self.chat:
                end = 0.0  # the app shows another chat (or none yet): the new id at once
            # A new status also answers at once, not only a new event; so does an `after` this run never gave: an
            # app still on the chat of an agent that was restarted, which would wait the whole poll to see the new id.
            state = self.state
            while not any(e["seq"] > after for e in self.events) and self.state == state and after <= self.seq \
                    and time.monotonic() < end:
                self.cond.wait(end - time.monotonic())
            return {"chat": self.chat, "status": self.state,
                    "events": [e for e in self.events if e["seq"] > after]}

    def post(self, text):
        log("you", text or "(keep going)")
        with self.cond:
            self._add("you", text)
            self.inbox.append(text)
            self.cond.notify_all()

    def stop(self, clear=False):
        with self.cond:
            was_running = self.running
            if was_running:
                self.stop_flag = True
                self.state = "stopping"
            self.inbox.clear()
            if clear:
                log("info", "New chat")
                self.chat += 1
                self.events = []
                self.state = "waiting for a task"
            self.cond.notify_all()
        if was_running:
            text = self.stop_robot()
            log("info", f"stopped by the operator ({text})")
            if not clear:
                with self.cond:
                    self._add("info", f"stopped by the operator ({text})")


def serve(chat, port):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body=b"", mime="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path, _, query = self.path.partition("?")
            q = dict(kv.split("=", 1) for kv in query.split("&") if "=" in kv)
            m = re.fullmatch(r"/image/(\d+)\.jpg", path)
            if path == "/chat":
                try:
                    after, wait = int(q.get("after", 0)), float(q.get("wait", 0))
                    shown = int(q["chat"]) if "chat" in q else None
                except ValueError:
                    return self._send(400, b'{"error": "after, wait and chat are numbers"}')
                self._send(200, json.dumps(chat.get(after, wait, shown)).encode())
            elif m and int(m.group(1)) in chat.images:
                self._send(200, base64.b64decode(chat.images[int(m.group(1))]), "image/jpeg")
            else:
                self._send(404, b'{"error": "not found"}')

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0)
            if self.path == "/chat":
                try:
                    chat.post(str(json.loads(body or b"{}").get("text", "")).strip())
                except ValueError:
                    return self._send(400, b'{"error": "the body must be JSON"}')
            elif self.path in ("/chat/stop", "/chat/new"):
                threading.Thread(target=chat.stop, args=(self.path == "/chat/new",), daemon=True).start()
            else:
                return self._send(404, b'{"error": "not found"}')
            self._send(200, json.dumps({"chat": chat.chat}).encode())

    class Server(ThreadingHTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):
            # The phone drops a poll when the app goes to the background or the Wi-Fi changes: not an error.
            if not isinstance(sys.exc_info()[1], ConnectionError):
                super().handle_error(request, client_address)

    server = Server(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def run_chat(io, mcp, args, tools, system, task):
    """One task, until the model answers and the operator ends it, the time is up, or the operator stops it.
    Returns the number of tool calls."""
    # The task alone. A pee note after it (find_something, then visit with the trick) made the 4B refuse every other
    # task ("go under the table" 10 times in 10), and the 4B finds the thing without it (19 in 20). The 2B needs it:
    # without it, it called the trick at once for "pee on the chair" 8 times in 10 (2026-11-03).
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    end = time.monotonic() + args.minutes * 60
    calls = 0
    while calls < args.max_calls and not io.stopped():
        more = io.interjection()
        if more is not None:
            messages.append({"role": "user", "content": more or KEEP_GOING})
        trim(messages, args.keep)
        io.status("thinking")
        t = time.monotonic()
        reply = chat(args.ollama, args.model, messages, tools, args.think, args.ctx)
        if io.stopped():
            break
        said = (reply.get("content") or "").strip()
        if said:
            io.show("said", f"[{time.monotonic() - t:.1f}s] {said}")
        todo = reply.get("tool_calls") or []
        messages.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": todo})
        if not todo:
            answer = io.answer()
            if answer is None or (not answer and args.minutes > 0 and time.monotonic() > end):
                break
            messages.append({"role": "user", "content": answer or KEEP_GOING})
            continue
        for tc in todo:
            name, targs = tc["function"]["name"], tc["function"].get("arguments") or {}
            if isinstance(targs, str):
                targs = json.loads(targs or "{}")
            io.show("call", f"{name}({json.dumps(targs)[1:-1]})", name)
            io.status(f"calling {name}")
            def note(data, name=name):
                """Before the result: go_to_place's photo with its points, before Nav2 has the goal."""
                if isinstance(data, dict):
                    image = data.get("image") or {}
                    io.show("note", data.get("text", ""), name, [image["data"]] if image.get("data") else [])

            try:
                res = mcp.call(name, targs, note)
            except (urllib.error.URLError, RuntimeError) as e:
                res = {"content": [{"type": "text", "text": f"call failed: {e}"}], "isError": True}
            calls += 1
            content = res.get("content", [])
            texts = [c["text"] for c in content if c.get("type") == "text"]
            images = [c for c in content if c.get("type") == "image"]
            out = "\n".join(texts) or ("photo attached below" if images else "(no output)")
            io.show("error" if res.get("isError") else "result", out, name, [c["data"] for c in images])
            messages.append({"role": "tool", "tool_name": name, "content": out})
            seen = [c["data"] for c in images if for_model(c)]
            if seen:
                messages.append({"role": "user", "content": f"The photo from {name}.", "images": seen})
            if io.stopped():
                break
        if args.minutes > 0 and time.monotonic() > end:
            io.show("info", f"{args.minutes:g} minutes are up.")
            break
    return calls


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("task", nargs="?", default="",
                    help="what to do (default: the having-fun loop, or exploring without object detection)")
    ap.add_argument("--app", action="store_true", help="take tasks and answers from the app's Chat tab (HTTP)")
    ap.add_argument("--port", type=int, default=8766, help="--app: the chat's HTTP port")
    ap.add_argument("--mcp", default="http://127.0.0.1:8765/mcp")
    ap.add_argument("--token", default=os.environ.get("QUADRUPED_MCP_TOKEN", ""))
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--model", default="qwen3.8:27b")
    ap.add_argument("--minutes", type=float, default=10.0, help="keep going this long (each chat); 0 = no limit")
    ap.add_argument("--max_calls", type=int, default=200, help="tool calls in one chat")
    ap.add_argument("--think", action="store_true", help="let the model think first (slower)")
    ap.add_argument("--ctx", type=int, default=32768, help="model context in tokens")
    ap.add_argument("--keep", type=int, default=40, help="recent messages kept in context")
    ap.add_argument("--auto", action="store_true", help="never wait for the operator: always tell it to keep going")
    ap.add_argument("--keep_loaded", action="store_true", help="leave the model in Ollama's memory on exit")
    ap.add_argument("--log_dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                      "logs", "agent"), help='the run log\'s folder; "" for none')
    args = ap.parse_args()
    global LOG
    if args.log_dir:
        LOG = RunLog(args.log_dir)
        print(f"Log: {os.path.join(LOG.dir, 'log.md')}")

    mcp = Mcp(args.mcp, args.token)
    instructions, mcp_tools = mcp.start()
    tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                               "parameters": t["inputSchema"]}} for t in mcp_tools]
    print(f"{args.model} with {len(tools)} tools from {args.mcp}: {', '.join(t['name'] for t in mcp_tools)}")
    names = {t["name"] for t in mcp_tools}
    if LOG is not None:
        LOG.header(args, mcp.meta, [t["name"] for t in mcp_tools])
    default_task = FUN if "list_objects" in names else FUN_LOOK if "find_something" in names else EXPLORE

    # The app's Stop arrives while a tool call blocks: its own MCP session sends stop at once.
    stopper = Mcp(args.mcp, args.token)

    def stop_robot():
        try:
            return stopper.call("stop", {})["content"][0]["text"]
        except Exception as e:
            return f"stop failed: {e}"

    calls = 0
    try:
        if args.app:
            stopper.start()
            app = AppChat(stop_robot)
            serve(app, args.port)
            print(f"Waiting for a task from the app's Chat tab (http://<robot>:{args.port}/chat). Ctrl+C ends.")
            system = instructions + "\n" + OPERATOR
            while True:
                task = app.next_task()
                print(f"\n=== task: {task}")
                try:
                    calls += run_chat(app, mcp, args, tools, system, task)
                except OllamaDown as e:
                    # Stopped Ollama (systemctl isolate multi-user.target stops it) ended llm_host and the
                    # server with it (2026-11-04): say so in the chat, and wait for a task again.
                    app.show("error", f"{e}; then send the task again")
                    continue
                if not app.stopped():
                    app.show("info", "chat ended: send a new task, or New chat")
        else:
            ask = not args.auto and sys.stdin.isatty()
            system = instructions + ("\n" + OPERATOR if ask else "")
            log("you", args.task or default_task)
            calls = run_chat(Terminal(ask), mcp, args, tools, system, args.task or default_task)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        try:
            print("stop:", mcp.call("stop", {})["content"][0]["text"])
        except Exception as e:
            print(f"stop failed: {e}")
        if not args.keep_loaded:
            try:
                unload(args.ollama, args.model)
                print(f"{args.model} unloaded")
            except (urllib.error.URLError, OSError) as e:
                print(f"unload failed: {e}")
    print(f"{calls} tool calls")


if __name__ == "__main__":
    main()
