"""LLM host: a local model on Ollama drives the robot through the MCP server's tools.

Lists the server's tools (mcp_server in quadruped_operator), hands them to the model, runs each tool call
it makes and gives it the result, until it answers without one. With --minutes it is told to keep going
until the time is up. Ctrl+C sends stop before leaving. Standard library only, so it runs on any machine
that reaches both Ollama and the robot:

    python3 Tools/llm_host.py                                  # the "having fun" loop, 10 minutes
    python3 Tools/llm_host.py "walk to the nearest cone and sniff it" --minutes 0
    python3 Tools/llm_host.py --mcp http://<robot>:8765/mcp --model qwen3.5:9b

QUADRUPED_MCP_TOKEN (or --token) is sent as the bearer token when the server requires one.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request

FUN = """You are the robot dog. Have fun with the traffic cones in this room, on your own:
1. list_objects to see the cones you know about. Pick one whose "done" list is empty, or the one done longest ago.
2. go_to_object it, then skill sniff. Sometimes (about half the time) also pee_left or pee_right.
3. If no cone is left to visit, look for more: move to turn on the spot (turn 0.8 for 3 s), list_objects again,
   or go_to an open spot a few metres away.
Call skill walk first if you are not walking. If a call fails, read why and try something else.
Keep each message short: one line on what you do next, then the tool call."""

KEEP_GOING = "Keep going: pick the next thing to do and call a tool."


class Mcp:
    def __init__(self, url, token):
        self.url, self.token, self.session, self.next_id = url, token, None, 1

    def _post(self, msg):
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        req = urllib.request.Request(self.url, json.dumps(msg).encode(), headers)
        with urllib.request.urlopen(req, timeout=600) as r:
            self.session = r.headers.get("Mcp-Session-Id") or self.session
            body = r.read().decode()
            if "event-stream" in r.headers.get("Content-Type", ""):
                lines = [ln[5:].strip() for ln in body.splitlines() if ln.startswith("data:")]
                body = lines[-1] if lines else ""
        return json.loads(body) if body.strip() else None

    def request(self, method, params=None):
        msg = {"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params or {}}
        self.next_id += 1
        reply = self._post(msg)
        if "error" in reply:
            raise RuntimeError(f"{method}: {reply['error'].get('message')}")
        return reply["result"]

    def start(self):
        init = self.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                           "clientInfo": {"name": "llm_host", "version": "0.1.0"}})
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return init.get("instructions", ""), self.request("tools/list")["tools"]

    def call(self, name, args):
        return self.request("tools/call", {"name": name, "arguments": args})


def chat(url, model, messages, tools, think, ctx):
    body = {"model": model, "messages": messages, "tools": tools, "stream": False, "think": think,
            "options": {"num_ctx": ctx}}
    req = urllib.request.Request(f"{url}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["message"]


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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("task", nargs="?", default=FUN, help="what to do (default: the having-fun loop)")
    ap.add_argument("--mcp", default="http://127.0.0.1:8765/mcp")
    ap.add_argument("--token", default=os.environ.get("QUADRUPED_MCP_TOKEN", ""))
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--model", default="qwen3.8:27b")
    ap.add_argument("--minutes", type=float, default=10.0, help="keep going this long; 0 = until it stops")
    ap.add_argument("--max_calls", type=int, default=200)
    ap.add_argument("--think", action="store_true", help="let the model think first (slower)")
    ap.add_argument("--ctx", type=int, default=32768, help="model context in tokens")
    ap.add_argument("--keep", type=int, default=40, help="recent messages kept in context")
    args = ap.parse_args()

    mcp = Mcp(args.mcp, args.token)
    instructions, mcp_tools = mcp.start()
    tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                               "parameters": t["inputSchema"]}} for t in mcp_tools]
    print(f"{args.model} with {len(tools)} tools from {args.mcp}: {', '.join(t['name'] for t in mcp_tools)}")
    messages = [{"role": "system", "content": instructions}, {"role": "user", "content": args.task}]
    end = time.monotonic() + args.minutes * 60
    calls = 0
    try:
        while calls < args.max_calls:
            trim(messages, args.keep)
            t = time.monotonic()
            reply = chat(args.ollama, args.model, messages, tools, args.think, args.ctx)
            said = (reply.get("content") or "").strip()
            if said:
                print(f"\n[{time.monotonic() - t:.1f}s] {said}")
            todo = reply.get("tool_calls") or []
            messages.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": todo})
            if not todo:
                if args.minutes <= 0 or time.monotonic() > end:
                    break
                messages.append({"role": "user", "content": KEEP_GOING})
                continue
            for tc in todo:
                name, targs = tc["function"]["name"], tc["function"].get("arguments") or {}
                if isinstance(targs, str):
                    targs = json.loads(targs or "{}")
                print(f"  -> {name}({json.dumps(targs)[1:-1]})", flush=True)
                try:
                    res = mcp.call(name, targs)
                except (urllib.error.URLError, RuntimeError) as e:
                    res = {"content": [{"type": "text", "text": f"call failed: {e}"}], "isError": True}
                calls += 1
                texts = [c["text"] for c in res.get("content", []) if c.get("type") == "text"]
                images = [c["data"] for c in res.get("content", []) if c.get("type") == "image"]
                out = "\n".join(texts) or ("photo attached below" if images else "(no output)")
                print(f"     {'ERROR ' if res.get('isError') else ''}{out[:300]}")
                messages.append({"role": "tool", "tool_name": name, "content": out})
                if images:
                    messages.append({"role": "user", "content": f"The photo from {name}.", "images": images})
            if args.minutes > 0 and time.monotonic() > end:
                print(f"\n{args.minutes:g} minutes are up.")
                break
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        try:
            print("stop:", mcp.call("stop", {})["content"][0]["text"])
        except Exception as e:
            print(f"stop failed: {e}")
    print(f"{calls} tool calls")


if __name__ == "__main__":
    main()
