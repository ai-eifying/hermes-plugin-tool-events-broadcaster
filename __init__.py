"""tool-events-broadcaster plugin — create tool windows via JARVIS API.

Listens to pre_tool_call and post_tool_call hooks.
On tool.start: creates a new tool-window-single instance via JARVIS API
On tool.complete: pushes update data to the instance
On post_llm_call: kills all tool windows created in this turn
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

JARVIS_URL = os.getenv("JARVIS_DASHBOARD_URL", "http://localhost:9090")

# Per-session tracking: session_id -> List[{tool_id, instance_id, tool_name, started_at}]
_session_instances: Dict[str, List[dict]] = {}
# Active tools: tool_id -> {instance_id, session_id}
_active_tools: Dict[str, dict] = {}
# Track which sessions had tools in current turn
_sessions_with_tools: set = set()
_lock = threading.Lock()


def _api_call(method: str, path: str, data: dict = None) -> dict:
    """Call JARVIS REST API."""
    try:
        import urllib.request
        url = f"{JARVIS_URL}{path}"
        body = json.dumps(data).encode("utf-8") if data else None
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method=method
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"[tool-events] API error {path}: {e}", file=sys.stderr)
        return {}


def _read_file_content(path: str) -> str:
    """Read file content for diff display."""
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception:
        pass
    return ""


def _spawn_tool_window(tool_id: str, name: str, context: str, args: dict) -> str | None:
    """Create a new tool-window-single instance."""
    config = {
        "tool_id": tool_id,
        "tool_name": name,
        "context": context,
        "args": args
    }
    
    # For write_file, read old content for diff display
    if name == "write_file" and args.get("path"):
        config["old_content"] = _read_file_content(args["path"])
    
    result = _api_call("POST", "/api/plugins/spawn", {
        "plugin_id": "tool-window-single",
        "config": config
    })
    if result.get("ok"):
        return result["instance"]["instance_id"]
    return None


def _kill_instance(instance_id: str):
    """Kill a tool-window-single instance."""
    _api_call("POST", f"/api/plugins/{instance_id}/kill")


def _push_update(instance_id: str, data: dict):
    """Push update data to an instance."""
    _api_call("POST", f"/api/plugins/{instance_id}/data", {
        "key": "data",
        "value": data
    })


def _cleanup_session(session_id: str):
    """Kill all tool windows created in this session."""
    with _lock:
        instances = _session_instances.pop(session_id, [])
    
    if not instances:
        return
    
    for inst in instances:
        iid = inst.get("instance_id")
        if iid:
            _kill_instance(iid)


def _extract_context(tool_name: str, args: dict) -> str:
    """Extract display context based on tool type."""
    if tool_name == 'terminal':
        return str(args.get("command", ""))[:100]
    elif tool_name in ('read_file', 'write_file', 'patch'):
        return str(args.get("path", ""))[:100]
    elif tool_name in ('vision_analyze', 'browser_vision'):
        return str(args.get("image_url", args.get("question", "")))[:100]
    elif tool_name == 'web_search':
        return str(args.get("query", ""))[:100]
    elif tool_name == 'web_extract':
        return str(args.get("url", ""))[:100]
    elif tool_name == 'search_files':
        return str(args.get("pattern", ""))[:100]
    elif tool_name == 'memory':
        return str(args.get("action", "query"))[:50]
    elif tool_name in ('skill_view', 'skill_manage'):
        return str(args.get("name", ""))[:100]
    elif tool_name == 'todo':
        return str(args.get("action", "view"))[:50]
    elif tool_name == 'execute_code':
        return "Python"
    elif tool_name == 'send_message':
        return str(args.get("target", args.get("message", "")))[:100]
    elif tool_name == 'cronjob':
        return str(args.get("action", "list"))[:50]
    elif tool_name == 'delegate_task':
        return str(args.get("goal", ""))[:100]
    elif tool_name == 'browser_navigate':
        return str(args.get("url", ""))[:100]
    elif tool_name == 'browser_click':
        return str(args.get("ref", ""))[:50]
    elif tool_name == 'browser_type':
        return str(args.get("text", ""))[:50]
    elif tool_name == 'browser_snapshot':
        return "snapshot"
    elif tool_name == 'browser_scroll':
        return str(args.get("direction", "down"))[:20]
    elif tool_name == 'browser_console':
        return str(args.get("expression", "logs"))[:50]
    else:
        for key in ('command', 'path', 'query', 'url', 'name', 'expression'):
            if key in args:
                return str(args[key])[:100]
        return ""


def register(ctx) -> None:
    """Plugin entry point — register hooks."""
    
    def on_pre_tool_call(tool_name, args, task_id="", session_id="", tool_call_id="", **kwargs):
        try:
            tid = tool_call_id or f"{tool_name}-{task_id}-{int(time.time()*1000)}"
            sid = session_id or "default"
            context = _extract_context(tool_name, args or {})
            
            with _lock:
                _sessions_with_tools.add(sid)
            
            def do_spawn():
                instance_id = _spawn_tool_window(tid, tool_name, context, args or {})
                if instance_id:
                    with _lock:
                        _active_tools[tid] = {
                            "instance_id": instance_id,
                            "session_id": sid
                        }
                        if sid not in _session_instances:
                            _session_instances[sid] = []
                        _session_instances[sid].append({
                            "tool_id": tid,
                            "instance_id": instance_id,
                            "tool_name": tool_name,
                            "started_at": time.time()
                        })
            
            threading.Thread(target=do_spawn, daemon=True).start()
        except Exception as e:
            print(f"[tool-events] pre_tool_call error: {e}", file=sys.stderr)
    
    def on_post_tool_call(tool_name, args, result=None, task_id="", session_id="", 
                          duration_ms=0, status="", error_type="", error_message="", 
                          tool_call_id="", **kwargs):
        try:
            tid = tool_call_id or f"{tool_name}-{task_id}-0"
            
            with _lock:
                tool_info = _active_tools.get(tid)
                instance_id = tool_info["instance_id"] if tool_info else None
            
            if not instance_id:
                return
            
            output = ""
            if isinstance(result, str):
                try:
                    d = json.loads(result)
                    output = d.get("output", d.get("result", d.get("content", "")))[:500]
                except:
                    output = result[:500]
            elif isinstance(result, dict):
                output = str(result.get("output", result.get("result", result.get("content", ""))))[:500]
            
            def do_update():
                _push_update(instance_id, {
                    "status": status or ("failed" if error_type else "completed"),
                    "duration": (duration_ms / 1000.0) if duration_ms else 0,
                    "output": output,
                    "error": error_message if error_type else None
                })
                with _lock:
                    _active_tools.pop(tid, None)
            
            threading.Thread(target=do_update, daemon=True).start()
        except Exception as e:
            print(f"[tool-events] post_tool_call error: {e}", file=sys.stderr)
    
    def on_post_llm_call(session_id="", **kwargs):
        """Clean up all tool windows when turn ends (LLM call completes)."""
        sid = session_id or "default"
        
        with _lock:
            had_tools = sid in _sessions_with_tools
            _sessions_with_tools.discard(sid)
        
        if not had_tools:
            return
        
        threading.Thread(target=_cleanup_session, args=(sid,), daemon=True).start()
    
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
