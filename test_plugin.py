"""Tests for tool-events-broadcaster plugin lifecycle and tool window management."""
import importlib.util
import threading
import time
import unittest

# Load the plugin module
_spec = importlib.util.spec_from_file_location(
    "teb", "/home/eifying/.hermes/plugins/tool-events-broadcaster/__init__.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class TestLifecycle(unittest.TestCase):
    """Test hook lifecycle: cleanup, eviction, turn boundaries."""

    def setUp(self):
        _mod._session_instances.clear()
        _mod._active_tools.clear()
        _mod._sessions_with_tools.clear()
        _mod._cascade_counters.clear()

    def tearDown(self):
        _mod._session_instances.clear()
        _mod._active_tools.clear()
        _mod._sessions_with_tools.clear()
        _mod._cascade_counters.clear()

    # ── Cleanup ───────────────────────────────────────────────

    def test_cleanup_removes_all_instances(self):
        sid = "s1"
        _mod._session_instances[sid] = [
            {"instance_id": "i1", "tool_name": "terminal"},
            {"instance_id": "i2", "tool_name": "todo"},
        ]
        _mod._cascade_counters[sid] = 3
        _mod._cleanup_session(sid)
        self.assertNotIn(sid, _mod._session_instances)
        self.assertNotIn(sid, _mod._cascade_counters)
        self.assertNotIn(sid, _mod._sessions_with_tools)

    def test_cleanup_empty_session_noop(self):
        _mod._cleanup_session("nonexistent")  # should not raise

    # ── post_llm_call: no cleanup, just mark boundary ────────

    def test_post_llm_call_does_not_cleanup(self):
        sid = "s2"
        _mod._session_instances[sid] = [{"instance_id": "i1", "tool_name": "terminal"}]
        _mod._sessions_with_tools.add(sid)
        # Simulate post_llm_call logic
        with _mod._lock:
            _mod._sessions_with_tools.discard(sid)
        self.assertIn(sid, _mod._session_instances)
        self.assertNotIn(sid, _mod._sessions_with_tools)

    # ── Turn boundary: stale todo eviction ────────────────────

    def test_evict_stale_todos_removes_only_todos(self):
        sid = "s3"
        _mod._session_instances[sid] = [
            {"instance_id": "t1", "tool_name": "todo"},
            {"instance_id": "n1", "tool_name": "terminal"},
            {"instance_id": "t2", "tool_name": "todo"},
        ]
        _mod._evict_stale_todos(sid)
        remaining = [i["tool_name"] for i in _mod._session_instances[sid]]
        self.assertEqual(remaining, ["terminal"])

    def test_evict_stale_todos_noop_when_no_todos(self):
        sid = "s4"
        _mod._session_instances[sid] = [
            {"instance_id": "n1", "tool_name": "terminal"},
        ]
        _mod._evict_stale_todos(sid)
        self.assertEqual(len(_mod._session_instances[sid]), 1)

    def test_evict_stale_todos_noop_when_empty(self):
        _mod._evict_stale_todos("nonexistent")  # should not raise

    def test_new_turn_evicts_stale_todos(self):
        sid = "s5"
        _mod._session_instances[sid] = [{"instance_id": "old", "tool_name": "todo"}]
        # Simulate pre_tool_call turn detection
        with _mod._lock:
            is_new_turn = sid not in _mod._sessions_with_tools
            _mod._sessions_with_tools.add(sid)
        self.assertTrue(is_new_turn)
        if is_new_turn:
            _mod._evict_stale_todos(sid)
        self.assertEqual(len(_mod._session_instances.get(sid, [])), 0)

    def test_continuing_turn_does_not_evict(self):
        sid = "s6"
        _mod._sessions_with_tools.add(sid)
        _mod._session_instances[sid] = [{"instance_id": "t1", "tool_name": "todo"}]
        with _mod._lock:
            is_new_turn = sid not in _mod._sessions_with_tools
            _mod._sessions_with_tools.add(sid)
        self.assertFalse(is_new_turn)

    # ── FIFO eviction: non-todo evicted first ─────────────────

    def test_fifo_evicts_non_todo_first(self):
        sid = "s7"
        _mod._session_instances[sid] = [
            {"instance_id": "t1", "tool_name": "todo"},
            {"instance_id": "n1", "tool_name": "terminal"},
            {"instance_id": "n2", "tool_name": "read_file"},
            {"instance_id": "n3", "tool_name": "patch"},
        ]
        # At capacity (4), try to evict
        instances = _mod._session_instances[sid]
        victim = None
        for i, inst in enumerate(instances):
            if inst.get("tool_name") != "todo":
                victim = instances.pop(i)
                break
        self.assertIsNotNone(victim)
        self.assertEqual(victim["tool_name"], "terminal")
        remaining = [i["tool_name"] for i in _mod._session_instances[sid]]
        self.assertIn("todo", remaining)

    def test_fifo_no_eviction_below_capacity(self):
        sid = "s8"
        _mod._session_instances[sid] = [
            {"instance_id": "n1", "tool_name": "terminal"},
            {"instance_id": "n2", "tool_name": "read_file"},
        ]
        instances = _mod._session_instances[sid]
        self.assertLess(len(instances), _mod._MAX_WINDOWS)

    # ── Cascade positions ─────────────────────────────────────

    def test_todo_gets_fixed_last_slot(self):
        idx = _mod._MAX_WINDOWS - 1
        self.assertEqual(idx, 3)
        self.assertEqual(20 + idx * _mod._CASCADE_STEP_X, 140)
        self.assertEqual(20 + idx * _mod._CASCADE_STEP_Y, 110)

    def test_non_todo_cycles_0_to_max_minus_2(self):
        positions = []
        sid = "s9"
        _mod._cascade_counters[sid] = 0
        for i in range(6):
            idx = _mod._cascade_counters.get(sid, 0)
            _mod._cascade_counters[sid] = (idx + 1) % (_mod._MAX_WINDOWS - 1)
            positions.append((20 + idx * _mod._CASCADE_STEP_X, 20 + idx * _mod._CASCADE_STEP_Y))
        # First 3 unique, then wraps
        self.assertEqual(len(set(positions[:3])), 3)
        self.assertEqual(positions[3], positions[0])
        self.assertEqual(positions[4], positions[1])

    # ── Hook registration ─────────────────────────────────────

    def test_register_exposes_all_hooks(self):
        registered = []
        _mod.register(type("Ctx", (), {"register_hook": lambda self, n, c: registered.append(n)})())
        self.assertIn("pre_tool_call", registered)
        self.assertIn("post_tool_call", registered)
        self.assertIn("post_llm_call", registered)
        self.assertIn("on_session_end", registered)
        self.assertIn("on_session_reset", registered)

    # ── Constants ─────────────────────────────────────────────

    def test_max_windows_is_4(self):
        self.assertEqual(_mod._MAX_WINDOWS, 4)

    def test_cascade_steps(self):
        self.assertEqual(_mod._CASCADE_STEP_X, 40)
        self.assertEqual(_mod._CASCADE_STEP_Y, 30)


class TestExtractContext(unittest.TestCase):
    """Test context extraction for various tool types."""

    def test_terminal(self):
        self.assertEqual(_mod._extract_context("terminal", {"command": "ls"}), "ls")

    def test_read_file(self):
        self.assertEqual(_mod._extract_context("read_file", {"path": "/tmp/f.py"}), "/tmp/f.py")

    def test_web_search(self):
        self.assertEqual(_mod._extract_context("web_search", {"query": "python"}), "python")

    def test_todo(self):
        self.assertEqual(_mod._extract_context("todo", {"action": "read"}), "read")

    def test_unknown_tool(self):
        self.assertEqual(_mod._extract_context("unknown_tool", {"foo": "bar"}), "")
        self.assertEqual(_mod._extract_context("unknown_tool", {"command": "ls"}), "ls")

    def test_empty_args(self):
        self.assertEqual(_mod._extract_context("terminal", {}), "")


if __name__ == "__main__":
    unittest.main()
