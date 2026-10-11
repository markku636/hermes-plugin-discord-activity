import re
import time
import unittest

from _load import discord_api, render as r, tracker as tr

ANSI = re.compile(r"\u001b\[[0-9;]*m")


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


class FakeClient:
    def __init__(self, fail_create=None):
        self.calls = []
        self.fail_create = fail_create
        self.n = 0

    def create(self, channel, content, reply_to=None):
        self.calls.append(("create", channel, content, reply_to))
        if self.fail_create:
            raise self.fail_create
        self.n += 1
        return f"m{self.n}"

    def edit(self, channel, message_id, content):
        self.calls.append(("edit", channel, message_id, content))

    def delete(self, channel, message_id):
        self.calls.append(("delete", channel, message_id))

    def kinds(self):
        return [c[0] for c in self.calls]

    def last_content(self):
        kind = self.calls[-1][0]
        text = self.calls[-1][2] if kind == "create" else self.calls[-1][3]
        return ANSI.sub("", text)


T = tr.Target(channel_id="c1", reply_to="u1")


def make(cfg=None, client=None):
    clock = Clock()
    client = client or FakeClient()
    t = tr.Tracker(client, cfg or tr.TrackerConfig(), clock=clock)

    def send(*event):
        t.process((clock(),) + event)

    return t, client, clock, send


class TurnTest(unittest.TestCase):
    def test_successful_turn_posts_edits_then_deletes(self):
        t, client, clock, send = make()
        send("turn_start", T, "s1")
        send("api_start", T, "s1", "qwen", 0)
        clock.advance(1.0)
        send("api_end", T, "s1", 1.0, {"prompt_tokens": 100, "output_tokens": 5})
        send("tool_start", T, "s1", "call-1", "mcp__docs__search_pages", {"query": "日本"})
        t.tick()
        self.assertEqual(client.kinds(), ["create"])
        self.assertEqual(client.calls[0][3], "u1")  # replies to the user's message
        clock.advance(0.5)
        send("tool_end", T, "s1", "call-1", "mcp__docs__search_pages", 500, "ok", "", '{"items":[1,2]}')
        t.tick()
        self.assertEqual(client.kinds(), ["create"])  # within edit_interval: coalesced
        clock.advance(1.0)
        t.tick()
        self.assertEqual(client.kinds(), ["create", "edit"])
        final = client.last_content()
        self.assertIn("docs.search_pages ▾ ✔ [500ms]", final)
        self.assertIn("result: 2 items", final)
        send("turn_end", T, "s1")
        self.assertNotIn("c1", t.runs)
        t.tick()
        self.assertNotIn("delete", client.kinds())  # waits delete_delay
        clock.advance(2.0)
        t.tick()
        self.assertEqual(client.kinds()[-1], "delete")

    def test_failed_turn_is_kept(self):
        t, client, clock, send = make()
        send("turn_start", T, "s1")
        send("tool_start", T, "s1", "c", "get_order", {"id": "x"})
        t.tick()
        send("tool_end", T, "s1", "c", "get_order", 30, "error", "order not found", None)
        send("turn_end", T, "s1")
        clock.advance(10)
        t.tick()
        self.assertNotIn("delete", client.kinds())
        self.assertIn("error: order not found", client.last_content())

    def test_keep_after_done(self):
        t, client, clock, send = make(tr.TrackerConfig(keep_after_done=True))
        send("turn_start", T, "s1")
        send("tool_start", T, "s1", "c", "web_fetch", {"url": "https://example.com"})
        send("tool_end", T, "s1", "c", "web_fetch", 30, "ok", "", "hi")
        send("turn_end", T, "s1")  # ended before the first tick: still posts a final card
        clock.advance(10)
        t.tick()
        self.assertEqual(client.kinds(), ["create"])

    def test_quick_chat_turn_posts_nothing(self):
        t, client, clock, send = make()
        send("turn_start", T, "s1")
        send("api_start", T, "s1", "qwen", 0)
        clock.advance(1.0)
        t.tick()
        send("api_end", T, "s1", 1.0, {})
        send("turn_end", T, "s1")
        clock.advance(10)
        t.tick()
        self.assertEqual(client.calls, [])

    def test_slow_model_shows_after_threshold(self):
        t, client, clock, send = make()
        send("turn_start", T, "s1")
        send("api_start", T, "s1", "qwen", 0)
        clock.advance(3.5)
        t.tick()
        self.assertEqual(client.kinds(), ["create"])
        self.assertIn("🧠 qwen", ANSI.sub("", client.calls[0][2]))

    def test_model_retries_and_error(self):
        t, client, clock, send = make()
        send("turn_start", T, "s1")
        send("api_start", T, "s1", "qwen", 0)
        clock.advance(1)
        send("api_error", T, "s1", 1.0, "429 model_busy")
        send("api_start", T, "s1", "qwen", 1)
        clock.advance(1)
        send("api_error", T, "s1", 1.0, "429 model_busy")
        send("turn_end", T, "s1")
        text = client.last_content()
        self.assertIn("🧠 qwen ▾ ✘ [2.00s]", text)
        self.assertIn("retry: 1", text)
        self.assertIn("error: 429 model_busy", text)
        clock.advance(10)
        t.tick()
        self.assertNotIn("delete", client.kinds())

    def test_subagent_turn_end_does_not_close_parent(self):
        t, client, clock, send = make()
        send("turn_start", T, "parent")
        send("tool_start", T, "child", "c", "web_search", {"q": "x"})
        send("turn_end", T, "child")
        self.assertIn("c1", t.runs)
        send("turn_end", T, "parent")
        self.assertNotIn("c1", t.runs)

    def test_new_turn_closes_unfinished_previous(self):
        t, client, clock, send = make()
        send("turn_start", T, "s1")
        send("tool_start", T, "s1", "c", "terminal", {"command": "sleep 100"})
        t.tick()
        send("turn_start", T, "s1")
        last = client.last_content()
        self.assertIn("■", last)  # marked stopped, not left spinning
        self.assertEqual(len(t.runs["c1"].tools), 0)

    def test_tool_end_without_call_id_matches_by_name(self):
        t, client, clock, send = make()
        send("turn_start", T, "s1")
        send("tool_start", T, "s1", "", "terminal", {"command": "a"})
        send("tool_start", T, "s1", "", "terminal", {"command": "b"})
        send("tool_end", T, "s1", "", "terminal", 10, "ok", "", "done")
        statuses = [x.status for x in t.runs["c1"].tools]
        self.assertEqual(statuses, [r.OK, r.PENDING])

    def test_permanent_error_stops_touching_the_channel(self):
        client = FakeClient(fail_create=discord_api.PermanentError("403"))
        t, client, clock, send = make(client=client)
        send("turn_start", T, "s1")
        send("tool_start", T, "s1", "c", "x", {})
        t.tick()
        clock.advance(5)
        send("tool_end", T, "s1", "c", "x", 1, "ok", "", "")
        t.tick()
        send("turn_end", T, "s1")
        self.assertEqual(client.kinds(), ["create"])

    def test_idle_run_is_closed(self):
        t, client, clock, send = make(tr.TrackerConfig(max_display_seconds=60))
        send("turn_start", T, "s1")
        send("tool_start", T, "s1", "c", "x", {})
        t.tick()
        clock.advance(61)
        t.tick()
        self.assertNotIn("c1", t.runs)

    def test_running_timer_refreshes_slowly(self):
        t, client, clock, send = make()
        send("turn_start", T, "s1")
        send("tool_start", T, "s1", "c", "tts", {})
        t.tick()
        for _ in range(10):  # 10 seconds of a running tool, ticking every second
            clock.advance(1)
            t.tick()
        self.assertEqual(client.kinds().count("edit"), 2)  # every timer_refresh (5s), not every tick


class HookLatencyTest(unittest.TestCase):
    def test_hooks_return_immediately_even_if_discord_hangs(self):
        from _load import pkg

        class SlowClient(FakeClient):
            def create(self, *a, **k):
                time.sleep(1.0)
                return super().create(*a, **k)

        t = tr.Tracker(SlowClient(), tr.TrackerConfig())
        t.start()
        pkg._tracker = t
        pkg._remember("s1", T)
        try:
            started = time.perf_counter()
            for i in range(50):
                pkg.on_pre_tool_call(tool_name="terminal", args={"command": "ls"}, session_id="s1", tool_call_id=str(i))
                pkg.on_post_tool_call(tool_name="terminal", result="ok", session_id="s1", tool_call_id=str(i), duration_ms=1)
            elapsed = time.perf_counter() - started
            self.assertLess(elapsed, 0.05, f"100 hook calls took {elapsed:.3f}s")
        finally:
            pkg._tracker = None

    def test_non_discord_platform_is_ignored(self):
        from _load import pkg

        t = tr.Tracker(FakeClient(), tr.TrackerConfig())
        pkg._tracker = t
        try:
            pkg.on_pre_llm_call(session_id="line-1", platform="line")
            pkg.on_pre_api_request(session_id="line-1", platform="line", model="m")
            self.assertTrue(t.events.empty())
        finally:
            pkg._tracker = None


if __name__ == "__main__":
    unittest.main()
