import re
import unittest

from _load import render as r

ANSI = re.compile(r"\u001b\[[0-9;]*m")


def plain(text):
    return ANSI.sub("", text)


class FormatTest(unittest.TestCase):
    def test_duration(self):
        self.assertEqual(r.format_duration(850), "850ms")
        self.assertEqual(r.format_duration(999.6), "1000ms")
        self.assertEqual(r.format_duration(4210), "4.21s")
        self.assertEqual(r.format_duration(12345), "12.3s")

    def test_clip_counts_code_points_and_hints(self):
        cfg = r.RenderConfig(locale="zh-TW")
        self.assertEqual(r.clip("短", 70, cfg), "短")
        out = plain(r.clip("字" * 100, 70, cfg))
        self.assertTrue(out.startswith("字" * 70))
        self.assertTrue(out.endswith("(+30 字)"))

    def test_clip_flattens_newlines(self):
        self.assertEqual(r.clip("a\n  b\tc", 70, r.RenderConfig()), "a b c")

    def test_mcp_name(self):
        self.assertEqual(r.normalize_tool_name("mcp__docs__search_pages"), "docs.search_pages")
        self.assertEqual(r.normalize_tool_name("web_fetch"), "web_fetch")
        self.assertEqual(r.mcp_server_of("mcp__seal_tts__tts_synthesize"), "seal_tts")
        self.assertIsNone(r.mcp_server_of("terminal"))

    def test_secret_args_are_masked(self):
        args = dict(r.summarize_args({"url": "https://x", "api_key": "sk-123", "Authorization": "Bearer y",
                                      "empty": "", "none": None, "limit": 5}))
        self.assertEqual(args["api_key"], "***")
        self.assertEqual(args["Authorization"], "***")
        self.assertEqual(args["limit"], "5")
        self.assertNotIn("empty", args)
        self.assertNotIn("none", args)

    def test_result_summaries(self):
        cfg = r.RenderConfig()
        self.assertEqual(r.summarize_result('[1,2,3]', cfg), "3 items")
        self.assertEqual(r.summarize_result('{"items":[1,2]}', cfg), "2 items")
        self.assertEqual(r.summarize_result('{"content":[{"type":"text","text":"{\\"packages\\":[1,2,3,4]}"}]}', cfg), "4 items")
        self.assertEqual(r.summarize_result('{"result":"7 memories recalled"}', cfg), "7 memories recalled")
        self.assertEqual(r.summarize_result("plain text", cfg), "plain text")
        self.assertIsNone(r.summarize_result("", cfg))


class LayoutTest(unittest.TestCase):
    def tool(self, i, status=r.OK, args=None, result="ok"):
        t = r.ToolEntry(call_id=str(i), name=f"tool_{i}", icon="⚙️", started=0.0,
                        args=args or [("query", f"q{i}")], status=status, result=result)
        if status != r.PENDING:
            t.duration_ms = 1000.0 + i
        return t

    def test_card_shape(self):
        model = r.ModelEntry(model="qwen", calls=2, total_ms=4210, prompt_tokens=12480, output_tokens=312, status=r.OK)
        tools = [self.tool(1), r.ToolEntry(call_id="2", name="docs.search_pages", icon="📚", started=10.0,
                                           args=[("query", "日本 5GB"), ("limit", "5")])]
        out = r.render(model, tools, now=11.15, cfg=r.RenderConfig())
        self.assertTrue(out.startswith("```ansi\n") and out.endswith("\n```"))
        text = plain(out)
        self.assertIn("🧠 qwen ▾ ✔ [4.21s]", text)
        self.assertIn("calls: 2 · tokens: 12,480 → 312", text)
        self.assertIn("📚 docs.search_pages ▾ ← [1.15s]", text)
        self.assertIn("├─ query: q1", text)
        self.assertIn("└─ result: ok", text)
        self.assertIn("query: 日本 5GB · limit: 5", text)

    def test_error_row(self):
        t = self.tool(1, status=r.ERROR, result=None)
        t.error = "404 not found"
        text = plain(r.render(None, [t], 0, r.RenderConfig()))
        self.assertIn("✘", text)
        self.assertIn("└─ error: 404 not found", text)

    def test_slot_budget_removes_oldest_finished_first(self):
        tools = [self.tool(i) for i in range(10)] + [self.tool(10, status=r.PENDING)]
        text = plain(r.render(r.ModelEntry(model="m"), tools, 0, r.RenderConfig(max_rows=6)))
        self.assertIn("… 6 earlier", text)
        self.assertNotIn("tool_5 ", text)
        for i in (6, 7, 8, 9, 10):
            self.assertIn(f"tool_{i} ", text)

    def test_length_budget_and_monotonic(self):
        big = [("text", "x" * 300), ("voice", "y" * 300), ("style", "z" * 300)]
        tools = [self.tool(i, args=big, result="r" * 200) for i in range(5)]
        cfg = r.RenderConfig(max_rows=6, max_length=700)
        out = r.render(None, tools, 0, cfg)
        self.assertLessEqual(len(out), 700)
        levels = [t.level for t in tools]
        self.assertTrue(any(l != r.EXPANDED for l in levels))
        # a later render with more room never re-expands what was folded
        r.render(None, tools, 0, r.RenderConfig(max_rows=6, max_length=1900))
        self.assertEqual([max(a, b) for a, b in zip(levels, [t.level for t in tools])], [t.level for t in tools])
        self.assertTrue(all(b >= a for a, b in zip(levels, [t.level for t in tools])))

    def test_hard_cap_never_exceeded(self):
        tools = [self.tool(i, status=r.PENDING, args=[("k", "v" * 500)]) for i in range(6)]
        out = r.render(r.ModelEntry(model="m" * 500), tools, 0, r.RenderConfig(max_length=300))
        self.assertLessEqual(len(out), 300)
        self.assertTrue(out.endswith("```"))

    def test_zh_tw_labels(self):
        tools = [self.tool(i) for i in range(8)]
        text = plain(r.render(None, tools, 0, r.RenderConfig(locale="zh-TW")))
        self.assertIn("… 前面還有 2 個", text)


if __name__ == "__main__":
    unittest.main()
