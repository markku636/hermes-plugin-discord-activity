import json
import re
import unittest

from _load import render as r, tracker as tr
from test_tracker import FakeClient, T, make

ANSI = re.compile(r"\u001b\[[0-9;]*m")


def plain(text):
    return ANSI.sub("", text)


class ParseProgressTest(unittest.TestCase):
    def frac(self, result):
        p = r.parse_progress(result)
        return None if p is None else round(p.fraction, 3)

    def test_formats(self):
        self.assertEqual(self.frac('{"status":"running","progress":"12/40"}'), 0.3)   # seal_tts jobs
        self.assertEqual(self.frac('{"progress":0.45}'), 0.45)                          # 0–1
        self.assertEqual(self.frac('{"progress":45}'), 0.45)                            # 0–100
        self.assertEqual(self.frac('{"progress":"45%"}'), 0.45)
        self.assertEqual(self.frac('{"percent":80}'), 0.8)
        self.assertEqual(self.frac('{"step":3,"total":4}'), 0.75)
        self.assertEqual(self.frac('{"job":{"progress":"2/4","state":"training"}}'), 0.5)
        self.assertEqual(self.frac('{"content":[{"type":"text","text":"{\\"progress\\":\\"1/2\\"}"}]}'), 0.5)

    def test_no_numbers_means_no_bar(self):
        for result in ['{"status":"running"}', '{"progress":"rendering"}', '{"progress":null}',
                       '{"progress":250}', '{"progress":"5/3"}', '{"step":3}', 'plain text', '', None,
                       '{"progress":true}', '[1,2,3]']:
            self.assertIsNone(r.parse_progress(result), result)

    def test_status_is_kept(self):
        self.assertEqual(r.parse_progress('{"status":"running","progress":"1/4"}').status, "running")

    def test_bar(self):
        out = plain(r.progress_bar(r.Progress(fraction=0.6, done=3, total=5)))
        self.assertEqual(out, "▕██████░░░░▏ 3/5  60%")
        self.assertEqual(plain(r.progress_bar(r.Progress(fraction=0.0))), "▕░░░░░░░░░░▏  0%")
        self.assertEqual(plain(r.progress_bar(r.Progress(fraction=1.0, done=40, total=40))), "▕██████████▏ 40/40  100%")


class PlanTest(unittest.TestCase):
    TODOS = json.dumps({"todos": [
        {"id": "1", "content": "查 Ani 的定妝照", "status": "completed"},
        {"id": "2", "content": "生成海邊照", "status": "in_progress"},
        {"id": "3", "content": "生成影片", "status": "pending"},
        {"id": "4", "content": "不做了", "status": "cancelled"},
    ], "summary": {}}, ensure_ascii=False)

    def test_plan_row_counts_and_current(self):
        plan = r.parse_todos(self.TODOS)
        self.assertEqual(plan.counts(), (1, 3))  # cancelled not in the denominator
        text = plain(r.render(None, [], 0, r.RenderConfig(locale="zh-TW"), plan))
        self.assertIn("📋 計畫 ▕███░░░░░░░▏ 1/3  33%", text)
        self.assertIn("▶ 生成海邊照", text)

    def test_all_done_is_100_without_current(self):
        plan = r.PlanEntry(items=[("a", "completed"), ("b", "completed")])
        text = plain(r.render(None, [], 0, r.RenderConfig(), plan))
        self.assertIn("2/2  100%", text)
        self.assertNotIn("▶", text)

    def test_not_a_todo_payload(self):
        self.assertIsNone(r.parse_todos('{"items":[1]}'))
        self.assertIsNone(r.parse_todos('{"todos":[]}'))

    def test_tracker_shows_plan_not_a_todo_row(self):
        t, client, clock, send = make()
        send("turn_start", T, "s1")
        send("tool_start", T, "s1", "c1", "todo_list", {"todos": []})
        send("tool_end", T, "s1", "c1", "todo_list", 5, "ok", "", self.TODOS)
        t.tick()
        text = client.last_content()
        self.assertIn("📋 plan", text)
        self.assertNotIn("todo_list", text)

    def test_plan_keeps_its_slot_under_pressure(self):
        plan = r.parse_todos(self.TODOS)
        tools = [r.ToolEntry(call_id=str(i), name=f"tool_{i}", icon="⚙️", started=0, status=r.OK, duration_ms=1,
                             args=[("q", "x" * 200)], result="y" * 200) for i in range(12)]
        out = r.render(r.ModelEntry(model="m"), tools, 0, r.RenderConfig(max_length=900), plan)
        self.assertLessEqual(len(out), 900)
        self.assertIn("📋 plan", plain(out))


class PollMergeTest(unittest.TestCase):
    def test_same_job_polls_update_one_row(self):
        t, client, clock, send = make()
        send("turn_start", T, "s1")
        for i, prog in enumerate(["2/40", "12/40", "30/40"]):
            send("tool_start", T, "s1", f"c{i}", "mcp__seal_tts__tts_podcast_status", {"job_id": "8f2c"})
            send("tool_end", T, "s1", f"c{i}", "mcp__seal_tts__tts_podcast_status", 300, "ok", "",
                 json.dumps({"status": "running", "progress": prog}))
        run = t.runs["c1"]
        self.assertEqual(len(run.tools), 1)
        self.assertEqual(run.tools[0].repeats, 3)
        t.tick()
        text = client.last_content()
        self.assertIn("seal_tts.tts_podcast_status ▾ ✔ [300ms] ×3", text)
        self.assertIn("▕████████░░▏ 30/40  75% · running", text)
        self.assertNotIn("result:", text)  # the bar replaces the raw result line

    def test_different_jobs_stay_separate(self):
        t, client, clock, send = make()
        send("turn_start", T, "s1")
        for job in ("a", "b"):
            send("tool_start", T, "s1", job, "mcp__ai_gen__get_generation", {"id": job})
            send("tool_end", T, "s1", job, "mcp__ai_gen__get_generation", 10, "ok", "", '{"status":"done"}')
        self.assertEqual(len(t.runs["c1"].tools), 2)

    def test_no_job_id_never_merges(self):
        t, client, clock, send = make()
        send("turn_start", T, "s1")
        for i in range(2):
            send("tool_start", T, "s1", f"c{i}", "web_search", {"query": "x"})
            send("tool_end", T, "s1", f"c{i}", "web_search", 10, "ok", "", "[]")
        self.assertEqual(len(t.runs["c1"].tools), 2)

    def test_tool_without_progress_has_no_bar(self):
        t, client, clock, send = make()
        send("turn_start", T, "s1")
        send("tool_start", T, "s1", "c", "mcp__ai_gen__get_generation", {"id": "g1"})
        send("tool_end", T, "s1", "c", "mcp__ai_gen__get_generation", 10, "ok", "", '{"status":"queued"}')
        t.tick()
        text = client.last_content()
        self.assertNotIn("▕", text)
        self.assertIn("result: queued", text)


if __name__ == "__main__":
    unittest.main()
