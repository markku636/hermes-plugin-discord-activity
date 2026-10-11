"""Per-channel run state, driven by events from one background thread.

Hooks never touch this state directly: they put a tuple on ``Tracker.events`` and return. Only the
worker thread mutates runs and talks to Discord, so there are no locks around run state and a slow
or failing Discord call can never stall (or, for ``pre_tool_call``, block) the agent.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import render as r
from .discord_api import DiscordClient, PermanentError, TransientError

logger = logging.getLogger(__name__)


@dataclass
class TrackerConfig:
    keep_after_done: bool = False
    show_model: bool = True
    show_after_seconds: float = 3.0
    edit_interval: float = 1.2
    timer_refresh: float = 5.0  # re-render running timers this often even without new events
    delete_delay: float = 2.0
    max_display_seconds: float = 600.0
    reply_to_message: bool = True
    render: r.RenderConfig = field(default_factory=r.RenderConfig)


@dataclass
class Target:
    channel_id: str
    reply_to: str = ""


@dataclass
class Run:
    target: Target
    session_id: str
    started: float
    last_event: float
    model: Optional[r.ModelEntry] = None
    plan: Optional[r.PlanEntry] = None
    tools: List[r.ToolEntry] = field(default_factory=list)
    message_id: Optional[str] = None
    last_content: str = ""
    last_flush: float = 0.0
    dead: bool = False  # Discord said no (permissions, deleted message): stop trying
    dirty: bool = True
    model_error_open: bool = False

    @property
    def failed(self) -> bool:
        if any(t.status == r.ERROR for t in self.tools):
            return True
        return self.model is not None and self.model.status == r.ERROR


PLAN_TOOLS = {"todo_list", "todo"}
# Argument names that identify one long-running job across repeated status polls
_JOB_KEYS = ("job_id", "jobId", "generation_id", "generationId", "task_id", "taskId", "run_id", "runId", "id")


def job_key(name: str, args: Any) -> Optional[str]:
    if not isinstance(args, dict):
        return None
    for key in _JOB_KEYS:
        value = args.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return f"{name}|{key}={value}"
    return None


class Tracker:
    def __init__(
        self,
        client: DiscordClient,
        cfg: TrackerConfig,
        icon_for: Callable[[str], str] = lambda name: "⚙️",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.cfg = cfg
        self.icon_for = icon_for
        self.clock = clock
        self.events: "queue.Queue[Tuple[Any, ...]]" = queue.Queue()
        self.runs: Dict[str, Run] = {}
        self.pending_deletes: List[Tuple[float, str, str, int]] = []  # (due, channel, message, attempt)
        self._thread: Optional[threading.Thread] = None

    # ── producer side (any thread) ────────────────────────────────────────────────────────────

    def submit(self, *event: Any) -> None:
        self.events.put((self.clock(),) + event)

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="discord-activity", daemon=True)
            self._thread.start()

    # ── worker ────────────────────────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while True:
            try:
                event = self.events.get(timeout=0.25)
            except queue.Empty:
                event = None
            try:
                if event is not None:
                    self.process(event)
                    # drain whatever else is queued before touching the network
                    while True:
                        try:
                            self.process(self.events.get_nowait())
                        except queue.Empty:
                            break
                self.tick()
            except Exception:  # never let the worker die
                logger.exception("discord-activity worker error")

    def process(self, event: Tuple[Any, ...]) -> None:
        now, kind = event[0], event[1]
        handler = getattr(self, f"_on_{kind}", None)
        if handler is None:
            logger.debug("discord-activity: unknown event %s", kind)
            return
        handler(now, *event[2:])

    def tick(self) -> None:
        now = self.clock()
        for channel_id, run in list(self.runs.items()):
            if now - run.last_event > self.cfg.max_display_seconds:
                self._finish(channel_id, now, stopped=True)
                continue
            if not self._should_show(run, now):
                continue
            since = now - run.last_flush
            if (run.dirty and since >= self.cfg.edit_interval) or (self._has_running(run) and since >= self.cfg.timer_refresh):
                self._flush(run, now)
        due = [d for d in self.pending_deletes if d[0] <= now]
        self.pending_deletes = [d for d in self.pending_deletes if d[0] > now]
        for _, channel_id, message_id, attempt in due:
            try:
                self.client.delete(channel_id, message_id)
            except TransientError as exc:
                if attempt == 0:
                    self.pending_deletes.append((now + 5.0, channel_id, message_id, 1))
                logger.info("discord-activity: delete failed (%s)", exc)
            except PermanentError as exc:
                logger.info("discord-activity: delete refused (%s)", exc)

    # ── event handlers ────────────────────────────────────────────────────────────────────────

    def _run_for(self, target: Target, session_id: str, now: float) -> Run:
        run = self.runs.get(target.channel_id)
        if run is None:
            run = Run(target=target, session_id=session_id, started=now, last_event=now)
            self.runs[target.channel_id] = run
        return run

    def _on_turn_start(self, now: float, target: Target, session_id: str) -> None:
        old = self.runs.get(target.channel_id)
        if old is not None:
            # a previous turn in this channel never reported its end: close it as-is
            self._finish(target.channel_id, now, stopped=True)
        self.runs[target.channel_id] = Run(target=target, session_id=session_id, started=now, last_event=now)

    def _on_turn_end(self, now: float, target: Target, session_id: str) -> None:
        run = self.runs.get(target.channel_id)
        if run is None or run.session_id != session_id:
            return  # a subagent's turn ending, or a turn we never saw start
        self._finish(target.channel_id, now, stopped=False)

    def _on_stopped(self, now: float, target: Target) -> None:
        if target.channel_id in self.runs:
            self._finish(target.channel_id, now, stopped=True)

    def _on_api_start(self, now: float, target: Target, session_id: str, model: str, retry_count: int) -> None:
        if not self.cfg.show_model:
            return
        run = self._run_for(target, session_id, now)
        run.last_event = now
        run.dirty = True
        m = run.model
        if m is None:
            m = run.model = r.ModelEntry(model=model or "model")
        if retry_count:
            m.retries += 1
        else:
            m.calls += 1
        m.inflight_since = now
        m.status = r.PENDING

    def _on_api_end(self, now: float, target: Target, session_id: str, duration_s: float, usage: Dict[str, Any]) -> None:
        run = self.runs.get(target.channel_id)
        if run is None or run.model is None:
            return
        run.last_event = now
        run.dirty = True
        m = run.model
        m.total_ms += _duration_ms(duration_s, m.inflight_since, now)
        m.inflight_since = None
        m.error = None
        run.model_error_open = False
        if usage:
            m.prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or m.prompt_tokens or 0)
            m.output_tokens += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        m.status = r.OK

    def _on_api_error(self, now: float, target: Target, session_id: str, duration_s: float, error: str) -> None:
        run = self.runs.get(target.channel_id)
        if run is None or run.model is None:
            return
        run.last_event = now
        run.dirty = True
        m = run.model
        m.total_ms += _duration_ms(duration_s, m.inflight_since, now)
        m.inflight_since = None
        m.error = error or "request failed"
        run.model_error_open = True  # stays pending: Hermes usually retries

    def _on_tool_start(self, now: float, target: Target, session_id: str, call_id: str, name: str, args: Any) -> None:
        run = self._run_for(target, session_id, now)
        run.last_event = now
        run.dirty = True
        if run.model is not None and run.model.inflight_since is None and not run.model_error_open:
            run.model.status = r.OK
        if r.normalize_tool_name(name) in PLAN_TOOLS:
            return  # shown as the 📋 plan row instead of a tool row
        key = job_key(r.normalize_tool_name(name), args)
        if key:
            # polling the same job again: reuse its row so the bar updates in place
            for t in run.tools:
                if t.job_key == key and t.status != r.PENDING and t.level != r.REMOVED:
                    t.status, t.started, t.duration_ms, t.error = r.PENDING, now, None, None
                    t.call_id = call_id or t.call_id
                    t.repeats += 1
                    return
        run.tools.append(r.ToolEntry(
            call_id=call_id or f"anon-{len(run.tools)}",
            name=r.normalize_tool_name(name),
            icon=self.icon_for(name),
            started=now,
            args=r.summarize_args(args),
            job_key=key,
        ))

    def _on_tool_end(self, now: float, target: Target, session_id: str, call_id: str, name: str,
                     duration_ms: Optional[float], status: str, error: str, result: Any) -> None:
        run = self.runs.get(target.channel_id)
        if run is None:
            return
        run.last_event = now
        run.dirty = True
        if r.normalize_tool_name(name) in PLAN_TOOLS:
            plan = r.parse_todos(result)
            if plan is not None:
                run.plan = plan
            return
        entry = self._match(run, call_id, r.normalize_tool_name(name))
        if entry is None:
            return
        entry.duration_ms = float(duration_ms) if duration_ms else (now - entry.started) * 1000.0
        if status == "error" or error:
            entry.status = r.ERROR
            entry.error = error or r.summarize_result(result, self.cfg.render) or "error"
        else:
            entry.status = r.OK
            entry.result = r.summarize_result(result, self.cfg.render)
            progress = r.parse_progress(result)
            if progress is not None:
                entry.progress = progress

    @staticmethod
    def _match(run: Run, call_id: str, name: str) -> Optional[r.ToolEntry]:
        if call_id:
            for t in run.tools:
                if t.call_id == call_id:
                    return t
        for t in run.tools:  # no id: oldest pending call with that name
            if t.status == r.PENDING and t.name == name:
                return t
        return None

    # ── output ────────────────────────────────────────────────────────────────────────────────

    def _should_show(self, run: Run, now: float) -> bool:
        if run.dead:
            return False
        if run.message_id is not None or run.tools or (run.plan is not None and run.plan.items):
            return True
        return run.model is not None and now - run.started >= self.cfg.show_after_seconds

    @staticmethod
    def _has_running(run: Run) -> bool:
        if any(t.status == r.PENDING for t in run.tools):
            return True
        return run.model is not None and run.model.inflight_since is not None

    def _flush(self, run: Run, now: float) -> None:
        content = r.render(run.model, run.tools, now, self.cfg.render, run.plan)
        run.last_flush = now
        run.dirty = False
        if content == run.last_content:
            return
        try:
            if run.message_id is None:
                reply_to = run.target.reply_to if self.cfg.reply_to_message else ""
                run.message_id = self.client.create(run.target.channel_id, content, reply_to or None)
            else:
                self.client.edit(run.target.channel_id, run.message_id, content)
            run.last_content = content
        except PermanentError as exc:
            run.dead = True
            logger.warning("discord-activity: giving up on channel %s (%s)", run.target.channel_id, exc)
        except TransientError as exc:
            logger.info("discord-activity: update skipped (%s)", exc)

    def _finish(self, channel_id: str, now: float, stopped: bool) -> None:
        run = self.runs.pop(channel_id, None)
        if run is None:
            return
        for t in run.tools:
            if t.status == r.PENDING:
                t.status = r.STOPPED if stopped else r.OK
                t.duration_ms = (now - t.started) * 1000.0
        if run.model is not None:
            m = run.model
            if m.inflight_since is not None:
                m.total_ms += (now - m.inflight_since) * 1000.0
                m.inflight_since = None
            if run.model_error_open:
                m.status = r.ERROR
            elif m.status == r.PENDING:
                m.status = r.STOPPED if stopped else r.OK
        if run.message_id is None and not run.failed and not self._should_show(run, now):
            return  # quick turn, nothing was ever posted
        if run.message_id is None and not (stopped or run.failed or self.cfg.keep_after_done):
            return  # finished before the card appeared and nothing went wrong: stay quiet
        self._flush(run, now)
        keep = self.cfg.keep_after_done or stopped or run.failed
        if not keep and run.message_id and not run.dead:
            self.pending_deletes.append((now + self.cfg.delete_delay, channel_id, run.message_id, 0))


def _duration_ms(duration_s: Any, inflight_since: Optional[float], now: float) -> float:
    try:
        if duration_s is not None:
            return max(0.0, float(duration_s) * 1000.0)
    except (TypeError, ValueError):
        pass
    return (now - inflight_since) * 1000.0 if inflight_since is not None else 0.0
