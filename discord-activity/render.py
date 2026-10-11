"""Pure rendering: run state -> one Discord message (an ANSI code block).

Nothing here touches the network, threads or Hermes. Everything is deterministic given ``now``
so it can be unit-tested without a gateway.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── ANSI (Discord supports the 30–37 foreground range inside ```ansi blocks) ──────────────────────
RESET = "\u001b[0m"
GRAY = "\u001b[30m"
RED = "\u001b[31m"
GREEN = "\u001b[32m"
YELLOW = "\u001b[33m"
BLUE = "\u001b[34m"
PINK = "\u001b[35m"
CYAN = "\u001b[36m"
WHITE = "\u001b[37m"
BOLD_CYAN = "\u001b[1;36m"
ORANGE = "\u001b[33m"  # Discord has no orange; bold yellow reads as amber
BOLD_ORANGE = "\u001b[1;33m"

PENDING, OK, ERROR, STOPPED = "pending", "ok", "error", "stopped"
EXPANDED, COLLAPSED, REMOVED = 0, 1, 2

_STATUS_MARK = {
    PENDING: f"{YELLOW}←{RESET}",
    OK: f"{GREEN}✔{RESET}",
    ERROR: f"{RED}✘{RESET}",
    STOPPED: f"{GRAY}■{RESET}",
}

LABELS: Dict[str, Dict[str, str]] = {
    "en": {
        "more_chars": "(+{n} chars)",
        "earlier": "… {n} earlier",
        "calls": "calls",
        "retry": "retry",
        "tokens": "tokens",
        "result": "result",
        "error": "error",
        "items": "{n} items",
        "running": "running",
        "plan": "plan",
    },
    "zh-TW": {
        "more_chars": "(+{n} 字)",
        "earlier": "… 前面還有 {n} 個",
        "calls": "呼叫",
        "retry": "重試",
        "tokens": "tokens",
        "result": "result",
        "error": "error",
        "items": "{n} 筆",
        "running": "進行中",
        "plan": "計畫",
    },
}

_SECRET_KEY = re.compile(r"(token|secret|password|passwd|api[_-]?key|authorization|cookie|credential)", re.I)
_ANSI = re.compile(r"\u001b\[[0-9;]*m")


@dataclass
class RenderConfig:
    max_rows: int = 6
    max_length: int = 1700
    field_width: int = 70
    max_arg_lines: int = 2
    locale: str = "en"

    def label(self, key: str, **kw: Any) -> str:
        table = LABELS.get(self.locale) or LABELS["en"]
        return (table.get(key) or LABELS["en"][key]).format(**kw)


@dataclass
class ToolEntry:
    call_id: str
    name: str
    icon: str
    started: float
    args: List[Tuple[str, str]] = field(default_factory=list)
    status: str = PENDING
    duration_ms: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None
    level: int = EXPANDED  # only ever increases: expanded -> collapsed -> removed
    progress: Optional["Progress"] = None  # only when the tool reported real numbers
    job_key: Optional[str] = None          # same tool + same job id -> one row, updated in place
    repeats: int = 1

    def elapsed_ms(self, now: float) -> float:
        if self.duration_ms is not None:
            return self.duration_ms
        return max(0.0, (now - self.started) * 1000.0)


@dataclass
class Progress:
    """Real progress reported by a tool. ``done``/``total`` when known, else ``fraction`` only."""

    fraction: float
    done: Optional[float] = None
    total: Optional[float] = None
    status: Optional[str] = None


@dataclass
class PlanEntry:
    """The agent's own todo list (Hermes ``todo_list``): the only honest whole-task progress."""

    items: List[Tuple[str, str]] = field(default_factory=list)  # (content, status)

    def counts(self) -> Tuple[int, int]:
        live = [st for _, st in self.items if st != "cancelled"]
        return sum(1 for st in live if st == "completed"), len(live)

    def current(self) -> Optional[str]:
        for content, st in self.items:
            if st == "in_progress":
                return content
        for content, st in self.items:
            if st == "pending":
                return content
        return None


@dataclass
class ModelEntry:
    """Every LLM request of the turn folded into one row."""

    model: str
    calls: int = 0
    retries: int = 0
    total_ms: float = 0.0
    inflight_since: Optional[float] = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    status: str = PENDING
    error: Optional[str] = None
    level: int = EXPANDED

    def elapsed_ms(self, now: float) -> float:
        extra = (now - self.inflight_since) * 1000.0 if self.inflight_since is not None else 0.0
        return self.total_ms + max(0.0, extra)


# ── field helpers ─────────────────────────────────────────────────────────────────────────────

def visible_len(text: str) -> int:
    return len(_ANSI.sub("", text))


def clip(text: str, width: int, cfg: RenderConfig) -> str:
    """One line, at most ``width`` code points, with a gray ``(+N chars)`` hint when cut."""
    flat = " ".join(str(text).split())
    if len(flat) <= width:
        return flat
    return f"{flat[:width]}{GRAY}{cfg.label('more_chars', n=len(flat) - width)}{RESET}"


def format_duration(ms: float) -> str:
    if ms < 1000:
        return f"{int(round(ms))}ms"
    sec = ms / 1000.0
    return f"{sec:.2f}s" if sec < 10 else f"{sec:.1f}s"


def normalize_tool_name(name: str) -> str:
    """``mcp__docs__search_pages`` -> ``docs.search_pages``."""
    if name.startswith("mcp__"):
        parts = name[len("mcp__"):].split("__", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            return f"{parts[0]}.{parts[1]}"
    return name


def mcp_server_of(name: str) -> Optional[str]:
    if name.startswith("mcp__"):
        server = name[len("mcp__"):].split("__", 1)[0]
        return server or None
    return None


def _scalar(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    except Exception:
        return str(value)


def summarize_args(args: Any) -> List[Tuple[str, str]]:
    """Flat ``(key, value)`` pairs, secrets masked, empties dropped."""
    if not isinstance(args, dict):
        return []
    out: List[Tuple[str, str]] = []
    for key, value in args.items():
        if value is None or value == "" or value == [] or value == {}:
            continue
        key = str(key)
        out.append((key, "***" if _SECRET_KEY.search(key) else _scalar(value)))
    return out


_LIST_KEYS = ("items", "results", "data", "rows", "packages", "orders", "entries", "records")


def summarize_result(result: Any, cfg: RenderConfig) -> Optional[str]:
    """Short human summary of a tool result (a count when it is a list, else the first text)."""
    if result is None:
        return None
    parsed: Any = result
    if isinstance(result, str):
        text = result.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            return text
    # MCP / Hermes envelopes: {"result": "..."} or {"content": [{"type": "text", "text": "..."}]}
    for _ in range(3):
        if isinstance(parsed, dict) and len(parsed) <= 3:
            inner = parsed.get("result", parsed.get("content"))
            if isinstance(inner, list) and inner and isinstance(inner[0], dict) and "text" in inner[0]:
                inner = inner[0]["text"]
            if isinstance(inner, str):
                try:
                    parsed = json.loads(inner)
                    continue
                except Exception:
                    return inner
            if isinstance(inner, (dict, list)):
                parsed = inner
                continue
        break
    if isinstance(parsed, list):
        return cfg.label("items", n=len(parsed))
    if isinstance(parsed, dict):
        for key in _LIST_KEYS:
            if isinstance(parsed.get(key), list):
                return cfg.label("items", n=len(parsed[key]))
        for key in ("message", "summary", "status", "text", "output"):
            if isinstance(parsed.get(key), str) and parsed[key].strip():
                return parsed[key]
        return _scalar(parsed)
    return _scalar(parsed)


_PROGRESS_KEYS = ("progress", "percent", "percentage", "pct", "progress_percent", "progressPercent")
_DONE_KEYS = ("current", "step", "done", "completed", "processed", "current_step", "currentStep")
_TOTAL_KEYS = ("total", "total_steps", "totalSteps", "steps", "count")
_STATUS_KEYS = ("status", "state", "phase", "stage")
_FRACTION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$")
_PERCENT = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%\s*$")


def _unwrap(result: Any) -> Any:
    """JSON-decode a tool result and peel MCP / Hermes envelopes (same rules as summarize_result)."""
    parsed: Any = result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except Exception:
            return None
    for _ in range(3):
        if isinstance(parsed, dict) and len(parsed) <= 3:
            inner = parsed.get("result", parsed.get("content"))
            if isinstance(inner, list) and inner and isinstance(inner[0], dict) and "text" in inner[0]:
                inner = inner[0]["text"]
            if isinstance(inner, str):
                try:
                    parsed = json.loads(inner)
                    continue
                except Exception:
                    return parsed
            if isinstance(inner, (dict, list)):
                parsed = inner
                continue
        break
    return parsed


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _progress_from_value(value: Any, key: str) -> Optional[Progress]:
    if isinstance(value, str):
        m = _FRACTION.match(value)
        if m:
            done, total = float(m.group(1)), float(m.group(2))
            return Progress(fraction=done / total, done=done, total=total) if total > 0 and done <= total else None
        m = _PERCENT.match(value)
        if m:
            return Progress(fraction=float(m.group(1)) / 100.0)
        return None
    n = _num(value)
    if n is None or n < 0:
        return None
    if key == "progress" and n <= 1.0:
        return Progress(fraction=n)            # 0–1 fraction (0.45)
    if n <= 100.0:
        return Progress(fraction=n / 100.0)    # 0–100 percent (45)
    return None


def parse_progress(result: Any) -> Optional[Progress]:
    """Progress a tool actually reported, or None. Never guesses: no numbers -> no bar."""
    data = _unwrap(result)
    candidates = [data]
    if isinstance(data, dict):
        candidates += [v for k, v in data.items() if k in ("job", "data", "task", "generation", "run") and isinstance(v, dict)]
    for obj in candidates:
        if not isinstance(obj, dict):
            continue
        status = next((str(obj[k]) for k in _STATUS_KEYS if isinstance(obj.get(k), str) and obj[k].strip()), None)
        found: Optional[Progress] = None
        for key in _PROGRESS_KEYS:
            if key in obj:
                found = _progress_from_value(obj[key], key)
                if found:
                    break
        if found is None:
            done = next((_num(obj[k]) for k in _DONE_KEYS if _num(obj.get(k)) is not None), None)
            total = next((_num(obj[k]) for k in _TOTAL_KEYS if _num(obj.get(k)) is not None), None)
            if done is not None and total and 0 <= done <= total:
                found = Progress(fraction=done / total, done=done, total=total)
        if found is not None:
            found.fraction = max(0.0, min(1.0, found.fraction))
            found.status = status
            return found
    return None


def parse_todos(result: Any) -> Optional[PlanEntry]:
    """Hermes ``todo_list`` result -> PlanEntry; None when it is not a todo payload."""
    data = _unwrap(result)
    todos = data.get("todos") if isinstance(data, dict) else None
    if not isinstance(todos, list):
        return None
    items = []
    for t in todos:
        if isinstance(t, dict) and str(t.get("content") or "").strip():
            items.append((str(t["content"]), str(t.get("status") or "pending").lower()))
    return PlanEntry(items=items) if items else None


def _fmt_num(n: Optional[float]) -> str:
    if n is None:
        return ""
    return str(int(n)) if float(n).is_integer() else f"{n:g}"


def progress_bar(p: Progress, width: int = 10) -> str:
    """``▕██████░░░░▏ 3/5  60%`` — filled cells green, empty cells gray."""
    filled = int(round(p.fraction * width))
    filled = min(width, max(0, filled))
    bar = f"{GRAY}▕{RESET}{GREEN}{'█' * filled}{RESET}{GRAY}{'░' * (width - filled)}▏{RESET}"
    count = f" {WHITE}{_fmt_num(p.done)}/{_fmt_num(p.total)}{RESET}" if p.done is not None and p.total else ""
    return f"{bar}{count}  {BOLD_ORANGE}{int(round(p.fraction * 100))}%{RESET}"


# ── layout ────────────────────────────────────────────────────────────────────────────────────

def _pack_args(args: List[Tuple[str, str]], cfg: RenderConfig) -> List[str]:
    """``key: value · key: value`` lines; short pairs share a line, long ones get their own."""
    lines: List[str] = []
    current: List[str] = []
    current_len = 0
    for key, value in args:
        piece_plain_len = len(key) + 2 + len(" ".join(value.split()))
        piece = f"{PINK}{key}{RESET}: {WHITE}{clip(value, max(10, cfg.field_width - len(key) - 2), cfg)}{RESET}"
        if current and current_len + 3 + piece_plain_len > cfg.field_width:
            lines.append(f" {GRAY}·{RESET} ".join(current))
            current, current_len = [], 0
        current.append(piece)
        current_len += piece_plain_len + (3 if current_len else 0)
        if len(lines) >= cfg.max_arg_lines:
            break
    if current and len(lines) < cfg.max_arg_lines:
        lines.append(f" {GRAY}·{RESET} ".join(current))
    return lines[: cfg.max_arg_lines]


def _tree(children: List[str]) -> List[str]:
    out = []
    for i, child in enumerate(children):
        branch = "└─" if i == len(children) - 1 else "├─"
        out.append(f"   {GRAY}{branch}{RESET} {child}")
    return out


def _header(icon: str, name: str, level: int, status: str, ms: float, has_children: bool) -> str:
    disclosure = "▾" if level == EXPANDED and has_children else "▸" if has_children else " "
    return (f"{icon} {BOLD_CYAN}{name}{RESET} {GRAY}{disclosure}{RESET} {_STATUS_MARK.get(status, '')} "
            f"{ORANGE}[{format_duration(ms)}]{RESET}")


def _model_block(m: ModelEntry, now: float, cfg: RenderConfig) -> List[str]:
    children: List[str] = []
    if m.calls > 1:
        children.append(f"{PINK}{cfg.label('calls')}{RESET}: {WHITE}{m.calls}{RESET}")
    if m.retries:
        children.append(f"{PINK}{cfg.label('retry')}{RESET}: {YELLOW}{m.retries}{RESET}")
    if m.prompt_tokens or m.output_tokens:
        children.append(f"{PINK}{cfg.label('tokens')}{RESET}: {WHITE}{m.prompt_tokens:,} → {m.output_tokens:,}{RESET}")
    if m.error:
        children.append(f"{RED}{cfg.label('error')}{RESET}: {clip(m.error, cfg.field_width, cfg)}")
    if children:
        # the calls/retry/tokens facts are short: keep them on one line
        facts = [c for c in children if not c.startswith(RED)]
        rest = [c for c in children if c.startswith(RED)]
        children = ([f" {GRAY}·{RESET} ".join(facts)] if facts else []) + rest
    lines = [_header("🧠", m.model or "model", m.level, m.status, m.elapsed_ms(now), bool(children))]
    if m.level == EXPANDED:
        lines += _tree(children)
    return lines


def _tool_block(t: ToolEntry, now: float, cfg: RenderConfig) -> List[str]:
    children = _pack_args(t.args, cfg)
    if t.progress is not None:
        status = f" {GRAY}·{RESET} {WHITE}{clip(t.progress.status, 20, cfg)}{RESET}" if t.progress.status else ""
        children.append(progress_bar(t.progress) + status)
    if t.error:
        children.append(f"{RED}{cfg.label('error')}{RESET}: {clip(t.error, cfg.field_width, cfg)}")
    elif t.result and t.progress is None:
        children.append(f"{PINK}{cfg.label('result')}{RESET}: {GREEN}{clip(t.result, cfg.field_width, cfg)}{RESET}")
    lines = [_header(t.icon, t.name, t.level, t.status, t.elapsed_ms(now), bool(children))]
    if t.repeats > 1:
        lines[0] += f" {GRAY}×{t.repeats}{RESET}"
    if t.level == EXPANDED:
        lines += _tree(children)
    return lines


def _plan_block(plan: PlanEntry, cfg: RenderConfig) -> str:
    done, total = plan.counts()
    frac = done / total if total else 1.0
    line = f"📋 {BOLD_CYAN}{cfg.label('plan')}{RESET} " + progress_bar(Progress(fraction=frac, done=done, total=total))
    current = plan.current()
    if current and done < total:
        line += f"\n   {GRAY}└─{RESET} {YELLOW}▶{RESET} {clip(current, cfg.field_width, cfg)}"
    return line


def _compose(model: Optional[ModelEntry], tools: List[ToolEntry], removed: int, now: float, cfg: RenderConfig,
             plan: Optional[PlanEntry] = None) -> str:
    blocks: List[str] = []
    if plan is not None and plan.items:
        blocks.append(_plan_block(plan, cfg))  # protected: never collapsed or removed
    if removed:
        blocks.append(f"{GRAY}{cfg.label('earlier', n=removed)}{RESET}")
    if model is not None and model.level != REMOVED:
        blocks.append("\n".join(_model_block(model, now, cfg)))
    for t in tools:
        if t.level != REMOVED:
            blocks.append("\n".join(_tool_block(t, now, cfg)))
    return "```ansi\n" + "\n\n".join(blocks) + "\n```"


def render(model: Optional[ModelEntry], tools: List[ToolEntry], now: float, cfg: RenderConfig,
           plan: Optional[PlanEntry] = None) -> str:
    """Lay out the card within ``max_rows`` outer rows and ``max_length`` characters.

    Mutates each entry's ``level`` — but only upwards (expanded -> collapsed -> removed), so a later
    render never shows something an earlier one already folded away.
    """
    # 1. slot budget: the plan and model rows take one slot each; drop the oldest finished tools first
    slots = cfg.max_rows - (1 if model is not None else 0) - (1 if plan is not None and plan.items else 0)
    visible = [t for t in tools if t.level != REMOVED]
    for t in visible:
        if len([v for v in tools if v.level != REMOVED]) <= slots:
            break
        if t.status != PENDING:
            t.level = REMOVED
    # still over budget with only pending tools left: remove oldest anyway
    for t in tools:
        if len([v for v in tools if v.level != REMOVED]) <= slots:
            break
        t.level = max(t.level, REMOVED)

    def removed_count() -> int:
        return sum(1 for t in tools if t.level == REMOVED)

    text = _compose(model, tools, removed_count(), now, cfg, plan)
    # 2. length budget: collapse oldest finished, then oldest at all, then remove oldest finished
    candidates = [t for t in tools if t.level == EXPANDED and t.status != PENDING]
    candidates += [t for t in tools if t.level == EXPANDED and t.status == PENDING]
    if model is not None and model.level == EXPANDED:
        candidates.append(model)  # type: ignore[arg-type]
    for entry in candidates:
        if len(text) <= cfg.max_length:
            break
        entry.level = COLLAPSED
        text = _compose(model, tools, removed_count(), now, cfg, plan)
    for t in tools:
        if len(text) <= cfg.max_length:
            break
        if t.level != REMOVED and t.status != PENDING:
            t.level = REMOVED
            text = _compose(model, tools, removed_count(), now, cfg, plan)
    if len(text) > cfg.max_length:  # last resort, should not happen with sane limits
        text = text[: cfg.max_length - 4].rstrip() + "\n```"
    return text
