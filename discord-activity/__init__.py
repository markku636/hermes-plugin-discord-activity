"""discord-activity — live tool/model status card for Hermes Agent on Discord.

One message per Discord conversation, edited in place while the agent works: each tool gets a row
with its status (← running / ✔ done / ✘ failed), duration, arguments and a short result, plus one
🧠 row for the model calls of the turn.

Hooks only enqueue an event and return (``pre_tool_call`` is fail-closed on timeout in Hermes, so a
slow hook would block the tool). A daemon thread does the rendering and the Discord REST calls.
Everything fails open: errors are logged and the agent never notices.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional

from . import render as r
from .discord_api import DiscordClient
from .tracker import Target, Tracker, TrackerConfig

logger = logging.getLogger(__name__)

_tracker: Optional[Tracker] = None
_sessions: "OrderedDict[str, Target]" = OrderedDict()  # session_id / session_key -> target
_sessions_lock = threading.Lock()
_MAX_SESSIONS = 512


# ── config ────────────────────────────────────────────────────────────────────────────────────

def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        return min(hi, max(lo, float(_env(name) or default)))
    except ValueError:
        return default


def _env_map(name: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in _env(name).split(","):
        key, sep, value = part.partition("=")
        if sep and key.strip() and value.strip():
            out[key.strip()] = value.strip()
    return out


def load_config() -> TrackerConfig:
    return TrackerConfig(
        keep_after_done=_env_bool("DISCORD_ACTIVITY_KEEP_AFTER_DONE", False),
        show_model=_env_bool("DISCORD_ACTIVITY_SHOW_MODEL", True),
        show_after_seconds=_env_float("DISCORD_ACTIVITY_SHOW_AFTER_SECONDS", 3.0, 0.0, 120.0),
        edit_interval=_env_float("DISCORD_ACTIVITY_EDIT_INTERVAL", 1.2, 0.5, 30.0),
        delete_delay=_env_float("DISCORD_ACTIVITY_DELETE_DELAY", 2.0, 0.0, 60.0),
        max_display_seconds=_env_float("DISCORD_ACTIVITY_MAX_DISPLAY_SECONDS", 600.0, 30.0, 86400.0),
        reply_to_message=_env_bool("DISCORD_ACTIVITY_REPLY", True),
        render=r.RenderConfig(
            max_rows=int(_env_float("DISCORD_ACTIVITY_MAX_ROWS", 6, 2, 20)),
            max_length=int(_env_float("DISCORD_ACTIVITY_MAX_LENGTH", 1700, 300, 1900)),
            locale=_env("DISCORD_ACTIVITY_LOCALE", "en") or "en",
        ),
    )


# ── icons ─────────────────────────────────────────────────────────────────────────────────────

def _make_icon_for() -> Any:
    server_icons = _env_map("DISCORD_ACTIVITY_SERVER_ICONS")  # e.g. "github=🐙,search=🔍"

    def icon_for(tool_name: str) -> str:
        server = r.mcp_server_of(tool_name)
        if server:
            return server_icons.get(server, "🔌")
        try:
            from agent.display import get_tool_emoji  # Hermes' own per-tool emoji
            return get_tool_emoji(tool_name, default="⚙️")
        except Exception:
            return "⚙️"

    return icon_for


# ── which Discord conversation is this? ───────────────────────────────────────────────────────

def _session_var(name: str) -> str:
    try:
        from gateway.session_context import get_session_env
    except Exception:
        return ""
    try:
        return str(get_session_env(name, "") or "")
    except Exception:
        return ""


def _context_target() -> Optional[Target]:
    """The Discord channel the gateway bound to this task (a thread's chat id is the thread)."""
    if _session_var("HERMES_SESSION_PLATFORM").lower() != "discord":
        return None
    channel = _session_var("HERMES_SESSION_CHAT_ID") or _session_var("HERMES_SESSION_THREAD_ID")
    if not channel:
        return None
    return Target(channel_id=channel, reply_to=_session_var("HERMES_SESSION_MESSAGE_ID"))


def _remember(key: str, target: Target) -> None:
    if not key:
        return
    with _sessions_lock:
        _sessions[key] = target
        _sessions.move_to_end(key)
        while len(_sessions) > _MAX_SESSIONS:
            _sessions.popitem(last=False)


def _lookup(key: str) -> Optional[Target]:
    if not key:
        return None
    with _sessions_lock:
        return _sessions.get(key)


def _target_for(session_id: str, platform: str = "") -> Optional[Target]:
    if platform and platform.lower() != "discord":
        return None
    # contextvars first; tool calls running on a worker thread may not carry them, so fall back to
    # the session the turn was started from
    return _context_target() or _lookup(session_id)


# ── hooks (must stay cheap: enqueue and return) ───────────────────────────────────────────────

_HIDDEN_TOOLS = set()


def on_pre_llm_call(session_id: str = "", platform: str = "", parent_session_id: str = "", **_: Any) -> None:
    if _tracker is None or parent_session_id:  # subagents report into their parent's card
        return None
    if platform and platform.lower() != "discord":
        return None
    target = _context_target()
    if target is None:
        return None
    _remember(session_id, target)
    _remember(_session_var("HERMES_SESSION_KEY"), target)
    _tracker.submit("turn_start", target, session_id)
    return None


def on_post_llm_call(session_id: str = "", platform: str = "", **_: Any) -> None:
    if _tracker is None:
        return None
    target = _lookup(session_id)
    if target is not None:
        _tracker.submit("turn_end", target, session_id)
    return None


def on_agent_loop_stopped(session_key: str = "", platform: str = "", **_: Any) -> None:
    if _tracker is None or (platform and platform.lower() != "discord"):
        return None
    target = _lookup(session_key)
    if target is not None:
        _tracker.submit("stopped", target)
    return None


def on_pre_api_request(session_id: str = "", platform: str = "", model: str = "", retry_count: int = 0, **_: Any) -> None:
    if _tracker is None:
        return None
    target = _target_for(session_id, platform)
    if target is not None:
        _tracker.submit("api_start", target, session_id, str(model or ""), int(retry_count or 0))
    return None


def on_post_api_request(session_id: str = "", platform: str = "", api_duration: Any = None,
                        usage: Optional[Dict[str, Any]] = None, **_: Any) -> None:
    if _tracker is None:
        return None
    target = _target_for(session_id, platform)
    if target is not None:
        _tracker.submit("api_end", target, session_id, api_duration, dict(usage or {}))
    return None


def on_api_request_error(session_id: str = "", platform: str = "", api_duration: Any = None,
                         error: Any = None, status_code: Any = None, **_: Any) -> None:
    if _tracker is None:
        return None
    target = _target_for(session_id, platform)
    if target is None:
        return None
    message = ""
    if isinstance(error, dict):
        message = str(error.get("message") or error.get("type") or "")
    elif error:
        message = str(error)
    if status_code:
        message = f"{status_code} {message}".strip()
    _tracker.submit("api_error", target, session_id, api_duration, message)
    return None


def on_pre_tool_call(tool_name: str = "", args: Any = None, session_id: str = "", tool_call_id: str = "", **_: Any) -> None:
    if _tracker is None or not tool_name or tool_name in _HIDDEN_TOOLS:
        return None
    target = _target_for(session_id)
    if target is not None:
        _tracker.submit("tool_start", target, session_id, tool_call_id or "", tool_name,
                        dict(args) if isinstance(args, dict) else None)
    return None


def on_post_tool_call(tool_name: str = "", result: Any = None, session_id: str = "", tool_call_id: str = "",
                      duration_ms: Any = None, status: str = "", error_message: str = "", **_: Any) -> None:
    if _tracker is None or not tool_name or tool_name in _HIDDEN_TOOLS:
        return None
    target = _target_for(session_id)
    if target is not None:
        # results can be huge: keep only what the summary can possibly use
        if isinstance(result, str) and len(result) > 20000:
            result = result[:20000]
        _tracker.submit("tool_end", target, session_id, tool_call_id or "", tool_name,
                        duration_ms, str(status or ""), str(error_message or ""), result)
    return None


_HOOKS = {
    "pre_llm_call": on_pre_llm_call,
    "post_llm_call": on_post_llm_call,
    "agent_loop_stopped": on_agent_loop_stopped,
    "pre_api_request": on_pre_api_request,
    "post_api_request": on_post_api_request,
    "api_request_error": on_api_request_error,
    "pre_tool_call": on_pre_tool_call,
    "post_tool_call": on_post_tool_call,
}


def register(ctx: Any) -> None:
    global _tracker
    if not _env_bool("DISCORD_ACTIVITY_ENABLED", True):
        logger.info("discord-activity: disabled by DISCORD_ACTIVITY_ENABLED")
        return
    token = _env("DISCORD_BOT_TOKEN")
    if not token:
        logger.warning("discord-activity: DISCORD_BOT_TOKEN is not set; plugin inactive")
        return
    _HIDDEN_TOOLS.clear()
    _HIDDEN_TOOLS.update(t.strip() for t in _env("DISCORD_ACTIVITY_HIDE_TOOLS", "clarify").split(",") if t.strip())
    cfg = load_config()
    client = DiscordClient(token, api_base=_env("DISCORD_ACTIVITY_API_BASE", "https://discord.com/api/v10"))
    _tracker = Tracker(client, cfg, icon_for=_make_icon_for())
    _tracker.start()
    for name, callback in _HOOKS.items():
        ctx.register_hook(name, callback)
    logger.info("discord-activity: active (keep_after_done=%s, show_model=%s, locale=%s)",
                cfg.keep_after_done, cfg.show_model, cfg.render.locale)
