"""Minimal Discord REST client (standard library only).

Only three calls are needed: create, edit and delete a message. Retries are bounded; anything
that will not get better by retrying (missing token, 401/403/404) raises ``PermanentError`` so
the caller can stop touching that message.
"""

from __future__ import annotations

import json
import logging
import random
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

USER_AGENT = "DiscordBot (https://github.com/markku636/hermes-plugin-discord-activity, 0.1.0)"
SUPPRESS_EMBEDS = 1 << 2


class PermanentError(Exception):
    """The request will not succeed by retrying (auth, permissions, message gone)."""


class TransientError(Exception):
    """Retries ran out (rate limit, 5xx, network)."""


class DiscordClient:
    def __init__(
        self,
        token: str,
        api_base: str = "https://discord.com/api/v10",
        timeout: float = 10.0,
        max_attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._token = token
        self._base = api_base.rstrip("/")
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._sleep = sleep

    # ── public ────────────────────────────────────────────────────────────────────────────────

    def create(self, channel_id: str, content: str, reply_to: Optional[str] = None) -> str:
        body: Dict[str, Any] = {
            "content": content,
            "flags": SUPPRESS_EMBEDS,
            "allowed_mentions": {"parse": [], "replied_user": False},
            # Same nonce on every retry + enforce_nonce: a retried create cannot post twice.
            "nonce": str(random.randint(10**17, 10**18 - 1)),
            "enforce_nonce": True,
        }
        if reply_to:
            body["message_reference"] = {"message_id": reply_to, "channel_id": channel_id, "fail_if_not_exists": False}
        data = self._request("POST", f"/channels/{channel_id}/messages", body)
        message_id = (data or {}).get("id")
        if not message_id:
            raise TransientError("create returned no message id")
        return str(message_id)

    def edit(self, channel_id: str, message_id: str, content: str) -> None:
        self._request("PATCH", f"/channels/{channel_id}/messages/{message_id}",
                      {"content": content, "flags": SUPPRESS_EMBEDS})

    def delete(self, channel_id: str, message_id: str) -> None:
        try:
            self._request("DELETE", f"/channels/{channel_id}/messages/{message_id}", None)
        except PermanentError as exc:
            if "404" in str(exc):
                return  # already gone
            raise

    # ── transport ─────────────────────────────────────────────────────────────────────────────

    def _request(self, method: str, path: str, body: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not self._token:
            raise PermanentError("no bot token")
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        last_error = ""
        for attempt in range(self._max_attempts):
            req = urllib.request.Request(self._base + path, data=payload, method=method)
            req.add_header("Authorization", f"Bot {self._token}")
            req.add_header("User-Agent", USER_AGENT)
            if payload is not None:
                req.add_header("Content-Type", "application/json")
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    raw = resp.read()
                    return json.loads(raw) if raw else None
            except urllib.error.HTTPError as exc:
                status = exc.code
                detail = _read_body(exc)
                if status == 429:
                    retry_after = _retry_after(exc, detail)
                    last_error = f"429 retry_after={retry_after}"
                    self._sleep(min(retry_after, 5.0))
                    continue
                if status in (401, 403, 404) or 400 <= status < 500:
                    raise PermanentError(f"{status} {method} {path}: {detail[:200]}") from None
                last_error = f"{status}"
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = f"network: {exc}"
            self._sleep(0.5 * (2 ** attempt))
        raise TransientError(f"{method} {path} failed after {self._max_attempts} attempts ({last_error})")


def _read_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", "replace")
    except Exception:
        return ""


def _retry_after(exc: urllib.error.HTTPError, body: str) -> float:
    try:
        return max(0.0, float(json.loads(body).get("retry_after", 1.0)))
    except Exception:
        pass
    try:
        return max(0.0, float(exc.headers.get("Retry-After", "1")))
    except Exception:
        return 1.0
