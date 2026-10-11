# hermes-plugin-discord-activity

Live tool and model status for [Hermes Agent](https://github.com/NousResearch/hermes-agent) on Discord.

While the agent works, one message per conversation is edited in place: every tool call gets a row
with its status, duration, arguments and a short result, and the model calls of the turn are folded
into one 🧠 row.

```
🧠 qwen3.6-35b ▾ ✔ [1.30s]
   └─ tokens: 9,000 → 40

📚 docs.search_pages ▾ ✔ [1.15s]
   ├─ query: plugin hooks · limit: 5
   └─ result: 3 items

🌐 web_fetch ▾ ← [0.84s]
   └─ url: https://github.com/NousResearch/hermes-agent
```

(In Discord this is an `ansi` code block, so names, markers and durations are colored.)

- `←` running, `✔` done, `✘` failed, `■` stopped
- `▾` expanded, `▸` collapsed to fit
- Each field is capped at 70 characters with a gray `(+N chars)` hint
- Up to 6 rows and 1,700 characters; the oldest finished rows collapse, then drop off
  (`… N earlier`). Rows only ever go expanded → collapsed → removed, so the card never jumps back.
- Arguments whose name looks like a secret (`token`, `api_key`, `password`, `authorization`, …) are shown as `***`

After a successful turn the card is deleted (configurable); after a failure or `/stop` it stays as a
breadcrumb.

## How it works

It is a regular Hermes user plugin. Hooks used: `pre_llm_call`, `post_llm_call`,
`pre_api_request`, `post_api_request`, `api_request_error`, `pre_tool_call`, `post_tool_call`,
`agent_loop_stopped`.

- The Discord conversation comes from the gateway's session context (`HERMES_SESSION_PLATFORM`,
  `HERMES_SESSION_CHAT_ID`, `HERMES_SESSION_MESSAGE_ID`). Turns on other platforms are ignored.
- Hooks only put an event on a queue and return. In Hermes `pre_tool_call` is fail-closed on
  timeout, so the plugin never does network I/O inside a hook.
- A daemon thread renders the card and calls the Discord REST API with the bot token Hermes already
  uses (`DISCORD_BOT_TOKEN`). Edits are coalesced (at most one per 1.2 s; running timers refresh
  every 5 s). Creates use `enforce_nonce` so a retry cannot post twice. 429 and 5xx are retried a
  few times; 401/403/404 stop updates for that card.
- Everything fails open: errors are logged, the agent never sees them.
- Short chat turns stay quiet: the card appears when the first tool starts, or when a model call
  takes longer than 3 s.
- Subagents (`delegate_task`) report into the parent's card.

Standard library only, no extra dependencies. Tested against Hermes Agent v0.21.6.

## Install

```sh
git clone https://github.com/markku636/hermes-plugin-discord-activity
cp -r hermes-plugin-discord-activity/discord-activity "$HERMES_HOME/plugins/"   # usually ~/.hermes/plugins/
```

Enable it in `config.yaml`:

```yaml
plugins:
  enabled:
    - discord-activity

# optional: turn off Hermes' built-in one-line progress bubble on Discord so you don't get both
display:
  platforms:
    discord:
      tool_progress: "off"
```

Restart the gateway. The log should show `discord-activity: active (...)`.

### Kubernetes / Helm

Ship the three `.py` files and `plugin.yaml` in a ConfigMap and copy them to
`$HERMES_HOME/plugins/discord-activity/` from an init container (copying is safer than mounting
over a path the Hermes image `chown`s at start-up).

## Configuration (environment variables)

| Variable | Default | |
| --- | --- | --- |
| `DISCORD_BOT_TOKEN` | — | Required; the same token the Hermes Discord adapter uses |
| `DISCORD_ACTIVITY_ENABLED` | `true` | |
| `DISCORD_ACTIVITY_KEEP_AFTER_DONE` | `false` | `true` keeps the card after successful turns too |
| `DISCORD_ACTIVITY_SHOW_MODEL` | `true` | The 🧠 row (calls, retries, tokens) |
| `DISCORD_ACTIVITY_SHOW_AFTER_SECONDS` | `3` | Show the card for a slow model call even without tools |
| `DISCORD_ACTIVITY_EDIT_INTERVAL` | `1.2` | Seconds between edits |
| `DISCORD_ACTIVITY_DELETE_DELAY` | `2` | Seconds after the turn before the card is deleted |
| `DISCORD_ACTIVITY_MAX_DISPLAY_SECONDS` | `600` | Close a card whose turn never reported an end |
| `DISCORD_ACTIVITY_MAX_ROWS` | `6` | |
| `DISCORD_ACTIVITY_MAX_LENGTH` | `1700` | 300–1900 |
| `DISCORD_ACTIVITY_REPLY` | `true` | Post the card as a reply to the user's message |
| `DISCORD_ACTIVITY_LOCALE` | `en` | `en` or `zh-TW` |
| `DISCORD_ACTIVITY_SERVER_ICONS` | — | Icons per MCP server, e.g. `github=🐙,search=🔍` (default 🔌) |
| `DISCORD_ACTIVITY_HIDE_TOOLS` | `clarify` | Comma-separated tool names not to show |
| `DISCORD_ACTIVITY_API_BASE` | `https://discord.com/api/v10` | For tests |

The bot needs **Send Messages** and **Read Message History** in the channel (and **Manage Messages**
is not needed: it only edits and deletes its own messages).

## Development

```sh
cd tests && python3 -m unittest -v
```

`render.py` is pure (no I/O, deterministic for a given `now`); `tracker.py` takes a fake client and
clock; `discord_api.py` is tested against a local HTTP server.

## Credits

The card layout follows [openclaw-plugin-discord-activity](https://github.com/ani6439walc/openclaw-plugin-discord-activity),
which does the same for OpenClaw. This is an independent implementation for Hermes' hook system.

## License

MIT
