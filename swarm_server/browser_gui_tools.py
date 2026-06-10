"""GUI-grade browser tools — the rungs Hermes' browser toolset is missing.

Hermes drives the team Chrome through the ``agent-browser`` CLI, which already
supports full GUI control (coordinate mouse, real keystrokes, hover, drag,
upload, screenshots) — but Hermes exposes only ~10 of its ~40 commands as
tools. The result, observed live: an agent signs into Medium fine (forms have
accessibility-tree refs) and then dies on the editor (contenteditable —
``browser_type``'s clear+fill never fires real key events), with no screenshot
to even see why.

This module registers the missing rungs as swarm-side tools riding the SAME
session plumbing as the built-in browser tools (``_run_browser_command`` keys
sessions by Hermes ``task_id``, which the daemon pins to ``agent_name:<name>``)
— no Hermes fork, same tab, same cookies:

  browser_keys            real keystrokes into the focused element
  browser_hover           hover (menus, tooltips)
  browser_dblclick        double-click
  browser_drag            drag & drop between two elements
  browser_upload          file upload into an <input type=file>
  browser_scrollintoview  bring an element into the viewport
  browser_wait            wait for a selector or a fixed delay
  browser_click_xy        coordinate click — works where no ref exists
  browser_screenshot      PNG into the agent's workspace (verifiable artifact)
  browser_locate          vision model grounds "describe the control" → x,y

Every action returns a post-action URL+title breadcrumb so the agent (and the
supervisor reading its transcript) can tell whether the page actually moved.

Exposure policy (enforced in agent.py): never for supervisors, and only when
the Hermes browser toolset itself is active for the agent.
"""

import base64
import json
import logging
import re
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("swarm.guitools")

# Vision grounding call ceiling — a screenshot upload + VLM answer can be slow.
VISION_LOCATE_TIMEOUT_SECONDS = 90


# ---------------------------------------------------------------------------
# Plumbing — ride Hermes' own session-scoped CLI dispatcher
# ---------------------------------------------------------------------------

def _ab(task_id: str, command: str, args: Optional[List[str]] = None,
        timeout: Optional[int] = None) -> Dict[str, Any]:
    """Run one agent-browser CLI command in the CALLER's browser session.

    Imports lazily so this module loads in environments without Hermes (tests
    stub ``tools.browser_tool``). ``_last_session_key`` mirrors what
    browser_vision/_browser_eval do — a task's traffic may have been routed to
    a local sidecar session and raw task_id would miss it.
    """
    from tools.browser_tool import _run_browser_command, _last_session_key

    effective = _last_session_key(task_id or "default")
    return _run_browser_command(effective, command, list(args or []), timeout=timeout)


def _caller_from_kwargs(kwargs: dict) -> str:
    task_id_arg = kwargs.get("task_id", "") or ""
    if task_id_arg.startswith("agent_name:"):
        return task_id_arg.split(":", 1)[1]
    return task_id_arg or "unknown"


def _task_id_from_kwargs(kwargs: dict) -> str:
    return kwargs.get("task_id", "") or "default"


# What the page is SAYING right now — error banners, validation messages,
# toasts. Surfaced on every action result so agents never burn turns hunting
# for "what error appeared?" with console scans and vision calls.
_ALERTS_JS = (
    "(function(){var sels='[role=\"alert\"],[aria-live=\"assertive\"],"
    ".error,.alert,.form__error,[class*=\"error-message\"],[id*=\"error-for\"]';"
    "var seen={},out=[];var els=document.querySelectorAll(sels);"
    "for(var i=0;i<els.length&&out.length<3;i++){var el=els[i];"
    "if(!el.offsetParent&&el.getClientRects().length===0)continue;"
    "var t=(el.innerText||'').replace(/\\s+/g,' ').trim();"
    "if(t.length>2&&t.length<400&&!seen[t]){seen[t]=1;out.push(t.slice(0,200));}}"
    "return out;})()"
)

_HUMAN_GATE_RE = re.compile(
    r"captcha|verif|wrong (email|password)|incorrect|sign in|log ?in|"
    r"two.?factor|2fa|code sent|robot", re.IGNORECASE)


def _breadcrumb(task_id: str) -> Dict[str, Any]:
    """Post-action {url, title, page_alerts} so every result shows where the
    page ended up AND what it is telling the user (visible error/alert text)."""
    try:
        from tools.browser_tool import _browser_eval

        raw = _browser_eval(
            "JSON.stringify({url: location.href, title: document.title, "
            "w: innerWidth, h: innerHeight, alerts: " + _ALERTS_JS + "})",
            task_id,
        )
        data = json.loads(raw)
        if data.get("success"):
            res = data.get("result")
            if isinstance(res, str):
                res = json.loads(res)
            if isinstance(res, dict):
                crumb: Dict[str, Any] = {
                    "url": res.get("url"), "title": res.get("title"),
                    "viewport": [res.get("w"), res.get("h")]}
                alerts = [a for a in (res.get("alerts") or [])
                          if isinstance(a, str) and a.strip()]
                if alerts:
                    crumb["page_alerts"] = alerts
                    if any(_HUMAN_GATE_RE.search(a) for a in alerts):
                        crumb["hint"] = (
                            "This looks like a login wall / CAPTCHA / verification "
                            "gate. Do NOT brute-force it — request_human_takeover "
                            "hands this exact browser session to the human.")
                return crumb
    except Exception as e:  # breadcrumb is best-effort, never fails the action
        log.debug("breadcrumb failed: %s", e)
    return {}


def _act(kwargs: dict, command: str, args: List[str],
         extra: Optional[Dict[str, Any]] = None) -> str:
    """Run an action command and return the standard JSON envelope."""
    task_id = _task_id_from_kwargs(kwargs)
    result = _ab(task_id, command, args)
    out: Dict[str, Any] = {
        "success": bool(result.get("success")),
        "command": " ".join([command] + [str(a) for a in args])[:200],
    }
    if not result.get("success"):
        out["error"] = result.get("error", "unknown agent-browser error")
    data = result.get("data")
    if data not in (None, {}, ""):
        out["data"] = data
    if extra:
        out.update(extra)
    out.update(_breadcrumb(task_id))
    return json.dumps(out, ensure_ascii=False, default=str)


def _agent_workspace(caller: str) -> Optional[Path]:
    """The calling agent's workspace dir (parent of its HERMES_HOME)."""
    try:
        from swarm_server.tools import _daemon_registry

        daemon = _daemon_registry.get(caller)
        home = getattr(daemon, "_hermes_home", None) if daemon else None
        if home:
            return Path(home).parent
    except Exception:
        pass
    return None


def _png_dimensions(path: Path) -> Optional[tuple]:
    """(width, height) from a PNG's IHDR header, no image libs needed."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
        if len(head) == 24 and head[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", head[16:24])
            return int(w), int(h)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Handlers — slice 1 (the missing action rungs)
# ---------------------------------------------------------------------------

KEYS_FROM_FILE_MAX_BYTES = 200_000


def _resolve_keys_file(from_file: str, caller: str) -> "tuple[Optional[Path], str]":
    """Resolve a from_file path: absolute, else relative to the caller's team
    project dir, else its workspace. Returns (path, error)."""
    p = Path(from_file)
    candidates = [p] if p.is_absolute() else []
    if not p.is_absolute():
        try:
            from swarm_server.config import _get_project_dir
            from swarm_server.tools import _daemon_registry

            daemon = _daemon_registry.get(caller)
            team = (getattr(daemon, "cfg", None) or {}).get("team_id")
            if team:
                candidates.append(_get_project_dir(team) / p)
        except Exception:
            pass
        ws = _agent_workspace(caller)
        if ws:
            candidates.append(ws / p)
    for cand in candidates:
        if cand.is_file():
            return cand, ""
    return None, (f"file not found: {from_file} (tried: "
                  f"{[str(c) for c in candidates] or 'absolute path only'})")


def _browser_keys_handler(args: dict, **kwargs) -> str:
    text = args.get("text")
    from_file = (args.get("from_file") or "").strip()
    if from_file:
        path, err = _resolve_keys_file(from_file, _caller_from_kwargs(kwargs))
        if path is None:
            return json.dumps({"success": False, "error": err})
        if path.stat().st_size > KEYS_FROM_FILE_MAX_BYTES:
            return json.dumps({"success": False,
                               "error": f"{path} exceeds {KEYS_FROM_FILE_MAX_BYTES} bytes."})
        text = path.read_text(encoding="utf-8", errors="replace")
    if not isinstance(text, str) or text == "":
        return json.dumps({"success": False,
                           "error": "'text' (or a non-empty 'from_file') is required."})

    mode = "inserttext" if args.get("paste") else "type"
    task_id = _task_id_from_kwargs(kwargs)

    # Multi-line text is typed line-by-line with a REAL Enter between lines —
    # one tool call types a whole document (rich editors treat Enter as the
    # paragraph break; embedding \n in a type event does not).
    lines = text.split("\n")
    failed: Optional[Dict[str, Any]] = None
    lines_done = 0
    for i, line in enumerate(lines):
        if line:
            result = _ab(task_id, "keyboard", [mode, line])
            if not result.get("success"):
                failed = result
                break
        if i < len(lines) - 1:
            result = _ab(task_id, "press", ["Enter"])
            if not result.get("success"):
                failed = result
                break
        lines_done = i + 1
    if failed is None and args.get("press_enter"):
        result = _ab(task_id, "press", ["Enter"])
        if not result.get("success"):
            failed = result

    out: Dict[str, Any] = {
        "success": failed is None,
        "command": f"keyboard {mode} <{len(text)} chars, {len(lines)} line(s)>"
                   + (" + Enter" if args.get("press_enter") else ""),
    }
    if from_file:
        out["from_file"] = str(path)
    if failed is not None:
        out["error"] = failed.get("error", "unknown agent-browser error")
        out["lines_typed"] = lines_done
        if len(lines) > 1:
            out["resume_hint"] = (
                f"Lines 1-{lines_done} of {len(lines)} are already in the page — "
                f"verify, then continue from line {lines_done + 1}; do NOT retype "
                f"from the start.")
    out.update(_breadcrumb(task_id))
    return json.dumps(out, ensure_ascii=False, default=str)


def _browser_hover_handler(args: dict, **kwargs) -> str:
    ref = (args.get("ref") or "").strip()
    if not ref:
        return json.dumps({"success": False, "error": "'ref' is required."})
    return _act(kwargs, "hover", [ref])


def _browser_dblclick_handler(args: dict, **kwargs) -> str:
    ref = (args.get("ref") or "").strip()
    if not ref:
        return json.dumps({"success": False, "error": "'ref' is required."})
    return _act(kwargs, "dblclick", [ref])


def _browser_drag_handler(args: dict, **kwargs) -> str:
    src = (args.get("from_ref") or "").strip()
    dst = (args.get("to_ref") or "").strip()
    if not src or not dst:
        return json.dumps({"success": False, "error": "'from_ref' and 'to_ref' are required."})
    return _act(kwargs, "drag", [src, dst])


def _browser_upload_handler(args: dict, **kwargs) -> str:
    ref = (args.get("ref") or "").strip()
    files = args.get("files") or []
    if isinstance(files, str):
        files = [files]
    if not ref or not files:
        return json.dumps({"success": False, "error": "'ref' and 'files' are required."})
    missing = [f for f in files if not Path(str(f)).is_file()]
    if missing:
        return json.dumps({"success": False,
                           "error": f"file(s) not found: {missing} — pass absolute paths."})
    return _act(kwargs, "upload", [ref] + [str(f) for f in files])


def _browser_scrollintoview_handler(args: dict, **kwargs) -> str:
    ref = (args.get("ref") or "").strip()
    if not ref:
        return json.dumps({"success": False, "error": "'ref' is required."})
    return _act(kwargs, "scrollintoview", [ref])


def _browser_wait_handler(args: dict, **kwargs) -> str:
    target = args.get("for")
    if target in (None, ""):
        return json.dumps({"success": False,
                           "error": "'for' is required (a selector/@ref, or milliseconds)."})
    target = str(target).strip()
    try:  # numeric waits are capped so an agent can't park itself for minutes
        ms = int(float(target))
        target = str(min(ms, 15000))
    except ValueError:
        pass
    return _act(kwargs, "wait", [target])


def _browser_click_xy_handler(args: dict, **kwargs) -> str:
    try:
        x, y = int(args.get("x")), int(args.get("y"))
    except (TypeError, ValueError):
        return json.dumps({"success": False, "error": "'x' and 'y' must be integers."})
    if x < 0 or y < 0:
        return json.dumps({"success": False, "error": "'x' and 'y' must be >= 0."})
    button = (args.get("button") or "left").strip().lower()
    if button not in ("left", "right", "middle"):
        return json.dumps({"success": False, "error": "button must be left|right|middle."})
    task_id = _task_id_from_kwargs(kwargs)

    seq = [("mouse", ["move", str(x), str(y)])]
    presses = 2 if args.get("double") else 1
    for _ in range(presses):
        seq.append(("mouse", ["down"] + ([button] if button != "left" else [])))
        seq.append(("mouse", ["up"] + ([button] if button != "left" else [])))
    for command, a in seq:
        result = _ab(task_id, command, a)
        if not result.get("success"):
            out = {"success": False,
                   "command": f"click_xy({x},{y},{button}{',double' if presses == 2 else ''})",
                   "error": result.get("error", f"failed at: {command} {' '.join(a)}")}
            out.update(_breadcrumb(task_id))
            return json.dumps(out, ensure_ascii=False, default=str)

    out = {"success": True,
           "command": f"click_xy({x},{y},{button}{',double' if presses == 2 else ''})"}
    out.update(_breadcrumb(task_id))
    return json.dumps(out, ensure_ascii=False, default=str)


def _browser_screenshot_handler(args: dict, **kwargs) -> str:
    caller = _caller_from_kwargs(kwargs)
    task_id = _task_id_from_kwargs(kwargs)
    label = re.sub(r"[^A-Za-z0-9_-]+", "-", (args.get("label") or "shot"))[:40]

    workspace = _agent_workspace(caller)
    if workspace is None:
        return json.dumps({"success": False,
                           "error": f"no workspace found for agent '{caller}'."})
    shots_dir = workspace / "screenshots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    path = shots_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}.png"

    shot_args: List[str] = []
    if args.get("annotate"):
        shot_args.append("--annotate")
    if args.get("full_page"):
        shot_args.append("--full")
    shot_args.append(str(path))

    result = _ab(task_id, "screenshot", shot_args)
    actual = (result.get("data") or {}).get("path") or str(path)
    apath = Path(actual)
    if not result.get("success") or not apath.exists():
        out = {"success": False,
               "error": result.get("error", f"screenshot not created at {actual}")}
        out.update(_breadcrumb(task_id))
        return json.dumps(out, ensure_ascii=False, default=str)

    out: Dict[str, Any] = {"success": True, "screenshot_path": str(apath)}
    dims = _png_dimensions(apath)
    if dims:
        out["width"], out["height"] = dims
    out.update(_breadcrumb(task_id))
    out["note"] = ("Saved in your workspace. Share with humans via "
                   f"MEDIA:{apath} in a message; inspect it with browser_vision "
                   "or browser_locate if you need to act on what it shows.")
    return json.dumps(out, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Slice 2 — vision-grounded locating (screenshot → "where is X?" → x,y)
# ---------------------------------------------------------------------------

_LOCATE_PROMPT = (
    "You are a precise GUI grounding engine. The attached browser screenshot is "
    "{w}x{h} pixels.\n"
    "Locate this UI element: \"{description}\"\n\n"
    "Reply with ONLY a JSON object, no prose, no code fences:\n"
    "{{\"found\": true|false, \"x\": <int>, \"y\": <int>, \"confidence\": <0.0-1.0>, "
    "\"what_is_there\": \"<one short sentence describing the element and its exact "
    "location, or what you see instead if not found>\"}}\n"
    "x,y must be the CENTER of the element in screenshot pixel coordinates "
    "(origin top-left). If the element is not visible, set found=false and use "
    "what_is_there to say what IS on screen (e.g. a modal covering it)."
)


def _resolve_vision_endpoint(caller: str) -> Dict[str, str]:
    """{base_url, api_key, model} for grounding calls — the agent's effective
    backend (same proxy its own turns use). The model is the agent's MAIN model
    when it can read images (probed once, cached), else the configured vision
    model."""
    from swarm_server.config import (
        LITELLM_API_BASE, LLM_API_KEY, get_vision_model, resolve_screenshot_model,
    )

    base_url, api_key, main_model = "", "", ""
    try:
        from swarm_server.model_config import resolve_model
        from swarm_server.tools import _daemon_registry

        daemon = _daemon_registry.get(caller)
        eff = resolve_model(getattr(daemon, "cfg", None) or {})
        base_url = (eff.get("base_url") or "").strip()
        api_key = (eff.get("api_key") or "").strip()
        main_model = (eff.get("model") or "").strip()
    except Exception as e:
        log.debug("vision endpoint resolve failed (%s); using proxy default", e)
    base_url = base_url or LITELLM_API_BASE
    api_key = api_key or LLM_API_KEY
    model = (resolve_screenshot_model(main_model, base_url, api_key)
             if main_model else get_vision_model())
    return {"base_url": base_url, "api_key": api_key, "model": model}


def _call_vision_model(endpoint: Dict[str, str], prompt: str, png_bytes: bytes) -> str:
    """One OpenAI-compatible chat call with an inline image; returns raw text.

    max_completion_tokens (not max_tokens — reasoning models burn a tight
    budget thinking and return nothing) sized for reasoning + the JSON answer;
    no temperature (the o-series/gpt-5 family rejects non-default values).
    Falls back to max_tokens for backends that reject max_completion_tokens.
    """
    url = endpoint["base_url"].rstrip("/") + "/chat/completions"
    base_payload = {
        "model": endpoint["model"],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,"
                           + base64.b64encode(png_bytes).decode("ascii")}},
            ],
        }],
    }

    def attempt(limit_key: str) -> str:
        payload = dict(base_payload)
        payload[limit_key] = 800
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {endpoint['api_key']}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=VISION_LOCATE_TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return (body.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""

    try:
        return attempt("max_completion_tokens")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        if e.code == 400 and "max_completion_tokens" in detail:
            return attempt("max_tokens")
        raise RuntimeError(f"HTTP {e.code}: {detail[:200]}") from None


def _parse_locate_reply(reply: str) -> Optional[Dict[str, Any]]:
    """Defensive JSON extraction — VLMs love fences and preambles."""
    m = re.search(r"\{.*\}", reply, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _browser_locate_handler(args: dict, **kwargs) -> str:
    description = (args.get("description") or "").strip()
    if not description:
        return json.dumps({"success": False, "error": "'description' is required."})
    caller = _caller_from_kwargs(kwargs)
    task_id = _task_id_from_kwargs(kwargs)

    # Viewport-only screenshot — coordinates must map 1:1 onto what
    # browser_click_xy can reach without scrolling.
    workspace = _agent_workspace(caller)
    shots_dir = (workspace / "screenshots") if workspace else Path("/tmp")
    shots_dir.mkdir(parents=True, exist_ok=True)
    path = shots_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-locate.png"
    shot = _ab(task_id, "screenshot", [str(path)])
    actual = Path((shot.get("data") or {}).get("path") or str(path))
    if not shot.get("success") or not actual.exists():
        return json.dumps({"success": False,
                           "error": shot.get("error", "screenshot failed")})

    dims = _png_dimensions(actual)
    if not dims:
        return json.dumps({"success": False,
                           "error": f"could not read PNG dimensions of {actual}"})
    img_w, img_h = dims

    endpoint = _resolve_vision_endpoint(caller)
    prompt = _LOCATE_PROMPT.format(w=img_w, h=img_h, description=description[:300])
    try:
        reply = _call_vision_model(endpoint, prompt, actual.read_bytes())
    except Exception as e:
        return json.dumps({"success": False, "screenshot_path": str(actual),
                           "error": f"vision model call failed ({endpoint['model']}): {e}"})

    parsed = _parse_locate_reply(reply)
    if parsed is None:
        return json.dumps({"success": False, "screenshot_path": str(actual),
                           "error": "vision model reply was not parseable JSON",
                           "raw_reply": reply[:400]})

    out: Dict[str, Any] = {
        "success": True,
        "found": bool(parsed.get("found")),
        "confidence": parsed.get("confidence"),
        "what_is_there": str(parsed.get("what_is_there", ""))[:300],
        "screenshot_path": str(actual),
        "vision_model": endpoint["model"],
    }
    if out["found"]:
        try:
            x_img, y_img = int(parsed.get("x")), int(parsed.get("y"))
        except (TypeError, ValueError):
            return json.dumps({"success": False, "screenshot_path": str(actual),
                               "error": "vision model returned found=true without integer x,y",
                               "raw_reply": reply[:400]})
        # Map screenshot pixels → CSS pixels (devicePixelRatio may differ).
        crumb = _breadcrumb(task_id)
        vp = crumb.get("viewport") or [None, None]
        if vp[0] and img_w:
            x_css = round(x_img * (vp[0] / img_w))
            y_css = round(y_img * ((vp[1] or img_h) / img_h))
        else:
            x_css, y_css = x_img, y_img
        out["x"], out["y"] = x_css, y_css
        out.update(crumb)
        out["next"] = f"browser_click_xy(x={x_css}, y={y_css}) clicks it."
    return json.dumps(out, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

def _schema(name: str, description: str, properties: Dict[str, Any],
            required: List[str]) -> Dict[str, Any]:
    return {"type": "function",
            "function": {"name": name, "description": description,
                         "parameters": {"type": "object",
                                        "properties": properties,
                                        "required": required}}}


_REF_PROP = {"type": "string",
             "description": "Element ref from the snapshot (e.g. '@e5') or a CSS selector."}

BROWSER_KEYS_TOOL_SCHEMA = _schema(
    "browser_keys",
    "Type with REAL keystrokes into whatever currently has focus — no selector. "
    "THE tool for rich-text editors (Medium/Notion/LinkedIn composers, "
    "contenteditable), canvas apps, and any field where browser_type's "
    "clear-and-fill fails or types into the wrong place. First click/focus the "
    "target area, then call this. Handles MULTI-LINE text: each \\n becomes a "
    "real Enter keypress, so ONE call types a whole document — never type an "
    "article one paragraph per call. To type a file's entire contents without "
    "pasting them into this call, pass from_file instead of text. Set "
    "paste=true to insert each line in one event (no per-key events) — some "
    "editors require real typing, so default is real keys.",
    {"text": {"type": "string",
              "description": "Text to type. Each \\n is pressed as a real Enter "
                             "(paragraph break) — multi-line is fine and preferred."},
     "from_file": {"type": "string",
                   "description": "Type the contents of this file instead of 'text' "
                                  "(absolute path, or relative to the project dir). "
                                  "The whole document goes in ONE call."},
     "paste": {"type": "boolean", "default": False,
               "description": "Insert each line in one event instead of per-key typing "
                              "(faster for long text)."},
     "press_enter": {"type": "boolean", "default": False,
                     "description": "Press Enter once more after everything (submit)."}},
    [],
)

BROWSER_HOVER_TOOL_SCHEMA = _schema(
    "browser_hover",
    "Hover the mouse over an element — opens hover-only menus, tooltips, and "
    "reveals controls that appear on mouse-over (common in dashboards and "
    "editors). Snapshot again afterwards to see what appeared.",
    {"ref": _REF_PROP}, ["ref"],
)

BROWSER_DBLCLICK_TOOL_SCHEMA = _schema(
    "browser_dblclick",
    "Double-click an element (select a word, open an item, enter edit mode in "
    "grids/canvases).",
    {"ref": _REF_PROP}, ["ref"],
)

BROWSER_DRAG_TOOL_SCHEMA = _schema(
    "browser_drag",
    "Drag one element onto another (sliders, kanban cards, reordering, "
    "drag-and-drop uploads zones).",
    {"from_ref": {**_REF_PROP, "description": "Element to drag (ref or selector)."},
     "to_ref": {**_REF_PROP, "description": "Drop target (ref or selector)."}},
    ["from_ref", "to_ref"],
)

BROWSER_UPLOAD_TOOL_SCHEMA = _schema(
    "browser_upload",
    "Attach file(s) to a file input / upload control — file-picker dialogs "
    "cannot be clicked through; this sets the files directly.",
    {"ref": {**_REF_PROP, "description": "The file input or upload button (ref or selector)."},
     "files": {"type": "array", "items": {"type": "string"},
               "description": "Absolute path(s) of file(s) in your workspace."}},
    ["ref", "files"],
)

BROWSER_SCROLLINTOVIEW_TOOL_SCHEMA = _schema(
    "browser_scrollintoview",
    "Scroll a specific element into the viewport (more precise than page "
    "up/down scrolling; needed before click_xy on far-down elements).",
    {"ref": _REF_PROP}, ["ref"],
)

BROWSER_WAIT_TOOL_SCHEMA = _schema(
    "browser_wait",
    "Wait for an element to appear (pass a selector/@ref) or for a fixed delay "
    "(pass milliseconds, max 15000). Use after actions that trigger slow "
    "loads/saves instead of immediately re-snapshotting.",
    {"for": {"type": "string",
             "description": "CSS selector / @ref to wait for, OR a number of milliseconds."}},
    ["for"],
)

BROWSER_CLICK_XY_TOOL_SCHEMA = _schema(
    "browser_click_xy",
    "Click at exact viewport coordinates (CSS pixels, origin top-left). The "
    "universal fallback when an element has NO usable snapshot ref: canvas "
    "UIs, custom widgets, overlays, cross-origin iframes. Get coordinates "
    "from browser_locate (describe the control in words) or from a "
    "browser_screenshot you inspected. Element must be inside the current "
    "viewport — browser_scrollintoview/scroll first if needed.",
    {"x": {"type": "integer", "description": "X in CSS pixels from the left edge."},
     "y": {"type": "integer", "description": "Y in CSS pixels from the top edge."},
     "button": {"type": "string", "enum": ["left", "right", "middle"], "default": "left"},
     "double": {"type": "boolean", "default": False, "description": "Double-click."}},
    ["x", "y"],
)

BROWSER_SCREENSHOT_TOOL_SCHEMA = _schema(
    "browser_screenshot",
    "Save a PNG screenshot of the current page into your workspace "
    "(screenshots/) and return its path — your EYES on the page, and durable "
    "PROOF of visual results others can verify. Use it to check what actually "
    "happened after an important action (did the modal close? did it say "
    "'Published'?), to attach evidence to a RESULT, or to share with a human "
    "via MEDIA:<path>. annotate=true overlays numbered [N] labels mapping to "
    "snapshot refs @eN.",
    {"label": {"type": "string", "description": "Short filename label (e.g. 'after-publish')."},
     "annotate": {"type": "boolean", "default": False,
                  "description": "Overlay [N] labels on interactive elements ([N] ↔ @eN refs)."},
     "full_page": {"type": "boolean", "default": False,
                   "description": "Capture the full page height, not just the viewport."}},
    [],
)

BROWSER_LOCATE_TOOL_SCHEMA = _schema(
    "browser_locate",
    "Find a UI element by DESCRIBING it in words — a vision model looks at a "
    "fresh screenshot and returns the element's x,y for browser_click_xy. Use "
    "when the snapshot has no usable ref for what you can see is on screen "
    "(canvas, icons without labels, custom widgets), or to double-check WHERE "
    "something is before clicking blind. Be visually specific: 'the blue "
    "Publish button in the top-right toolbar'.",
    {"description": {"type": "string",
                     "description": "Visual description of the element, specific enough to disambiguate."}},
    ["description"],
)

GUI_BROWSER_TOOL_SCHEMAS = (
    BROWSER_KEYS_TOOL_SCHEMA,
    BROWSER_HOVER_TOOL_SCHEMA,
    BROWSER_DBLCLICK_TOOL_SCHEMA,
    BROWSER_DRAG_TOOL_SCHEMA,
    BROWSER_UPLOAD_TOOL_SCHEMA,
    BROWSER_SCROLLINTOVIEW_TOOL_SCHEMA,
    BROWSER_WAIT_TOOL_SCHEMA,
    BROWSER_CLICK_XY_TOOL_SCHEMA,
    BROWSER_SCREENSHOT_TOOL_SCHEMA,
    BROWSER_LOCATE_TOOL_SCHEMA,
)

_HANDLERS = {
    "browser_keys": _browser_keys_handler,
    "browser_hover": _browser_hover_handler,
    "browser_dblclick": _browser_dblclick_handler,
    "browser_drag": _browser_drag_handler,
    "browser_upload": _browser_upload_handler,
    "browser_scrollintoview": _browser_scrollintoview_handler,
    "browser_wait": _browser_wait_handler,
    "browser_click_xy": _browser_click_xy_handler,
    "browser_screenshot": _browser_screenshot_handler,
    "browser_locate": _browser_locate_handler,
}


def register_gui_browser_tools(registry) -> None:
    """Register all GUI tools in the Hermes registry (idempotent)."""
    existing = registry.get_tool_to_toolset_map() or {}
    for schema in GUI_BROWSER_TOOL_SCHEMAS:
        name = schema["function"]["name"]
        if name in existing:
            continue
        registry.register(
            name=name,
            toolset="custom",
            schema=schema["function"],
            handler=_HANDLERS[name],
            description=schema["function"]["description"][:120],
        )
        log.info("[%s] Registered", name)
