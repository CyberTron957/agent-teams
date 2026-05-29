"""WebSocket broadcasting and shared async state."""

import asyncio
import json
import logging
import threading
from typing import Optional, Set

from fastapi import WebSocket

log = logging.getLogger("swarm.websocket")

# ---------------------------------------------------------------------------
# Global async state
# ---------------------------------------------------------------------------
_main_event_loop: Optional[asyncio.AbstractEventLoop] = None

# Global lock to serialize agent initialization (HERMES_HOME is process-scoped)
_agent_init_lock = threading.Lock()


class WSBroadcaster:
    def __init__(self):
        self.clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self.clients.add(ws)
        log.info("[WS] Client connected. Total: %d", len(self.clients))

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self.clients.discard(ws)
        log.info("[WS] Client disconnected. Total: %d", len(self.clients))

    async def broadcast(self, event_type: str, payload: dict):
        if not self.clients:
            return
        message = json.dumps({"type": event_type, "payload": payload})
        async with self._lock:
            clients = list(self.clients)

        async def _send(ws):
            try:
                # Per-client timeout: a slow/half-dead client must not stall
                # delivery to everyone else (sends now run concurrently).
                await asyncio.wait_for(ws.send_text(message), timeout=5.0)
                return None
            except Exception:
                return ws

        results = await asyncio.gather(*(_send(ws) for ws in clients))
        disconnected = [ws for ws in results if ws is not None]
        if disconnected:
            async with self._lock:
                for ws in disconnected:
                    self.clients.discard(ws)


ws_broadcaster = WSBroadcaster()


def _broadcast(event_type: str, payload: dict):
    """Thread-safe event broadcast. Works from any thread or async context."""
    if not _main_event_loop or not _main_event_loop.is_running():
        return
    try:
        try:
            current_loop = asyncio.get_running_loop()
            if current_loop is _main_event_loop:
                asyncio.create_task(ws_broadcaster.broadcast(event_type, payload))
                return
        except RuntimeError:
            pass

        asyncio.run_coroutine_threadsafe(
            ws_broadcaster.broadcast(event_type, payload),
            _main_event_loop,
        )
    except Exception as e:
        log.warning("[Broadcast] Failed (%s): %s", event_type, e)
