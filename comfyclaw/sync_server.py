"""
SyncServer — lightweight WebSocket server that broadcasts live workflow
updates from the Python process to connected browser clients (e.g. the
ComfyClaw-Sync ComfyUI extension).

Runs in a background daemon thread using its own asyncio event loop so it
never blocks the main harness loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

log = logging.getLogger(__name__)

try:
    import websockets
    import websockets.server

    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False


class SyncServer:
    """
    Parameters
    ----------
    port : TCP port to listen on (default 8765).
    host : Bind address (default ``"127.0.0.1"``).
    """

    def __init__(self, port: int = 8765, host: str = "0.0.0.0") -> None:
        self.port = port
        self.host = host

        # Shared state protected by a lock (asyncio loop + main thread both access it)
        self._clients: set[Any] = set()
        self._clients_lock = threading.Lock()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop_event: asyncio.Event | None = None
        self._ready = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the WebSocket server in a background daemon thread."""
        if not _WS_AVAILABLE:
            log.warning(
                "[SyncServer] 'websockets' is not installed — live sync disabled. "
                "Install with: pip install 'comfyclaw[sync]'"
            )
            return
        if self._thread and self._thread.is_alive():
            return  # already running

        self._ready.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="comfyclaw-sync")
        self._thread.start()
        self._ready.wait(timeout=5)

    def stop(self) -> None:
        """Signal the server to shut down gracefully."""
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread:
            self._thread.join(timeout=3)
        self._thread = None

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------------
    # Broadcast
    # ------------------------------------------------------------------

    def broadcast(self, workflow: dict) -> None:
        """
        Send a workflow snapshot to all connected clients.

        Safe to call from any thread.  No-op if no clients are connected or
        the server is not running.
        """
        if not self._loop or not self.is_running():
            return
        with self._clients_lock:
            current_clients = set(self._clients)
        if not current_clients:
            return
        payload = json.dumps({"type": "workflow_update", "workflow": workflow})
        asyncio.run_coroutine_threadsafe(
            self._async_broadcast(payload, current_clients), self._loop
        )

    # ------------------------------------------------------------------
    # Asyncio internals
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Entry point for the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            self._loop.close()
            self._loop = None

    async def _serve(self) -> None:
        self._stop_event = asyncio.Event()
        try:
            async with websockets.server.serve(self._handler, self.host, self.port):
                log.info("[SyncServer] Listening on ws://%s:%d", self.host, self.port)
                print(f"[SyncServer] ✅ Listening on ws://{self.host}:{self.port}")
                self._ready.set()
                await self._stop_event.wait()
        except Exception as exc:
            log.error("[SyncServer] Failed to start: %s", exc)
            print(f"[SyncServer] ❌ Failed to start: {exc}")
            self._ready.set()  # unblock start() even on failure

    async def _handler(self, websocket: Any) -> None:
        with self._clients_lock:
            self._clients.add(websocket)
        log.debug("[SyncServer] Client connected (%d total)", len(self._clients))
        try:
            async for _ in websocket:
                pass  # we only push; ignore any client messages
        except Exception:
            pass
        finally:
            with self._clients_lock:
                self._clients.discard(websocket)
            log.debug("[SyncServer] Client disconnected (%d remaining)", len(self._clients))

    async def _async_broadcast(self, payload: str, clients: set[Any]) -> None:
        import websockets.exceptions

        for ws in clients:
            try:
                await ws.send(payload)
            except websockets.exceptions.ConnectionClosed:
                with self._clients_lock:
                    self._clients.discard(ws)
            except Exception as exc:
                log.debug("[SyncServer] Send error: %s", exc)
