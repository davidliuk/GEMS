"""
SyncServer — lightweight WebSocket server that broadcasts live workflow
updates from the Python process to connected browser clients (e.g. the
ComfyClaw-Sync ComfyUI extension).

Runs in a background daemon thread using its own asyncio event loop so it
never blocks the main harness loop.

Supports these message types (server → client):
  - ``workflow_update``   — full workflow snapshot (reconnect / initial load).
  - ``workflow_diff``     — incremental ops (add_node, remove_node, update_node).
  - ``request_feedback``  — ask the human for feedback on a generated image.

And these (client → server):
  - ``human_feedback``    — human's text feedback, score, and action.
"""

from __future__ import annotations

import asyncio
import copy
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


# ------------------------------------------------------------------
# Diff helpers
# ------------------------------------------------------------------


def diff_workflows(old: dict, new: dict) -> list[dict]:
    """
    Compare two API-format workflow dicts and return a list of ops.

    Each op is one of::

        {"op": "add_node",    "id": "<node_id>", "data": {class_type, inputs, …}}
        {"op": "remove_node", "id": "<node_id>"}
        {"op": "update_node", "id": "<node_id>", "data": {class_type, inputs, …}}

    The full node ``data`` is included in ``update_node`` so the client can
    replace its state wholesale rather than patching individual keys.
    """
    ops: list[dict] = []
    old_keys = set(old)
    new_keys = set(new)

    for nid in sorted(new_keys - old_keys, key=lambda k: int(k)):
        ops.append({"op": "add_node", "id": nid, "data": new[nid]})

    for nid in sorted(old_keys - new_keys, key=lambda k: int(k)):
        ops.append({"op": "remove_node", "id": nid})

    for nid in sorted(old_keys & new_keys, key=lambda k: int(k)):
        if old[nid] != new[nid]:
            ops.append({"op": "update_node", "id": nid, "data": new[nid]})

    return ops


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

        self._clients: set[Any] = set()
        self._clients_lock = threading.Lock()

        self._last_workflow: dict | None = None
        self._wf_lock = threading.Lock()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop_event: asyncio.Event | None = None
        self._ready = threading.Event()

        self._feedback_future: asyncio.Future[dict] | None = None
        self._feedback_lock = threading.Lock()

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

    def reset(self, *, empty: bool = False) -> None:
        """Clear remembered workflow state.

        Args:
            empty: If True, set to ``{}`` so the next broadcast produces
                   diffs (add_node ops) instead of a full snapshot.
        """
        with self._wf_lock:
            self._last_workflow = {} if empty else None

    def has_clients(self) -> bool:
        """Return True if at least one WebSocket client is connected."""
        with self._clients_lock:
            return len(self._clients) > 0

    # ------------------------------------------------------------------
    # Human-in-the-loop feedback
    # ------------------------------------------------------------------

    def request_feedback(
        self,
        image_path: str | None = None,
        vlm_summary: str | None = None,
        iteration: int = 0,
        prompt: str = "",
    ) -> None:
        """Broadcast a ``request_feedback`` message to all connected clients."""
        if not self._loop or not self.is_running():
            return
        with self._clients_lock:
            clients = set(self._clients)
        if not clients:
            return

        msg = {
            "type": "request_feedback",
            "image_path": image_path,
            "vlm_summary": vlm_summary,
            "iteration": iteration,
            "prompt": prompt,
        }
        payload = json.dumps(msg)
        asyncio.run_coroutine_threadsafe(
            self._async_broadcast(payload, clients), self._loop
        )

    def wait_for_human_feedback(self, timeout: float = 600.0) -> dict | None:
        """Block the calling thread until a client sends ``human_feedback``.

        Returns the parsed feedback dict, or ``None`` on timeout.
        Safe to call from any thread (typically the main harness thread).
        """
        if not self._loop or not self.is_running():
            return None

        future: asyncio.Future[dict] = asyncio.run_coroutine_threadsafe(
            self._create_feedback_future(), self._loop
        ).result(timeout=5)

        try:
            return future.result(timeout=timeout)
        except Exception:
            with self._feedback_lock:
                self._feedback_future = None
            return None

    async def _create_feedback_future(self) -> asyncio.Future[dict]:
        """Create and store a Future in the event loop thread."""
        loop = asyncio.get_running_loop()
        with self._feedback_lock:
            if self._feedback_future and not self._feedback_future.done():
                self._feedback_future.cancel()
            self._feedback_future = loop.create_future()
            return self._feedback_future

    def _resolve_feedback(self, data: dict) -> None:
        """Resolve the pending feedback future (called from the event loop thread)."""
        with self._feedback_lock:
            if self._feedback_future and not self._feedback_future.done():
                self._feedback_future.set_result(data)
                self._feedback_future = None

    # ------------------------------------------------------------------
    # Broadcast
    # ------------------------------------------------------------------

    def broadcast(self, workflow: dict) -> None:
        """
        Send an incremental diff to all connected clients.

        On the first call (or after ``reset()``) a full ``workflow_update``
        is sent.  Subsequent calls compute the diff against the previous
        snapshot and send a ``workflow_diff`` with granular ops.

        Safe to call from any thread.
        """
        if not self._loop or not self.is_running():
            with self._wf_lock:
                self._last_workflow = copy.deepcopy(workflow)
            return

        with self._clients_lock:
            current_clients = set(self._clients)
        if not current_clients:
            with self._wf_lock:
                self._last_workflow = copy.deepcopy(workflow)
            return

        with self._wf_lock:
            prev = self._last_workflow
            if prev is None:
                payload = json.dumps({
                    "type": "workflow_update",
                    "workflow": workflow,
                })
            else:
                ops = diff_workflows(prev, workflow)
                if not ops:
                    self._last_workflow = copy.deepcopy(workflow)
                    return
                payload = json.dumps({
                    "type": "workflow_diff",
                    "ops": ops,
                    "full": workflow,
                })
            self._last_workflow = copy.deepcopy(workflow)

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

        # Bootstrap: send the full current workflow so the client starts
        # from the right state before receiving subsequent diffs.
        with self._wf_lock:
            snapshot = copy.deepcopy(self._last_workflow) if self._last_workflow else None
        if snapshot is not None:
            try:
                await websocket.send(json.dumps({
                    "type": "workflow_update",
                    "workflow": snapshot,
                }))
            except Exception:
                pass

        try:
            async for raw in websocket:
                try:
                    msg = json.loads(raw)
                    if isinstance(msg, dict) and msg.get("type") == "human_feedback":
                        log.info("[SyncServer] Received human_feedback")
                        self._resolve_feedback(msg)
                except (json.JSONDecodeError, TypeError):
                    pass
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
