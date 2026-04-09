"""
ComfyClient — thin HTTP client for the ComfyUI REST API.

All network operations live here. Nothing else in the package should
call urllib or requests directly.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any


class ComfyAPIError(Exception):
    """Raised when ComfyUI returns an HTTP error."""

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"ComfyUI API error {status}: {body[:400]}")


class ComfyClient:
    """
    Thin HTTP client for ComfyUI.

    Parameters
    ----------
    server_address : ``host:port`` of a running ComfyUI instance.
    client_id      : Optional stable UUID for queue tracking.
    """

    def __init__(
        self,
        server_address: str = "127.0.0.1:8188",
        client_id: str | None = None,
    ) -> None:
        self.server_address = server_address
        self.client_id = client_id or str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def queue_prompt(self, prompt: dict) -> dict:
        """Submit a workflow prompt to the queue. Returns ``{"prompt_id": "...", ...}``."""
        payload = json.dumps({"prompt": prompt, "client_id": self.client_id}).encode()
        req = urllib.request.Request(
            f"http://{self.server_address}/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ComfyAPIError(exc.code, body) from exc

    def get_history(self, prompt_id: str) -> dict:
        """Fetch execution history entry for *prompt_id*."""
        url = f"http://{self.server_address}/history/{urllib.parse.quote(prompt_id)}"
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read())

    def get_image(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        """Download a generated image from the ComfyUI output directory."""
        params = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": folder_type}
        )
        with urllib.request.urlopen(f"http://{self.server_address}/view?{params}") as resp:
            return resp.read()

    def get_json(self, path: str, timeout: int = 10) -> Any:
        """GET ``http://{server_address}{path}`` and parse JSON."""
        url = f"http://{self.server_address}{path}"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())

    def object_info(self, node_class: str | None = None) -> dict:
        """Return ComfyUI object_info for one node class or all nodes."""
        path = "/object_info"
        if node_class:
            path = f"/object_info/{urllib.parse.quote(node_class)}"
        return self.get_json(path)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def is_alive(self, timeout: int = 4) -> bool:
        """Return True if the ComfyUI server responds to /system_stats."""
        try:
            self.get_json("/system_stats", timeout=timeout)
            return True
        except Exception:
            pass
        try:
            self.get_json("/object_info", timeout=timeout)
            return True
        except Exception:
            return False

    def wait_for_completion(
        self, prompt_id: str, timeout: int = 600, poll_interval: float = 2.0
    ) -> dict:
        """
        Poll history until the prompt finishes or *timeout* seconds elapse.

        Returns a history entry dict.  On execution error, the dict contains
        an ``"error"`` key with the exception message extracted from ComfyUI's
        status messages.

        Raises
        ------
        TimeoutError
            If the workflow has not finished within *timeout* seconds.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                history = self.get_history(prompt_id)
            except Exception as exc:
                return {"error": str(exc)}

            entry = history.get(prompt_id)
            if entry is None:
                time.sleep(poll_interval)
                continue

            status = entry.get("status", {})
            if status.get("status_str") == "error":
                err = "(unknown ComfyUI execution error)"
                for kind, data in status.get("messages", []):
                    if kind == "execution_error":
                        err = data.get("exception_message", err)
                        break
                return {"error": f"ComfyUI execution error: {err}"}

            return entry

        raise TimeoutError(f"Workflow {prompt_id!r} did not finish within {timeout}s")

    def collect_images(self, history_entry: dict) -> list[bytes]:
        """Extract all output images from a history entry as a list of raw bytes."""
        images: list[bytes] = []
        for node_output in history_entry.get("outputs", {}).values():
            for img in node_output.get("images", []):
                images.append(
                    self.get_image(img["filename"], img["subfolder"], img["type"])
                )
        return images
