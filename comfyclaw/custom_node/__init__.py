"""
ComfyClaw-Sync — ComfyUI custom node that injects the live sync JS extension.

This directory is bundled inside the ``comfyclaw`` Python package.  Install it
into ComfyUI's custom_nodes folder with:

    comfyclaw install-node [--comfyui-dir /path/to/ComfyUI]

Or manually:

    # Option A — symlink (edits here take effect immediately):
    ln -s $(python -c "import comfyclaw, pathlib; print(pathlib.Path(comfyclaw.__file__).parent/'custom_node')") \\
          ~/Documents/ComfyUI/custom_nodes/ComfyClaw-Sync

    # Option B — copy:
    cp -r $(python -c "import comfyclaw, pathlib; print(pathlib.Path(comfyclaw.__file__).parent/'custom_node')") \\
          ~/Documents/ComfyUI/custom_nodes/ComfyClaw-Sync

After installation, restart ComfyUI.  The extension connects automatically to
ws://127.0.0.1:8765 (the default ComfyClaw sync server port).

Protocol
--------
The Python SyncServer pushes JSON messages over WebSocket::

    { "type": "workflow_update", "workflow": { "<node_id>": {...}, ... } }

The JS extension receives ``workflow``, converts it to LiteGraph format, and
reloads the canvas.  A status badge in the bottom-right corner shows
🟢 live / 🔴 disconnected / ✨ updated.

WebSocket URL resolution
------------------------
The JS client connects to ``ws://<server_hostname>:8765`` by default,
using ``window.location.hostname`` so it works correctly whether you access
ComfyUI locally or over a remote tunnel.  You can override the URL by setting
``localStorage.setItem('comfyclaw_ws_url', 'ws://...')`` in the browser console.
"""

from __future__ import annotations

import os

WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "js")

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
