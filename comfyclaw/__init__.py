"""
ComfyClaw — agentic harness for self-evolving ComfyUI workflows.

Public API
----------
>>> from comfyclaw import ClawHarness, HarnessConfig
>>> cfg = HarnessConfig(api_key="sk-ant-...", max_iterations=3)
>>> with ClawHarness.from_workflow_file("workflow_api.json", cfg) as h:
...     image_bytes = h.run("a red fox at dawn, photorealistic")
"""

from pathlib import Path

from .client import ComfyClient
from .harness import ClawHarness, HarnessConfig
from .memory import ClawMemory
from .sync_server import SyncServer
from .verifier import ClawVerifier, RegionIssue, VerifierResult
from .workflow import WorkflowManager


def custom_node_path() -> Path:
    """Return the filesystem path to the bundled ComfyClaw-Sync ComfyUI custom node."""
    return Path(__file__).resolve().parent / "custom_node"


__all__ = [
    "ClawHarness",
    "HarnessConfig",
    "ClawMemory",
    "ComfyClient",
    "ClawVerifier",
    "VerifierResult",
    "RegionIssue",
    "SyncServer",
    "WorkflowManager",
    "custom_node_path",
]

__version__ = "0.1.0"
