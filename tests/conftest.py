"""
Shared pytest fixtures for ComfyClaw tests.
"""

from __future__ import annotations

import pytest

from comfyclaw.workflow import WorkflowManager

# ---------------------------------------------------------------------------
# Workflow fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_workflow() -> dict:
    """
    A minimal 3-node SD1.5-style workflow dict in API format.

    Topology:
      [1] CheckpointLoaderSimple
      [2] CLIPTextEncode  (clip ← node1[1])
      [3] KSampler        (model ← node1[0], positive ← node2[0])
    """
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"},
            "inputs": {"ckpt_name": "v1-5-pruned.ckpt"},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive Prompt"},
            "inputs": {"clip": ["1", 1], "text": "a red fox"},
        },
        "3": {
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"},
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "seed": 42,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        },
    }


@pytest.fixture()
def wm(minimal_workflow: dict) -> WorkflowManager:
    return WorkflowManager(minimal_workflow)


# ---------------------------------------------------------------------------
# Image fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def png_bytes() -> bytes:
    """Minimal valid 1×1 white PNG."""
    return (
        b"\x89PNG\r\n\x1a\n"  # PNG magic
        b"\x00\x00\x00\rIHDR"  # IHDR chunk
        b"\x00\x00\x00\x01"  # width=1
        b"\x00\x00\x00\x01"  # height=1
        b"\x08\x02\x00\x00\x00\x90wS\xde"  # bit depth, color type, CRC
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )


@pytest.fixture()
def jpeg_bytes() -> bytes:
    """Minimal valid JPEG magic bytes prefix (enough for media type detection)."""
    return b"\xff\xd8\xff\xe0" + b"\x00" * 100
