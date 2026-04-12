"""Tests for HumanVerifier and HybridVerifier."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from comfyclaw.human_verifier import HumanVerifier, HybridVerifier
from comfyclaw.verifier import VerifierResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_sync():
    """A mock SyncServer with controllable behavior."""
    sync = MagicMock()
    sync.has_clients.return_value = True
    return sync


@pytest.fixture()
def png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01"
        b"\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )


@pytest.fixture()
def vlm_result() -> VerifierResult:
    return VerifierResult(
        score=0.7,
        checks=[],
        passed=["good composition"],
        failed=["blurry background"],
        overall_assessment="Decent image with some blur issues.",
        evolution_suggestions=["add depth ControlNet"],
    )


# ---------------------------------------------------------------------------
# HumanVerifier
# ---------------------------------------------------------------------------


class TestHumanVerifier:
    def test_websocket_feedback(self, mock_sync, png_bytes, tmp_path):
        mock_sync.wait_for_human_feedback.return_value = {
            "type": "human_feedback",
            "text": "The lighting is too flat, needs more contrast",
            "score": 0.4,
            "action": "override",
        }
        verifier = HumanVerifier(sync_server=mock_sync, output_dir=str(tmp_path))
        result = verifier.verify(png_bytes, "a cat", iteration=1)

        assert result.score == 0.4
        assert result.feedback_source == "human"
        assert "lighting" in result.overall_assessment.lower()
        assert len(result.evolution_suggestions) >= 1
        mock_sync.request_feedback.assert_called_once()

    def test_websocket_accept(self, mock_sync, png_bytes, tmp_path):
        mock_sync.wait_for_human_feedback.return_value = {
            "text": "",
            "score": 0.85,
            "action": "accept",
        }
        verifier = HumanVerifier(sync_server=mock_sync, output_dir=str(tmp_path))
        result = verifier.verify(png_bytes, "a cat", iteration=1)

        assert result.score == 0.85
        assert result.feedback_source == "human"

    def test_websocket_timeout(self, mock_sync, png_bytes, tmp_path):
        mock_sync.wait_for_human_feedback.return_value = None
        verifier = HumanVerifier(sync_server=mock_sync, output_dir=str(tmp_path))
        result = verifier.verify(png_bytes, "a cat", iteration=1)

        assert result.score == 0.5
        assert "timeout" in result.overall_assessment.lower()

    def test_terminal_fallback_no_sync(self, png_bytes, tmp_path):
        verifier = HumanVerifier(sync_server=None, output_dir=str(tmp_path))
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            result = verifier.verify(png_bytes, "a cat", iteration=1)

        assert result.score == 0.5
        assert result.feedback_source == "human"

    def test_terminal_fallback_no_clients(self, mock_sync, png_bytes, tmp_path):
        mock_sync.has_clients.return_value = False
        verifier = HumanVerifier(sync_server=mock_sync, output_dir=str(tmp_path))
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            result = verifier.verify(png_bytes, "a cat", iteration=1)

        assert result.score == 0.5
        mock_sync.request_feedback.assert_not_called()

    def test_multi_line_feedback_parsing(self, mock_sync, png_bytes, tmp_path):
        mock_sync.wait_for_human_feedback.return_value = {
            "text": "- too dark\n- needs warmer colors\n- add more detail to the eyes",
            "score": 0.3,
            "action": "override",
        }
        verifier = HumanVerifier(sync_server=mock_sync, output_dir=str(tmp_path))
        result = verifier.verify(png_bytes, "a cat", iteration=1)

        assert len(result.evolution_suggestions) == 3
        assert any("dark" in s for s in result.evolution_suggestions)
        assert any("warm" in s for s in result.evolution_suggestions)

    def test_saves_image_to_output_dir(self, mock_sync, png_bytes, tmp_path):
        mock_sync.wait_for_human_feedback.return_value = {
            "text": "",
            "score": 0.8,
            "action": "accept",
        }
        verifier = HumanVerifier(sync_server=mock_sync, output_dir=str(tmp_path))
        verifier.verify(png_bytes, "a cat", iteration=2)

        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert "iter2" in files[0].name


# ---------------------------------------------------------------------------
# HybridVerifier
# ---------------------------------------------------------------------------


class TestHybridVerifier:
    def test_accept_returns_vlm_result(self, mock_sync, png_bytes, vlm_result, tmp_path):
        mock_vlm = MagicMock()
        mock_vlm.verify.return_value = vlm_result

        mock_sync.wait_for_human_feedback.return_value = {
            "text": "",
            "score": 0.85,
            "action": "accept",
        }

        verifier = HybridVerifier(
            vlm_verifier=mock_vlm, sync_server=mock_sync, output_dir=str(tmp_path)
        )
        result = verifier.verify(png_bytes, "a cat", iteration=1)

        assert result.score == 0.7  # VLM score unchanged
        assert result is vlm_result
        mock_vlm.verify.assert_called_once()
        mock_sync.request_feedback.assert_called_once()
        call_kwargs = mock_sync.request_feedback.call_args
        assert call_kwargs[1].get("vlm_summary") is not None

    def test_override_merges_feedback(self, mock_sync, png_bytes, vlm_result, tmp_path):
        mock_vlm = MagicMock()
        mock_vlm.verify.return_value = vlm_result

        mock_sync.wait_for_human_feedback.return_value = {
            "text": "make it more colorful",
            "score": 0.5,
            "action": "override",
        }

        verifier = HybridVerifier(
            vlm_verifier=mock_vlm, sync_server=mock_sync, output_dir=str(tmp_path)
        )
        result = verifier.verify(png_bytes, "a cat", iteration=1)

        assert result.score == 0.5
        assert "[Human override]" in result.overall_assessment
        assert any("[HUMAN]" in s for s in result.evolution_suggestions)
        assert "add depth ControlNet" in result.evolution_suggestions

    def test_timeout_returns_vlm_result(self, mock_sync, png_bytes, vlm_result, tmp_path):
        mock_vlm = MagicMock()
        mock_vlm.verify.return_value = vlm_result
        mock_sync.wait_for_human_feedback.return_value = None

        verifier = HybridVerifier(
            vlm_verifier=mock_vlm, sync_server=mock_sync, output_dir=str(tmp_path)
        )
        result = verifier.verify(png_bytes, "a cat", iteration=1)

        assert result is vlm_result

    def test_no_clients_uses_terminal_fallback(self, png_bytes, vlm_result, tmp_path):
        mock_vlm = MagicMock()
        mock_vlm.verify.return_value = vlm_result
        mock_sync = MagicMock()
        mock_sync.has_clients.return_value = False

        verifier = HybridVerifier(
            vlm_verifier=mock_vlm, sync_server=mock_sync, output_dir=str(tmp_path)
        )
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            result = verifier.verify(png_bytes, "a cat", iteration=1)

        assert result is vlm_result
        mock_sync.request_feedback.assert_not_called()


# ---------------------------------------------------------------------------
# VerifierResult.feedback_source
# ---------------------------------------------------------------------------


class TestFeedbackSource:
    def test_default_feedback_source_is_vlm(self):
        result = VerifierResult(score=0.5, checks=[], passed=[], failed=[])
        assert result.feedback_source == "vlm"

    def test_human_verifier_sets_source(self, mock_sync, png_bytes, tmp_path):
        mock_sync.wait_for_human_feedback.return_value = {
            "text": "ok",
            "score": 0.7,
            "action": "override",
        }
        verifier = HumanVerifier(sync_server=mock_sync, output_dir=str(tmp_path))
        result = verifier.verify(png_bytes, "test", iteration=0)
        assert result.feedback_source == "human"

    def test_hybrid_override_sets_source(self, mock_sync, png_bytes, vlm_result, tmp_path):
        mock_vlm = MagicMock()
        mock_vlm.verify.return_value = vlm_result
        mock_sync.wait_for_human_feedback.return_value = {
            "text": "fix it",
            "score": 0.4,
            "action": "override",
        }
        verifier = HybridVerifier(
            vlm_verifier=mock_vlm, sync_server=mock_sync, output_dir=str(tmp_path)
        )
        result = verifier.verify(png_bytes, "test", iteration=0)
        assert result.feedback_source == "hybrid"
