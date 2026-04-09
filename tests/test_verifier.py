"""Unit tests for ClawVerifier (Anthropic client mocked)."""

from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock

import pytest

from comfyclaw.verifier import ClawVerifier, VerifierResult, _detect_media_type

# ---------------------------------------------------------------------------
# Media type detection
# ---------------------------------------------------------------------------


class TestDetectMediaType:
    def test_png_magic(self, png_bytes: bytes) -> None:
        assert _detect_media_type(png_bytes) == "image/png"

    def test_jpeg_magic(self, jpeg_bytes: bytes) -> None:
        assert _detect_media_type(jpeg_bytes) == "image/jpeg"

    def test_unknown_defaults_to_png(self) -> None:
        assert _detect_media_type(b"\x00\x01\x02\x03") == "image/png"


# ---------------------------------------------------------------------------
# Helpers to build mock Anthropic responses
# ---------------------------------------------------------------------------


def _text_response(text: str) -> MagicMock:
    content = MagicMock()
    content.text = text
    resp = MagicMock()
    resp.content = [content]
    return resp


def _make_verifier(mock_client: MagicMock) -> ClawVerifier:
    verifier = ClawVerifier.__new__(ClawVerifier)
    verifier.client = mock_client
    verifier.model = "claude-test"
    verifier.score_weights = (0.6, 0.4)
    verifier.max_workers = 2
    return verifier


# ---------------------------------------------------------------------------
# Decompose prompt
# ---------------------------------------------------------------------------


class TestDecomposePrompt:
    def test_parses_json_array(self) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response(
            '["Is it red?", "Is there a fox?"]'
        )
        v = _make_verifier(mock_client)
        questions = v._decompose_prompt("a red fox")
        assert questions == ["Is it red?", "Is there a fox?"]

    def test_falls_back_to_line_parse(self) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response(
            "Here are the questions:\nIs it red?\nIs there a fox?"
        )
        v = _make_verifier(mock_client)
        questions = v._decompose_prompt("a red fox")
        assert len(questions) == 2


# ---------------------------------------------------------------------------
# Requirement checks
# ---------------------------------------------------------------------------


class TestCheckRequirements:
    def test_all_yes_gives_full_score(self, png_bytes: bytes) -> None:
        mock_client = MagicMock()
        # decompose → 2 questions; each check → yes
        mock_client.messages.create.side_effect = [
            _text_response('["Is there a fox?", "Is the fox red?"]'),
            _text_response("yes"),
            _text_response("yes"),
            _text_response(
                json.dumps(
                    {
                        "overall_assessment": "Perfect",
                        "score": 1.0,
                        "region_issues": [],
                        "evolution_suggestions": [],
                    }
                )
            ),
        ]
        v = _make_verifier(mock_client)
        result = v.verify(png_bytes, "a red fox")
        assert result.passed == ["Is there a fox?", "Is the fox red?"]
        assert result.failed == []

    def test_partial_yes_no(self, png_bytes: bytes) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            _text_response('["Q1?", "Q2?"]'),
            _text_response("yes"),
            _text_response("no"),
            _text_response(
                json.dumps(
                    {
                        "overall_assessment": "Partial",
                        "score": 0.5,
                        "region_issues": [],
                        "evolution_suggestions": [],
                    }
                )
            ),
        ]
        v = _make_verifier(mock_client)
        result = v.verify(png_bytes, "prompt")
        assert len(result.passed) == 1
        assert len(result.failed) == 1

    def test_media_type_jpeg_sent_to_api(self, jpeg_bytes: bytes) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            _text_response('["Q?"]'),
            _text_response("yes"),
            _text_response(
                json.dumps(
                    {
                        "overall_assessment": "ok",
                        "score": 1.0,
                        "region_issues": [],
                        "evolution_suggestions": [],
                    }
                )
            ),
        ]
        v = _make_verifier(mock_client)
        v.verify(jpeg_bytes, "prompt")
        # Check that at least one call used image/jpeg
        all_calls = mock_client.messages.create.call_args_list
        image_calls = [
            c
            for c in all_calls
            if any(
                msg.get("content")
                and isinstance(msg["content"], list)
                and any(
                    part.get("type") == "image"
                    and part.get("source", {}).get("media_type") == "image/jpeg"
                    for part in msg["content"]
                    if isinstance(part, dict)
                )
                for msg in c.kwargs.get("messages", [])
            )
        ]
        assert len(image_calls) > 0

    def test_encode_once_shared_across_checks(self, png_bytes: bytes) -> None:
        """
        The base64 string passed to all API calls for the same verify() call
        must be identical (encoded once, not re-encoded per question).
        """
        call_b64_strings: list[str] = []
        mock_client = MagicMock()

        def capture_create(**kwargs):
            for msg in kwargs.get("messages", []):
                content = msg.get("content", [])
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "image":
                            call_b64_strings.append(part["source"]["data"])
            return _text_response("yes")

        mock_client.messages.create.side_effect = [
            _text_response('["Q1?", "Q2?", "Q3?"]'),
            capture_create,
            capture_create,
            capture_create,
            _text_response(
                json.dumps(
                    {
                        "overall_assessment": "ok",
                        "score": 0.9,
                        "region_issues": [],
                        "evolution_suggestions": [],
                    }
                )
            ),
        ]
        v = _make_verifier(mock_client)
        v.verify(png_bytes, "a prompt")

        expected_b64 = base64.standard_b64encode(png_bytes).decode()
        for s in call_b64_strings:
            assert s == expected_b64


# ---------------------------------------------------------------------------
# Region issue parsing
# ---------------------------------------------------------------------------


class TestRegionIssues:
    def test_region_issues_parsed(self, png_bytes: bytes) -> None:
        detail = {
            "overall_assessment": "Needs work",
            "score": 0.5,
            "region_issues": [
                {
                    "region": "background",
                    "issue_type": "lighting",
                    "description": "Too flat",
                    "severity": "high",
                    "fix_strategies": ["add_controlnet_depth"],
                },
                {
                    "region": "face",
                    "issue_type": "texture",
                    "description": "Waxy skin",
                    "severity": "medium",
                    "fix_strategies": ["inject_lora_detail"],
                },
            ],
            "evolution_suggestions": ["Add depth ControlNet"],
        }
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            _text_response('["Q?"]'),
            _text_response("yes"),
            _text_response(json.dumps(detail)),
        ]
        v = _make_verifier(mock_client)
        result = v.verify(png_bytes, "a portrait")
        assert len(result.region_issues) == 2
        assert result.region_issues[0].region == "background"
        assert result.region_issues[0].severity == "high"
        assert "add_controlnet_depth" in result.region_issues[0].fix_strategies
        assert result.evolution_suggestions == ["Add depth ControlNet"]


# ---------------------------------------------------------------------------
# Score blending
# ---------------------------------------------------------------------------


class TestScoreBlending:
    def test_blends_req_and_detail(self, png_bytes: bytes) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            _text_response('["Q1?", "Q2?"]'),
            _text_response("yes"),
            _text_response("no"),  # 50% requirement pass rate
            _text_response(
                json.dumps(
                    {
                        "overall_assessment": "ok",
                        "score": 0.8,
                        "region_issues": [],
                        "evolution_suggestions": [],
                    }
                )
            ),
        ]
        v = ClawVerifier.__new__(ClawVerifier)
        v.client = mock_client
        v.model = "test"
        v.score_weights = (0.5, 0.5)
        v.max_workers = 2
        result = v.verify(png_bytes, "p")
        # 0.5 * 0.5 + 0.5 * 0.8 = 0.65
        assert abs(result.score - 0.65) < 0.01

    def test_uses_req_only_when_detail_score_none(self, png_bytes: bytes) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            _text_response('["Q?"]'),
            _text_response("yes"),
            _text_response(
                '{"overall_assessment": "err", "region_issues": [], "evolution_suggestions": []}'
            ),
        ]
        v = _make_verifier(mock_client)
        result = v.verify(png_bytes, "p")
        assert result.score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# VerifierResult.format_feedback
# ---------------------------------------------------------------------------


class TestFormatFeedback:
    def test_contains_score(self) -> None:
        result = VerifierResult(
            score=0.75,
            checks=[],
            passed=["Q1"],
            failed=["Q2"],
            region_issues=[],
            overall_assessment="Good",
            evolution_suggestions=["Add LoRA"],
        )
        text = result.format_feedback()
        assert "0.75" in text
        assert "Q1" in text
        assert "Q2" in text
        assert "Add LoRA" in text
