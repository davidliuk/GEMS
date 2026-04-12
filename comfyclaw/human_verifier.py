"""
Human-in-the-loop verifiers for ComfyClaw.

HumanVerifier — collects feedback from a human reviewer via the ComfyUI
feedback panel (WebSocket) or terminal stdin as a fallback.

HybridVerifier — runs the VLM verifier first, then presents the VLM
results to the human for acceptance or override.

Both return ``VerifierResult`` objects compatible with the existing
harness loop — no changes needed downstream.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from .verifier import VerifierResult


class HumanVerifier:
    """Collect verification feedback from a human reviewer.

    Parameters
    ----------
    sync_server : Optional SyncServer instance for WebSocket-based feedback.
    timeout     : Seconds to wait for human feedback before timing out.
    output_dir  : Directory where generated images are saved for the human
                  to view (used in terminal fallback mode).
    """

    def __init__(
        self,
        sync_server: object | None = None,
        timeout: float = 600.0,
        output_dir: str | None = None,
    ) -> None:
        self._sync = sync_server
        self._timeout = timeout
        self._output_dir = output_dir or tempfile.gettempdir()

    def verify(self, image_bytes: bytes, prompt: str, iteration: int = 0) -> VerifierResult:
        """Request human feedback and convert it to a VerifierResult."""
        image_path = self._save_temp_image(image_bytes, iteration)

        if self._sync and self._sync.has_clients():
            return self._verify_via_websocket(image_path, prompt, iteration)
        return self._verify_via_terminal(image_path, prompt, iteration)

    def _save_temp_image(self, image_bytes: bytes, iteration: int) -> str:
        out_dir = Path(self._output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"comfyclaw_review_iter{iteration}.png"
        path.write_bytes(image_bytes)
        return str(path)

    def _verify_via_websocket(
        self, image_path: str, prompt: str, iteration: int
    ) -> VerifierResult:
        print(f"[HumanVerifier] 📝 Requesting feedback via ComfyUI panel (timeout {self._timeout}s)…")
        self._sync.request_feedback(
            image_path=image_path,
            vlm_summary=None,
            iteration=iteration,
            prompt=prompt,
        )
        feedback = self._sync.wait_for_human_feedback(timeout=self._timeout)

        if feedback is None:
            print("[HumanVerifier] ⏰ Timeout — no feedback received, using default score.")
            return self._make_result(
                score=0.5,
                text="No feedback received (timeout).",
                action="timeout",
            )

        return self._make_result(
            score=feedback.get("score", 0.5),
            text=feedback.get("text", ""),
            action=feedback.get("action", "override"),
        )

    def _verify_via_terminal(
        self, image_path: str, prompt: str, iteration: int
    ) -> VerifierResult:
        print(f"\n{'─' * 60}")
        print(f"[HumanVerifier] 📝 Image saved to: {image_path}")
        print(f"[HumanVerifier] Prompt: {prompt}")
        print(f"[HumanVerifier] Iteration: {iteration}")
        print(f"{'─' * 60}")

        if not sys.stdin.isatty():
            print("[HumanVerifier] stdin is not a terminal — using default score.")
            return self._make_result(score=0.5, text="Non-interactive mode.", action="accept")

        print("\nRate the image quality (0.0–1.0, or press Enter for 0.5):")
        score_raw = input("  Score> ").strip()
        try:
            score = float(score_raw) if score_raw else 0.5
            score = max(0.0, min(1.0, score))
        except ValueError:
            score = 0.5

        print("\nWhat should be improved? (Enter for nothing / type feedback):")
        text = input("  Feedback> ").strip()

        action = "accept" if not text else "override"
        return self._make_result(score=score, text=text, action=action)

    @staticmethod
    def _make_result(score: float, text: str, action: str) -> VerifierResult:
        suggestions = []
        if text:
            for line in text.replace(";", "\n").split("\n"):
                line = line.strip().lstrip("-•*0123456789.) ")
                if line:
                    suggestions.append(line)

        assessment = text if text else ("Accepted by human reviewer." if action == "accept" else "")

        return VerifierResult(
            score=score,
            checks=[],
            passed=[],
            failed=[],
            region_issues=[],
            overall_assessment=assessment,
            evolution_suggestions=suggestions,
            feedback_source="human",
        )


class HybridVerifier:
    """Run VLM verification first, then let a human accept or override.

    Parameters
    ----------
    vlm_verifier  : The existing VLM-based verifier.
    sync_server   : SyncServer for WebSocket feedback.
    timeout       : Seconds to wait for human feedback.
    output_dir    : Directory for temporary review images.
    """

    def __init__(
        self,
        vlm_verifier: object,
        sync_server: object | None = None,
        timeout: float = 600.0,
        output_dir: str | None = None,
    ) -> None:
        self._vlm = vlm_verifier
        self._sync = sync_server
        self._timeout = timeout
        self._output_dir = output_dir or tempfile.gettempdir()

    def verify(self, image_bytes: bytes, prompt: str, iteration: int = 0) -> VerifierResult:
        print("[HybridVerifier] 🤖 Running VLM verification first…")
        vlm_result = self._vlm.verify(image_bytes, prompt)
        print(f"[HybridVerifier] VLM score: {vlm_result.score:.2f}")
        print(vlm_result.format_feedback())

        image_path = self._save_temp_image(image_bytes, iteration)

        if self._sync and self._sync.has_clients():
            return self._hybrid_via_websocket(image_path, prompt, iteration, vlm_result)
        return self._hybrid_via_terminal(image_path, prompt, iteration, vlm_result)

    def _save_temp_image(self, image_bytes: bytes, iteration: int) -> str:
        out_dir = Path(self._output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"comfyclaw_review_iter{iteration}.png"
        path.write_bytes(image_bytes)
        return str(path)

    def _hybrid_via_websocket(
        self, image_path: str, prompt: str, iteration: int, vlm_result: VerifierResult
    ) -> VerifierResult:
        print(f"[HybridVerifier] 📝 Requesting human review via ComfyUI panel…")
        self._sync.request_feedback(
            image_path=image_path,
            vlm_summary=vlm_result.format_feedback(),
            iteration=iteration,
            prompt=prompt,
        )
        feedback = self._sync.wait_for_human_feedback(timeout=self._timeout)

        if feedback is None:
            print("[HybridVerifier] ⏰ Timeout — using VLM result as-is.")
            return vlm_result

        if feedback.get("action") == "accept":
            print("[HybridVerifier] ✅ Human accepted VLM result.")
            return vlm_result

        return self._merge_feedback(vlm_result, feedback)

    def _hybrid_via_terminal(
        self, image_path: str, prompt: str, iteration: int, vlm_result: VerifierResult
    ) -> VerifierResult:
        print(f"\n{'─' * 60}")
        print(f"[HybridVerifier] 📝 Image: {image_path}")
        print(f"[HybridVerifier] VLM score: {vlm_result.score:.2f}")
        print(f"{'─' * 60}")

        if not sys.stdin.isatty():
            print("[HybridVerifier] Non-interactive — using VLM result.")
            return vlm_result

        print("\nAccept VLM result? (Enter = accept, or type score to override):")
        score_raw = input("  Score> ").strip()
        if not score_raw:
            print("[HybridVerifier] ✅ Accepted VLM result.")
            return vlm_result

        try:
            score = float(score_raw)
            score = max(0.0, min(1.0, score))
        except ValueError:
            score = vlm_result.score

        print("Your feedback (Enter to skip):")
        text = input("  Feedback> ").strip()
        if not text:
            vlm_result.score = score
            return vlm_result

        return self._merge_feedback(vlm_result, {"score": score, "text": text})

    @staticmethod
    def _merge_feedback(vlm_result: VerifierResult, feedback: dict) -> VerifierResult:
        """Merge human feedback into the VLM result."""
        text = feedback.get("text", "")
        score = feedback.get("score", vlm_result.score)

        human_suggestions = []
        if text:
            for line in text.replace(";", "\n").split("\n"):
                line = line.strip().lstrip("-•*0123456789.) ")
                if line:
                    human_suggestions.append(f"[HUMAN] {line}")

        vlm_result.score = score
        vlm_result.overall_assessment = (
            f"[Human override] {text}" if text else vlm_result.overall_assessment
        )
        vlm_result.evolution_suggestions = human_suggestions + vlm_result.evolution_suggestions
        vlm_result.feedback_source = "hybrid"
        return vlm_result
