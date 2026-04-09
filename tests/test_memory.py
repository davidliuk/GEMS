"""Unit tests for ClawMemory."""
from __future__ import annotations

from comfyclaw.memory import ClawMemory


def _make_attempt(iteration: int = 1, score: float = 0.5, image: bytes | None = None) -> dict:
    return {
        "iteration": iteration,
        "workflow_snapshot": {"1": {"class_type": "KSampler", "inputs": {}}},
        "verifier_score": score,
        "passed": ["q1"] if score > 0.5 else [],
        "failed": ["q2"] if score < 1.0 else [],
        "experience": f"Attempt {iteration} experience.",
        "image_bytes": image,
    }


class TestRecord:
    def test_record_stores_attempt(self) -> None:
        mem = ClawMemory()
        mem.record(**_make_attempt(1, 0.7))
        assert len(mem) == 1

    def test_multiple_records(self) -> None:
        mem = ClawMemory()
        for i in range(5):
            mem.record(**_make_attempt(i + 1, i * 0.2))
        assert len(mem) == 5


class TestBestAttempt:
    def test_returns_highest_score(self) -> None:
        mem = ClawMemory()
        mem.record(**_make_attempt(1, 0.3))
        mem.record(**_make_attempt(2, 0.8))
        mem.record(**_make_attempt(3, 0.5))
        best = mem.best_attempt()
        assert best is not None
        assert best.iteration == 2
        assert best.verifier_score == 0.8

    def test_returns_none_when_empty(self) -> None:
        mem = ClawMemory()
        assert mem.best_attempt() is None


class TestFormatHistory:
    def test_contains_iteration_and_score(self) -> None:
        mem = ClawMemory()
        mem.record(**_make_attempt(1, 0.6))
        mem.record(**_make_attempt(2, 0.9))
        text = mem.format_history_for_agent()
        assert "Attempt 1" in text
        assert "0.60" in text
        assert "Attempt 2" in text
        assert "0.90" in text

    def test_empty_returns_no_previous(self) -> None:
        mem = ClawMemory()
        assert "No previous" in mem.format_history_for_agent()


class TestLatestExperience:
    def test_returns_last_experience(self) -> None:
        mem = ClawMemory()
        mem.record(**_make_attempt(1, 0.3))
        mem.record(**_make_attempt(2, 0.7))
        assert mem.latest_experience() == "Attempt 2 experience."

    def test_empty_returns_fallback(self) -> None:
        mem = ClawMemory()
        assert "No experience" in mem.latest_experience()


class TestClear:
    def test_clear_empties_attempts(self) -> None:
        mem = ClawMemory()
        mem.record(**_make_attempt(1))
        mem.clear()
        assert len(mem) == 0
        assert mem.best_attempt() is None


class TestImageCap:
    def test_old_images_evicted(self) -> None:
        mem = ClawMemory(max_images=2)
        img = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
        for i in range(5):
            mem.record(**_make_attempt(i + 1, 0.5, image=img))

        images_in_memory = [a for a in mem.attempts if a.image_bytes is not None]
        assert len(images_in_memory) <= 2

    def test_max_images_zero_stores_no_images(self) -> None:
        mem = ClawMemory(max_images=0)
        mem.record(**_make_attempt(1, 0.5, image=b"\xff\xd8\xff"))
        assert mem.attempts[0].image_bytes is None

    def test_latest_images_kept(self) -> None:
        mem = ClawMemory(max_images=2)
        img = b"\xff\xd8\xff"
        for i in range(4):
            mem.record(**_make_attempt(i + 1, 0.5, image=img))
        # Iterations 3 and 4 should still have images
        assert mem.attempts[2].image_bytes is not None
        assert mem.attempts[3].image_bytes is not None
        # Iterations 1 and 2 should be evicted
        assert mem.attempts[0].image_bytes is None
        assert mem.attempts[1].image_bytes is None
