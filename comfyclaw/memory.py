"""
ClawMemory — short-term per-run attempt history.

Stores workflow snapshots, verifier scores, and compressed lessons so the
agent can avoid repeating mistakes across iterations within a single run.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Attempt:
    iteration: int
    workflow_snapshot: dict
    verifier_score: float       # 0.0 – 1.0
    passed: list[str]
    failed: list[str]
    experience: str             # ≤80-word compressed lesson
    image_bytes: bytes | None = field(default=None, repr=False)


class ClawMemory:
    """
    Short-term memory: one instance per ``ClawHarness.run()`` call.

    Parameters
    ----------
    max_images : Maximum number of attempt image blobs to keep in RAM.
                 Older images are evicted (set to ``None``) when the cap is
                 exceeded. Use 0 to disable image storage entirely.
    """

    def __init__(self, max_images: int = 5) -> None:
        self.attempts: list[Attempt] = []
        self.max_images = max_images

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(
        self,
        iteration: int,
        workflow_snapshot: dict,
        verifier_score: float,
        passed: list[str],
        failed: list[str],
        experience: str,
        image_bytes: bytes | None = None,
    ) -> None:
        self.attempts.append(
            Attempt(
                iteration=iteration,
                workflow_snapshot=workflow_snapshot,
                verifier_score=verifier_score,
                passed=passed,
                failed=failed,
                experience=experience,
                image_bytes=image_bytes if self.max_images > 0 else None,
            )
        )
        self._evict_old_images()

    def _evict_old_images(self) -> None:
        """Keep only the most recent ``max_images`` blobs in RAM."""
        if self.max_images <= 0:
            return
        images_kept = 0
        for attempt in reversed(self.attempts):
            if attempt.image_bytes is not None:
                images_kept += 1
                if images_kept > self.max_images:
                    attempt.image_bytes = None

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def best_attempt(self) -> Attempt | None:
        """Return the attempt with the highest verifier score."""
        if not self.attempts:
            return None
        return max(self.attempts, key=lambda a: a.verifier_score)

    def format_history_for_agent(self) -> str:
        """Compact text summary for injection into an agent prompt."""
        if not self.attempts:
            return "No previous attempts."
        lines: list[str] = []
        for a in self.attempts:
            lines.append(
                f"Attempt {a.iteration} (score={a.verifier_score:.2f}):\n"
                f"  Passed: {', '.join(a.passed) or 'none'}\n"
                f"  Failed: {', '.join(a.failed) or 'none'}\n"
                f"  Experience: {a.experience}"
            )
        return "\n\n".join(lines)

    def latest_experience(self) -> str:
        if not self.attempts:
            return "No experience yet."
        return self.attempts[-1].experience

    def clear(self) -> None:
        self.attempts.clear()

    def __len__(self) -> int:
        return len(self.attempts)

    def __repr__(self) -> str:
        scores = [f"{a.verifier_score:.2f}" for a in self.attempts]
        return f"<ClawMemory attempts={len(self)} scores=[{', '.join(scores)}]>"
