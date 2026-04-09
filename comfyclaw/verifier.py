"""
ClawVerifier — Claude vision verifier with region-level spatial analysis.

Verification pipeline
---------------------
1. ``_decompose_prompt``  — ask Claude to break the prompt into yes/no questions.
2. ``_check_requirements`` — ask Claude yes/no for each question in parallel
                              (single base64 encode, shared across all threads).
3. ``_detailed_analysis``  — single-pass request returning region issues,
                              evolution suggestions, and a holistic quality score.
4. Blend the requirement score and the detail score into a final ``VerifierResult``.
"""

from __future__ import annotations

import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import anthropic

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RequirementCheck:
    question: str
    answer: str
    passed: bool


@dataclass
class RegionIssue:
    """A quality issue localised to a spatial region of the image."""

    region: str  # e.g. "background", "hands", "face"
    issue_type: str  # anatomy | texture | lighting | artifact | composition | detail
    description: str
    severity: str  # low | medium | high
    fix_strategies: list[str]


@dataclass
class VerifierResult:
    """Full verification report returned to ClawHarness."""

    score: float  # 0.0 – 1.0 blended quality score
    checks: list[RequirementCheck]
    passed: list[str]
    failed: list[str]
    region_issues: list[RegionIssue] = field(default_factory=list)
    overall_assessment: str = ""
    evolution_suggestions: list[str] = field(default_factory=list)

    def format_feedback(self) -> str:
        lines = [f"Overall score: {self.score:.2f}  |  {self.overall_assessment}"]
        if self.passed:
            lines.append(f"✅ Passed ({len(self.passed)}): {'; '.join(self.passed)}")
        if self.failed:
            lines.append(f"❌ Failed ({len(self.failed)}): {'; '.join(self.failed)}")
        if self.region_issues:
            lines.append("\n── Region Issues ──")
            for ri in self.region_issues:
                lines.append(
                    f"  [{ri.severity.upper()}] {ri.region} — {ri.issue_type}: {ri.description}"
                )
                lines.append(f"    Fix strategies: {', '.join(ri.fix_strategies)}")
        if self.evolution_suggestions:
            lines.append("\n── Suggested Workflow Evolutions ──")
            for i, s in enumerate(self.evolution_suggestions, 1):
                lines.append(f"  {i}. {s}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_DECOMPOSE_PROMPT = (
    "Analyze the following image generation prompt and break it down into specific, "
    "observable visual requirements. For each, write a yes/no question answerable from the image.\n\n"
    "Respond ONLY with a JSON array of question strings.\n\nPrompt: {prompt}"
)

_DETAILED_ANALYSIS_PROMPT = """\
You are an expert image quality analyst and ComfyUI workflow engineer.

Analyze this generated image against the intended prompt, then return a JSON object with:
{{
  "overall_assessment": "<1–2 sentence overall quality summary>",
  "score": <0.0–1.0 quality score>,
  "region_issues": [
    {{
      "region": "<specific area: foreground subject | background | face | hands | sky | texture surface | etc.>",
      "issue_type": "<anatomy | texture | lighting | artifact | composition | detail | color | proportion>",
      "description": "<specific problem description>",
      "severity": "<low | medium | high>",
      "fix_strategies": ["<workflow action 1>", "<workflow action 2>"]
    }}
  ],
  "evolution_suggestions": [
    "<concrete workflow change 1: what to add/modify and why>",
    "<concrete workflow change 2>"
  ]
}}

Fix strategy vocabulary (use these exact strings):
  add_controlnet_canny | add_controlnet_depth | add_controlnet_normal
  add_controlnet_pose  | add_controlnet_tile  | add_controlnet_seg
  inject_lora_detail   | inject_lora_style    | inject_lora_anatomy  | inject_lora_lighting
  add_regional_prompt  | add_hires_fix        | add_inpaint_pass     | add_ip_adapter
  refine_positive_prompt | refine_negative_prompt | increase_steps | adjust_cfg | adjust_sampler

Intended prompt: {prompt}
"""


# ---------------------------------------------------------------------------
# Media-type detection
# ---------------------------------------------------------------------------


def _detect_media_type(image_bytes: bytes) -> str:
    """Detect image MIME type from magic bytes."""
    if image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"  # safe fallback


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class ClawVerifier:
    """
    Parameters
    ----------
    api_key       : Anthropic API key.
    model         : Claude model (must support vision).
    score_weights : ``(requirement_weight, detail_weight)`` summing to 1.0.
                    Defaults to ``(0.6, 0.4)``.
    max_workers   : Parallel threads for yes/no requirement checks.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5",
        score_weights: tuple[float, float] = (0.6, 0.4),
        max_workers: int = 6,
    ) -> None:
        self.client = anthropic.Anthropic(
            api_key=api_key,
            base_url="https://api.anthropic.com",
        )
        self.model = model
        self.score_weights = score_weights
        self.max_workers = max_workers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, image_bytes: bytes, prompt: str) -> VerifierResult:
        """
        Run full verification: requirement checks + detailed region analysis.

        Parameters
        ----------
        image_bytes : Raw image bytes (PNG or JPEG).
        prompt      : The original text prompt used to generate the image.
        """
        # Encode once — shared across all parallel requirement checks
        b64_data = base64.standard_b64encode(image_bytes).decode()
        media_type = _detect_media_type(image_bytes)

        questions = self._decompose_prompt(prompt)
        if not questions:
            questions = ["Does the image match the described scene?"]

        checks = self._check_requirements(b64_data, media_type, questions)
        detail = self._detailed_analysis(b64_data, media_type, prompt)

        passed = [c.question for c in checks if c.passed]
        failed = [c.question for c in checks if not c.passed]
        req_score = len(passed) / len(checks) if checks else 0.0

        detail_score = detail.get("score")
        w_req, w_det = self.score_weights
        if detail_score is not None:
            score = w_req * req_score + w_det * float(detail_score)
        else:
            score = req_score

        region_issues = [
            RegionIssue(
                region=ri.get("region", "unknown"),
                issue_type=ri.get("issue_type", "unknown"),
                description=ri.get("description", ""),
                severity=ri.get("severity", "medium"),
                fix_strategies=ri.get("fix_strategies", []),
            )
            for ri in detail.get("region_issues", [])
        ]

        return VerifierResult(
            score=round(score, 3),
            checks=checks,
            passed=passed,
            failed=failed,
            region_issues=region_issues,
            overall_assessment=detail.get("overall_assessment", ""),
            evolution_suggestions=detail.get("evolution_suggestions", []),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _decompose_prompt(self, prompt: str) -> list[str]:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": _DECOMPOSE_PROMPT.format(prompt=prompt)}],
        )
        text = resp.content[0].text.strip()
        try:
            m = re.search(r"\[.*\]", text, re.DOTALL)
            return json.loads(m.group() if m else text)
        except Exception:
            return [ln.strip() for ln in text.splitlines() if "?" in ln]

    def _check_requirements(
        self,
        b64_data: str,
        media_type: str,
        questions: list[str],
    ) -> list[RequirementCheck]:
        """Check each question in parallel using pre-encoded image data."""

        def check_one(q: str) -> RequirementCheck:
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=16,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": b64_data,
                                    },
                                },
                                {"type": "text", "text": f"Answer only 'yes' or 'no'. {q}"},
                            ],
                        }
                    ],
                )
                ans = resp.content[0].text.strip().lower()
                return RequirementCheck(q, ans, "yes" in ans and "no" not in ans)
            except Exception as exc:
                return RequirementCheck(q, f"error: {exc}", False)

        n_workers = min(len(questions), self.max_workers)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            return list(ex.map(check_one, questions))

    def _detailed_analysis(self, b64_data: str, media_type: str, prompt: str) -> dict:
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": _DETAILED_ANALYSIS_PROMPT.format(prompt=prompt),
                            },
                        ],
                    }
                ],
            )
            text = resp.content[0].text.strip()
            m = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(m.group() if m else text)
        except Exception as exc:
            return {
                "overall_assessment": f"Analysis error: {exc}",
                "score": None,
                "region_issues": [],
                "evolution_suggestions": [],
            }
