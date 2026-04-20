"""
ClawVerifier — LLM vision verifier with region-level spatial analysis.

Verification pipeline
---------------------
1. ``_decompose_prompt``  — ask the LLM to break the prompt into yes/no questions.
2. ``_check_requirements`` — ask the LLM yes/no for each question in parallel
                              (single base64 encode, shared across all threads).
3. ``_detailed_analysis``  — single-pass request returning region issues,
                              evolution suggestions, and a holistic quality score.
4. Blend the requirement score and the detail score into a final ``VerifierResult``.

Any vision-capable model supported by LiteLLM can be used — Anthropic Claude,
OpenAI GPT-4o, Google Gemini, local LLaVA via Ollama, etc.
"""

from __future__ import annotations

import base64
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import litellm

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
    feedback_source: str = "vlm"  # "vlm", "human", or "hybrid"

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

_DECOMPOSE_AND_CHECK_PROMPT = """\
You are verifying a generated image against its prompt.

1. Decompose the prompt into specific, observable visual requirements.
2. For each requirement, write a yes/no question and answer it based on the image.

Respond ONLY with a JSON array of objects:
[{{"question": "<yes/no question>", "answer": "yes" or "no"}}]

Prompt: {prompt}"""

_DETAILED_ANALYSIS_PROMPT = """\
You are an expert image quality analyst and ComfyUI workflow engineer.

Analyze this generated image against the intended prompt, then return a JSON object with:
{{
  "overall_assessment": "<1–2 sentence overall quality summary>",
  "score": <integer 1–10>,
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

Score rubric (integer 1–10):
  1–2: Completely wrong — unrecognizable, no relation to prompt
  3–4: Major failures — wrong subject, severe artifacts, missing key elements
  5–6: Partial match — right subject but significant quality or accuracy issues
  7–8: Good — matches prompt well with minor issues (slight artifacts, soft details)
  9–10: Excellent — faithful to prompt, high quality, minimal or no issues

Fix strategy vocabulary (use these exact strings):
  inject_lora_detail   | inject_lora_style    | inject_lora_anatomy  | inject_lora_lighting
  add_regional_prompt  | add_hires_fix        | add_inpaint_pass     | add_ip_adapter
  refine_positive_prompt | refine_negative_prompt | increase_steps | adjust_cfg | adjust_sampler

Intended prompt: {prompt}
"""


_UNIFIED_VERIFY_PROMPT = """\
You are an expert image quality analyst and ComfyUI workflow engineer.

You are given one generated image and the intended prompt. Do BOTH tasks in a
SINGLE pass and return ONE JSON object. Do not emit any prose, markdown, or
code fences — just the JSON.

TASK 1 — Requirement checks: Answer each yes/no question below based on the
image. Use strict "yes" / "no" answers (lower-case). Every question MUST be
answered; never skip.

Questions (answer all):
{questions_block}

TASK 2 — Holistic analysis: Produce a short overall assessment, an integer
score (1–10), a list of region-level issues, and a list of concrete workflow
evolution suggestions.

Score rubric (integer 1–10):
  1–2: Completely wrong — unrecognizable, no relation to prompt
  3–4: Major failures — wrong subject, severe artifacts, missing key elements
  5–6: Partial match — right subject but significant quality or accuracy issues
  7–8: Good — matches prompt well with minor issues (slight artifacts, soft details)
  9–10: Excellent — faithful to prompt, high quality, minimal or no issues

Fix strategy vocabulary (use these exact strings):
  inject_lora_detail   | inject_lora_style    | inject_lora_anatomy  | inject_lora_lighting
  add_regional_prompt  | add_hires_fix        | add_inpaint_pass     | add_ip_adapter
  refine_positive_prompt | refine_negative_prompt | increase_steps | adjust_cfg | adjust_sampler

Return exactly this JSON schema:
{{
  "requirements": [
    {{"question": "<verbatim question text>", "answer": "yes" or "no"}}
  ],
  "overall_assessment": "<1–2 sentence overall quality summary>",
  "score": <integer 1–10>,
  "region_issues": [
    {{
      "region": "<foreground subject | background | face | hands | sky | texture surface | etc.>",
      "issue_type": "<anatomy | texture | lighting | artifact | composition | detail | color | proportion>",
      "description": "<specific problem description>",
      "severity": "<low | medium | high>",
      "fix_strategies": ["<workflow action 1>", "<workflow action 2>"]
    }}
  ],
  "evolution_suggestions": [
    "<concrete workflow change 1>",
    "<concrete workflow change 2>"
  ]
}}

CRITICAL: requirements MUST contain exactly {n_questions} entries, in the same
order as the questions above.

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
    api_key       : API key for the LLM provider.  Written into the appropriate
                    env-var so LiteLLM can pick it up automatically.  You can
                    also set the env-var directly and leave this empty.
    model         : LiteLLM model string (must support vision), e.g.
                    ``"anthropic/claude-sonnet-4-5"``, ``"openai/gpt-4o"``,
                    ``"gemini/gemini-2.0-flash"``, ``"ollama/llava"``.
    score_weights : ``(requirement_weight, detail_weight)`` summing to 1.0.
                    Defaults to ``(0.6, 0.4)``.
    max_workers   : Parallel threads for yes/no requirement checks (legacy path).
    batch_mode    : When ``True`` (default), ``verify()`` collapses the N
                    per-question yes/no vision calls AND the detailed-analysis
                    call into ONE unified vision call (see
                    :meth:`_unified_verify_call`).  Cost drops from ``2 + N``
                    LLM calls per verify (with N image re-uploads) to ``1 + 1``
                    calls (image uploaded once).  The text-only decomposition
                    stays cached per-prompt so iterations 2+ pay 0 extra tokens
                    for question generation.  Set to ``False`` to restore the
                    legacy parallel-per-question path.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "anthropic/claude-sonnet-4-5",
        score_weights: tuple[float, float] = (0.6, 0.4),
        max_workers: int = 6,
        multi_scale: bool = False,
        weighted_requirements: bool = False,
        batch_mode: bool = True,
        decompose_model: str | None = None,
    ) -> None:
        if api_key:
            from .agent import _set_llm_api_key
            _set_llm_api_key(api_key, model)
        self.model = model
        # P2/item 4: the prompt-decomposition step is a pure-text call that
        # turns a user prompt into a list of yes/no verifier questions.  It
        # doesn't need the flagship VLM model; a cheaper text-only model
        # (Haiku, GPT-4o-mini, …) is a drop-in replacement that saves ~3s
        # of first-verify latency and ~95% of input tokens for the decompose
        # step.  None means "use self.model", preserving legacy behaviour.
        self.decompose_model: str = decompose_model or model
        self.score_weights = score_weights
        self.max_workers = max_workers
        self.multi_scale = multi_scale
        self.weighted_requirements = weighted_requirements
        self.batch_mode = batch_mode
        self._decomposition_cache: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, image_bytes: bytes, prompt: str, iteration: int = 0) -> VerifierResult:
        """
        Run full verification: requirement checks + detailed region analysis.

        Parameters
        ----------
        image_bytes : Raw image bytes (PNG or JPEG).
        prompt      : The original text prompt used to generate the image.
        """
        b64_data = base64.standard_b64encode(image_bytes).decode()
        media_type = _detect_media_type(image_bytes)

        questions = self._decompose_prompt(prompt)
        if not questions:
            questions = ["Does the image match the described scene?"]

        # Batched path: one vision call returns both per-question yes/no
        # answers AND the detailed analysis.  This cuts total LLM calls
        # from (2 + N) to (1 + 1) and uploads the image once instead of
        # (N + 1) times.  getattr() keeps legacy tests (which build the
        # verifier via __new__ and never set batch_mode) on the old path.
        if getattr(self, "batch_mode", False):
            checks, detail = self._unified_verify_call(
                b64_data, media_type, prompt, questions
            )
            # Fallback: if the unified call fails to parse (rare provider
            # output), drop back to the legacy parallel path so we still
            # produce a useful VerifierResult instead of an all-zero one.
            if not checks:
                checks = self._check_requirements(b64_data, media_type, questions)
                detail = self._detailed_analysis(b64_data, media_type, prompt)
        else:
            checks = self._check_requirements(b64_data, media_type, questions)
            detail = self._detailed_analysis(b64_data, media_type, prompt)

        passed = [c.question for c in checks if c.passed]
        failed = [c.question for c in checks if not c.passed]

        if self.weighted_requirements:
            from .verifier_utils import weighted_requirement_score

            req_score = weighted_requirement_score(
                [c.question for c in checks],
                [c.passed for c in checks],
            )
        else:
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

        # Multi-scale: run additional checks on key regions
        if self.multi_scale and image_bytes:
            region_checks = self._multi_scale_verify(image_bytes, prompt, media_type)
            region_issues.extend(region_checks)

        return VerifierResult(
            score=round(score, 3),
            checks=checks,
            passed=passed,
            failed=failed,
            region_issues=region_issues,
            overall_assessment=detail.get("overall_assessment", ""),
            evolution_suggestions=detail.get("evolution_suggestions", []),
        )

    def verify_comparative(
        self,
        image_a: bytes,
        image_b: bytes,
        prompt: str,
    ) -> dict:
        """Compare two images and return which is better and why.

        Returns a dict with ``winner`` ("A" or "B"), ``confidence``,
        ``reason``, and per-image strengths/weaknesses.
        """
        from .verifier_utils import (
            build_comparative_message,
            encode_image_b64,
            parse_comparative_result,
        )

        a_b64 = encode_image_b64(image_a)
        b_b64 = encode_image_b64(image_b)
        media_type = _detect_media_type(image_a)

        messages = build_comparative_message(a_b64, b_b64, prompt, media_type)
        try:
            resp = litellm.completion(
                model=self.model,
                max_tokens=1024,
                messages=messages,
            )
            text = (resp.choices[0].message.content or "").strip()
            return parse_comparative_result(text)
        except Exception as exc:
            return {"winner": "A", "confidence": 0.5, "reason": f"Error: {exc}"}

    def _multi_scale_verify(
        self,
        image_bytes: bytes,
        prompt: str,
        media_type: str,
    ) -> list[RegionIssue]:
        """Verify key regions at higher zoom for granular feedback."""
        from .verifier_utils import crop_region, encode_image_b64

        regions_to_check = ["face_region", "center", "background"]
        issues: list[RegionIssue] = []

        _REGION_CHECK_PROMPT = (
            "Examine this cropped region of a generated image. "
            "The full image was generated from: {prompt}\n"
            "This crop shows the {region} area.\n"
            "List any quality issues (artifacts, anatomy errors, blur, "
            "inconsistencies). Respond with JSON:\n"
            '{{"issues": [{{"issue": "description", "severity": "low|medium|high"}}]}}\n'
            "If no issues, respond: {{\"issues\": []}}"
        )

        for region_name in regions_to_check:
            try:
                cropped = crop_region(image_bytes, region_name)
                b64 = encode_image_b64(cropped)
                image_block = {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{b64}"},
                }
                resp = litellm.completion(
                    model=self.model,
                    max_tokens=512,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                image_block,
                                {
                                    "type": "text",
                                    "text": _REGION_CHECK_PROMPT.format(
                                        prompt=prompt, region=region_name,
                                    ),
                                },
                            ],
                        }
                    ],
                )
                text = (resp.choices[0].message.content or "").strip()
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if m:
                    data = json.loads(m.group())
                    for iss in data.get("issues", []):
                        issues.append(
                            RegionIssue(
                                region=region_name,
                                issue_type="detail",
                                description=iss.get("issue", ""),
                                severity=iss.get("severity", "medium"),
                                fix_strategies=[],
                            )
                        )
            except Exception:
                pass

        return issues

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def complete(self, prompt: str, max_tokens: int = 200) -> str:
        """
        Plain text completion — no image, no tools.

        Used by the harness for lightweight summarisation tasks.
        """
        resp = litellm.completion(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()

    def _decompose_prompt(self, prompt: str) -> list[str]:
        cached = self._decomposition_cache.get(prompt)
        if cached is not None:
            return cached

        resp = litellm.completion(
            # P2/item 4: routed through ``decompose_model`` (defaults to the
            # main verifier model) so benchmarks can swap in a cheap text
            # model for this pure-text step without downgrading the vision
            # calls.
            model=getattr(self, "decompose_model", self.model),
            max_tokens=1024,
            messages=[{"role": "user", "content": _DECOMPOSE_PROMPT.format(prompt=prompt)}],
        )
        text = (resp.choices[0].message.content or "").strip()
        try:
            m = re.search(r"\[.*\]", text, re.DOTALL)
            questions = json.loads(m.group() if m else text)
        except Exception:
            questions = [ln.strip() for ln in text.splitlines() if "?" in ln]

        self._decomposition_cache[prompt] = questions
        return questions

    def _decompose_and_check(
        self,
        b64_data: str,
        media_type: str,
        prompt: str,
    ) -> list[RequirementCheck]:
        """Decompose the prompt AND check each requirement in a single vision LLM call."""
        image_block = {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64_data}"},
        }
        try:
            resp = litellm.completion(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            image_block,
                            {
                                "type": "text",
                                "text": _DECOMPOSE_AND_CHECK_PROMPT.format(prompt=prompt),
                            },
                        ],
                    }
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            m = re.search(r"\[.*\]", text, re.DOTALL)
            items = json.loads(m.group() if m else text)
            checks = []
            for item in items:
                q = item.get("question", "")
                a = str(item.get("answer", "no")).strip().lower()
                checks.append(RequirementCheck(q, a, "yes" in a and "no" not in a))
            return checks
        except Exception:
            return []

    def _unified_verify_call(
        self,
        b64_data: str,
        media_type: str,
        prompt: str,
        questions: list[str],
    ) -> tuple[list[RequirementCheck], dict]:
        """One vision call returning both per-question answers AND detail dict.

        Returns ``([], {})`` on any parse / network failure so the caller can
        cleanly fall back to the legacy per-question path.
        """
        image_block = {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64_data}"},
        }
        questions_block = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(questions))

        try:
            resp = litellm.completion(
                model=self.model,
                max_tokens=2000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            image_block,
                            {
                                "type": "text",
                                "text": _UNIFIED_VERIFY_PROMPT.format(
                                    questions_block=questions_block,
                                    n_questions=len(questions),
                                    prompt=prompt,
                                ),
                            },
                        ],
                    }
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            # Tolerate providers wrapping the JSON in ```json ... ``` fences.
            m = re.search(r"\{.*\}", text, re.DOTALL)
            data = json.loads(m.group() if m else text)
        except Exception:
            return [], {}

        raw_reqs = data.get("requirements") or []
        if not isinstance(raw_reqs, list) or not raw_reqs:
            return [], {}

        # Align answers to the original question list.  Prefer positional
        # matching (same length, same order), fall back to question-text
        # match so a chatty model rewriting a question doesn't desync us.
        by_text = {
            str(item.get("question", "")).strip(): item
            for item in raw_reqs
            if isinstance(item, dict)
        }
        checks: list[RequirementCheck] = []
        for i, q in enumerate(questions):
            if i < len(raw_reqs) and isinstance(raw_reqs[i], dict):
                item = raw_reqs[i]
            else:
                item = by_text.get(q.strip(), {})
            ans = str(item.get("answer", "")).strip().lower()
            # A valid yes/no answer contains "yes" xor "no".  Anything else
            # (empty / "unsure" / "maybe") counts as failed so we conservatively
            # ask the agent to iterate.
            passed = ("yes" in ans) and ("no" not in ans)
            checks.append(RequirementCheck(q, ans or "no", passed))

        # Normalise 1–10 score into 0–1 (same logic as legacy path).
        detail = {
            "overall_assessment": data.get("overall_assessment", ""),
            "score": data.get("score"),
            "region_issues": data.get("region_issues", []) or [],
            "evolution_suggestions": data.get("evolution_suggestions", []) or [],
        }
        raw = detail["score"]
        if raw is not None:
            try:
                raw = float(raw)
                if raw > 1.0:
                    detail["score"] = max(0.0, min(1.0, (raw - 1) / 9.0))
                else:
                    detail["score"] = max(0.0, min(1.0, raw))
            except (TypeError, ValueError):
                detail["score"] = None

        return checks, detail

    def _check_requirements(
        self,
        b64_data: str,
        media_type: str,
        questions: list[str],
    ) -> list[RequirementCheck]:
        """Check each question in parallel using pre-encoded image data."""

        # OpenAI-style image_url with base64 data URI — LiteLLM translates
        # this to the provider-specific format (Anthropic, Gemini, etc.).
        image_block = {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64_data}"},
        }

        def check_one(q: str) -> RequirementCheck:
            try:
                resp = litellm.completion(
                    model=self.model,
                    max_tokens=16,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                image_block,
                                {"type": "text", "text": f"Answer only 'yes' or 'no'. {q}"},
                            ],
                        }
                    ],
                )
                ans = (resp.choices[0].message.content or "").strip().lower()
                return RequirementCheck(q, ans, "yes" in ans and "no" not in ans)
            except Exception as exc:
                return RequirementCheck(q, f"error: {exc}", False)

        n_workers = min(len(questions), self.max_workers)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            return list(ex.map(check_one, questions))

    def _detailed_analysis(self, b64_data: str, media_type: str, prompt: str) -> dict:
        image_block = {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64_data}"},
        }
        try:
            resp = litellm.completion(
                model=self.model,
                max_tokens=1500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            image_block,
                            {
                                "type": "text",
                                "text": _DETAILED_ANALYSIS_PROMPT.format(prompt=prompt),
                            },
                        ],
                    }
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            m = re.search(r"\{.*\}", text, re.DOTALL)
            data = json.loads(m.group() if m else text)

            raw = data.get("score")
            if raw is not None:
                raw = float(raw)
                if raw > 1.0:
                    data["score"] = max(0.0, min(1.0, (raw - 1) / 9.0))

            return data
        except Exception as exc:
            return {
                "overall_assessment": f"Analysis error: {exc}",
                "score": None,
                "region_issues": [],
                "evolution_suggestions": [],
            }
