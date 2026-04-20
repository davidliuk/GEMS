"""Tests for the skill-description-writing guide.

Context: evolved-skill descriptions are the ONLY thing the agent sees in
``<available_skills>`` before it decides whether to call ``read_skill``.  In
practice, vague "this is a skill about X" descriptions cause agents to
under-trigger their own learnings.  Following the skill-creator SKILL's
"Description Optimization" guidance, our LLM-facing prompts now teach the
writer to produce pushy, trigger-signal-heavy descriptions.

These tests don't invoke the LLM — they lock in the *prompts* themselves,
since the prompts are the mechanism that shapes every future description.
"""

from __future__ import annotations

import inspect

from comfyclaw import agent as agent_mod
from comfyclaw.evolve import DESCRIPTION_WRITING_GUIDE, _PROPOSE_MUTATION_PROMPT


# ---------------------------------------------------------------------------
# _PROPOSE_MUTATION_PROMPT — the mutation-proposal prompt used by SkillEvolver
# ---------------------------------------------------------------------------


class TestProposeMutationPromptDescriptionGuide:
    """The prompt must actively teach the LLM to write triggering descriptions."""

    def test_includes_description_writing_guide_header(self) -> None:
        assert "Description-writing guide" in _PROPOSE_MUTATION_PROMPT

    def test_names_both_what_and_when_as_required(self) -> None:
        # The two-part structure (capability summary + pushy WHEN clause) is
        # the core of the skill-creator best practice — surface explicitly.
        assert "WHAT" in _PROPOSE_MUTATION_PROMPT
        assert "WHEN" in _PROPOSE_MUTATION_PROMPT

    def test_includes_imperative_trigger_phrases(self) -> None:
        # The LLM should see concrete example phrasings it can mimic.
        for phrase in ("Consult when", "Read this before", "Apply whenever"):
            assert phrase in _PROPOSE_MUTATION_PROMPT, (
                f"prompt is missing imperative trigger phrasing: {phrase!r}"
            )

    def test_warns_about_undertriggering(self) -> None:
        # Explicit under-trigger callout mirrors the skill-creator wisdom.
        assert "under-trigger" in _PROPOSE_MUTATION_PROMPT.lower()

    def test_contains_good_and_bad_examples(self) -> None:
        # Concrete GOOD + BAD examples are the most reliable way to move
        # description quality.  Lock them in so future edits don't drop them.
        assert "GOOD" in _PROPOSE_MUTATION_PROMPT
        assert "BAD" in _PROPOSE_MUTATION_PROMPT

    def test_tag_guidance_still_present(self) -> None:
        # Regression guard: the new description section must not displace
        # the existing tag guidance that the runner relies on.
        assert "Tag guidance:" in _PROPOSE_MUTATION_PROMPT
        assert "model:<short>" in _PROPOSE_MUTATION_PROMPT

    def test_format_placeholders_unchanged(self) -> None:
        # Any call site that .format()s this prompt must still work.
        rendered = _PROPOSE_MUTATION_PROMPT.format(
            cluster_json="{}",
            builtin_skills_manifest="(none)",
            evolved_skills_manifest="(none)",
        )
        assert "(none)" in rendered

    def test_mutation_types_still_enumerated(self) -> None:
        for mt in ("create", "update", "merge", "delete", "reinforce"):
            assert mt in _PROPOSE_MUTATION_PROMPT, (
                f"mutation type {mt!r} disappeared from prompt"
            )


# ---------------------------------------------------------------------------
# Agent system prompt — the "proactively consult evolved skills" nudge
# ---------------------------------------------------------------------------


class TestAgentSystemPromptEvolvedSkillsNudge:
    """The agent's system prompt must push it to scan evolved skills up front."""

    def _preamble(self) -> str:
        # _PREAMBLE_TEMPLATE is a module-level constant on ClawAgent.  We
        # read it through the module so the test tracks text-level edits.
        # Fall back to rendering a minimal system prompt if the constant
        # name ever changes, so the test can still find the nudge.
        pre = getattr(agent_mod, "_PREAMBLE_TEMPLATE", None)
        if pre is not None:
            return pre
        return agent_mod._build_system_prompt(  # type: ignore[attr-defined]
            pinned_image_model=None, available_skills_xml=""
        )

    def test_has_proactive_consultation_section(self) -> None:
        preamble = self._preamble()
        assert "Proactively consulting evolved skills" in preamble

    def test_mentions_trigger_scanning(self) -> None:
        preamble = self._preamble()
        # The core behaviour we want: scan descriptions for matching triggers.
        low = preamble.lower()
        assert "trigger" in low
        assert "read_skill" in preamble

    def test_warns_against_under_reading(self) -> None:
        # Agents tend to under-trigger skills; make the cost explicit.
        preamble = self._preamble()
        low = preamble.lower()
        assert "under-reading" in low or "under reading" in low


# ---------------------------------------------------------------------------
# Shared DESCRIPTION_WRITING_GUIDE — used by *every* SKILL.md writer
# ---------------------------------------------------------------------------


class TestSharedDescriptionWritingGuide:
    """Every code path that synthesises a SKILL.md description must use the
    shared guide — otherwise different writers drift out of sync and some
    produce vague, un-triggering descriptions again.
    """

    def test_guide_is_substantial(self) -> None:
        # >=300 chars is a soft lower bound — the guide must cover WHAT/WHEN,
        # imperative phrasings, GOOD+BAD examples, and length targets.
        assert len(DESCRIPTION_WRITING_GUIDE) > 300

    def test_guide_covers_both_what_and_when(self) -> None:
        assert "WHAT" in DESCRIPTION_WRITING_GUIDE
        assert "WHEN" in DESCRIPTION_WRITING_GUIDE

    def test_guide_shows_good_and_bad_examples(self) -> None:
        assert "GOOD" in DESCRIPTION_WRITING_GUIDE
        assert "BAD" in DESCRIPTION_WRITING_GUIDE

    def test_propose_mutation_prompt_embeds_the_guide(self) -> None:
        # Regression guard: if someone inlines their own copy in the
        # mutation prompt and the shared guide drifts, tests must catch it.
        assert DESCRIPTION_WRITING_GUIDE in _PROPOSE_MUTATION_PROMPT

    def test_run_benchmark_synthesizers_import_the_guide(self) -> None:
        # Both learned-errors and learned-successes paths in
        # experiments/run_benchmark.py must thread the same guide through
        # to their LLM prompt.
        from experiments import run_benchmark as rb

        src_err = inspect.getsource(rb.synthesize_learned_skill)
        src_ok = inspect.getsource(rb.synthesize_success_patterns)
        for src, label in [(src_err, "synthesize_learned_skill"),
                           (src_ok, "synthesize_success_patterns")]:
            assert "DESCRIPTION_WRITING_GUIDE" in src, (
                f"{label} no longer uses the shared description-writing guide; "
                "its generated SKILL.md descriptions will drift back to vague."
            )
