"""Phase 2 fixes — trimming the planning loop.

See ``canvases/comfyclaw-pipeline-review.canvas.tsx`` for motivation.  Phase 2
targets the ~40% of per-iteration wall time spent in the agent's plan+tools
LLM loop:

- **Item 1 — system prompt diet**: move the per-model / per-symptom "Decision
  heuristics" table from the always-sent base system prompt into a new
  ``planner-playbook`` built-in skill that the agent reads on demand.
- **Item 2 — parallel tool-call dispatch**: the dispatcher already handles
  multiple ``tool_calls`` in one assistant turn; add a system-prompt nudge so
  the agent actually emits them.
- **Item 3 — per-plan_and_patch skill-read dedup** (B6): when the agent calls
  ``read_skill`` twice for the same skill in one planning turn, return a
  short stub instead of resending the multi-kB body.
- **Item 4 — cheaper decompose model**: route ``ClawVerifier._decompose_prompt``
  through a configurable ``decompose_model`` (defaults to the main verifier
  model — fully backward compatible).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from comfyclaw.agent import ClawAgent, _SYSTEM_PROMPT_BASE, _build_system_prompt
from comfyclaw.harness import HarnessConfig
from comfyclaw.skill_manager import SkillManager
from comfyclaw.verifier import ClawVerifier


# ---------------------------------------------------------------------------
# Item 1 — System prompt diet + planner-playbook skill
# ---------------------------------------------------------------------------


class TestP2SystemPromptDiet:
    """The Decision heuristics table and the "Available workflow tools" list
    have been extracted from the always-sent system prompt.  The prompt now
    points the agent at ``planner-playbook`` for the full decision table and
    relies on the tool JSON-schema for the tool list (which the API sends
    anyway)."""

    def test_decision_table_content_no_longer_inline(self) -> None:
        # The old inline table listed specific model -> skill mappings with
        # ASCII-arrow-style entries; those should now live in the skill.
        assert "read_skill(\"qwen-image-2512\") FIRST" not in _SYSTEM_PROMPT_BASE
        assert "Plasticky skin / poor texture" not in _SYSTEM_PROMPT_BASE
        assert "User asks for creative / artistic" not in _SYSTEM_PROMPT_BASE

    def test_decision_section_points_to_planner_playbook(self) -> None:
        assert "planner-playbook" in _SYSTEM_PROMPT_BASE
        assert 'read_skill("planner-playbook")' in _SYSTEM_PROMPT_BASE

    def test_available_tools_verbose_list_removed(self) -> None:
        # Lines like "inspect_workflow          — view all nodes, IDs..."
        # are redundant with the JSON tool schema.  They should be gone.
        assert "inspect_workflow          — view all nodes" not in _SYSTEM_PROMPT_BASE
        assert "query_available_models    — list checkpoints" not in _SYSTEM_PROMPT_BASE

    def test_prompt_is_shorter_than_pre_phase2_baseline(self) -> None:
        # Pre-Phase-2 baseline was ~10_849 chars.  We expect at least a 15%
        # shrink — lock it in so future edits don't silently re-bloat it.
        assert len(_SYSTEM_PROMPT_BASE) < 10_000, (
            f"System prompt regressed to {len(_SYSTEM_PROMPT_BASE)} chars; "
            "Phase 2 target is < 10k chars (pre-Phase-2 was 10,849)."
        )

    def test_node_parameter_constraints_remain_inline(self) -> None:
        # These are HTTP-400-causing gotchas that fire rarely but burn a whole
        # iteration when they do.  Cheap insurance: keep them in the always-
        # sent prompt even though a fuller version is also in planner-playbook.
        assert "weight_dtype" in _SYSTEM_PROMPT_BASE
        assert "fp8_e4m3fn" in _SYSTEM_PROMPT_BASE
        assert "LoraLoaderModelOnly" in _SYSTEM_PROMPT_BASE

    def test_planner_playbook_skill_is_loadable(self, tmp_path: Path) -> None:
        # The skill must actually exist under the built-in skills dir so the
        # ``read_skill("planner-playbook")`` the prompt tells the agent to
        # call will succeed.
        skills_root = Path(__file__).resolve().parent.parent / "comfyclaw" / "skills"
        sm = SkillManager(skills_root, evolved_skills_dir=None)
        assert "planner-playbook" in sm.skill_names
        body = sm.get_body("planner-playbook")
        # The body must contain the core architecture-specific entries the
        # agent relied on pre-diet.
        assert "qwen-image-2512" in body
        assert "longcat-image" in body
        assert "z-image-turbo" in body
        assert "lora-enhancement" in body

    def test_planner_playbook_description_is_triggerable(self) -> None:
        skills_root = Path(__file__).resolve().parent.parent / "comfyclaw" / "skills"
        sm = SkillManager(skills_root, evolved_skills_dir=None)
        props = sm.get_properties("planner-playbook")
        desc = (props.description or "").lower()
        # Must pull the agent in when uncertain — otherwise the diet is a
        # silent regression.
        assert "decision" in desc or "index" in desc or "playbook" in desc
        assert "first" in desc or "before" in desc or "unsure" in desc


# ---------------------------------------------------------------------------
# Item 2 — Parallel tool-call hint
# ---------------------------------------------------------------------------


class TestP2ParallelToolCallHint:
    """The dispatcher always handled multiple ``tool_calls`` per assistant
    turn; the agent just never emitted them because nothing in the prompt
    invited it to.  A short nudge should tip the behaviour."""

    def test_prompt_mentions_parallel_tool_calls(self) -> None:
        p = _SYSTEM_PROMPT_BASE.lower()
        assert "parallel" in p or "single assistant turn" in p
        assert "tool_calls" in p or "multiple" in p

    def test_prompt_warns_against_batching_dependent_calls(self) -> None:
        # The agent must NOT batch add_node + connect_nodes, because the new
        # id isn't known until add_node returns.  The hint spells this out.
        p = _SYSTEM_PROMPT_BASE
        assert "add_node" in p and "connect_nodes" in p
        assert "do not" in p.lower() or "not batch" in p.lower() or "don't" in p.lower()


# ---------------------------------------------------------------------------
# Item 3 — Per-plan_and_patch skill read dedup (B6)
# ---------------------------------------------------------------------------


class TestP2SkillReadDedup:
    """Within a single ``plan_and_patch`` call, duplicate ``read_skill``
    invocations should return a short stub instead of re-sending the full
    skill body.  Cross-iteration re-reads (different ``plan_and_patch``
    calls) must still load the body, because the messages array is rebuilt
    from scratch each iteration and the agent would otherwise lose it."""

    def _mk_agent_with_stub_skill_manager(self, body: str = "FULL BODY" * 200) -> ClawAgent:
        agent = ClawAgent.__new__(ClawAgent)
        agent.skill_manager = MagicMock()
        agent.skill_manager.get_body = MagicMock(return_value=body)
        agent.skill_manager.skill_names = ["regional-control"]
        agent.skills_read = []
        agent._loaded_skill_bodies = set()
        return agent

    def test_first_read_returns_full_body(self) -> None:
        agent = self._mk_agent_with_stub_skill_manager(body="BIG " * 500)
        out = agent._read_skill("regional-control")
        assert "BIG BIG" in out
        assert "regional-control" in out
        assert "regional-control" in agent._loaded_skill_bodies

    def test_duplicate_read_returns_short_stub(self) -> None:
        agent = self._mk_agent_with_stub_skill_manager(body="BIG " * 500)
        first = agent._read_skill("regional-control")
        second = agent._read_skill("regional-control")
        assert len(second) < len(first)
        assert "already loaded" in second.lower() or "scroll up" in second.lower()
        # Must NOT contain the full body the second time.
        assert "BIG BIG BIG BIG BIG" not in second

    def test_dedup_scoped_to_single_plan_and_patch(self) -> None:
        # Simulate what plan_and_patch does between iterations: resets the
        # dedup set.  The second iteration must pay full-body cost again
        # because the messages array was rebuilt.
        agent = self._mk_agent_with_stub_skill_manager(body="BIG " * 500)
        first_iter_out = agent._read_skill("regional-control")
        agent._loaded_skill_bodies = set()
        second_iter_out = agent._read_skill("regional-control")
        assert "BIG BIG" in second_iter_out
        assert len(second_iter_out) == len(first_iter_out)

    def test_different_skills_do_not_shadow_each_other(self) -> None:
        agent = self._mk_agent_with_stub_skill_manager(body="BIG " * 500)
        agent.skill_manager.skill_names = ["regional-control", "hires-fix"]
        out1 = agent._read_skill("regional-control")
        out2 = agent._read_skill("hires-fix")
        assert "BIG BIG" in out1
        assert "BIG BIG" in out2

    def test_missing_skill_still_returns_error_not_stub(self) -> None:
        agent = self._mk_agent_with_stub_skill_manager()
        agent.skill_manager.get_body = MagicMock(side_effect=KeyError("nope"))
        agent.skill_manager.skill_names = ["regional-control"]
        out = agent._read_skill("does-not-exist")
        assert "not found" in out
        # And "does-not-exist" should NOT be marked as loaded — otherwise the
        # next retry would silently swallow the error.
        assert "does-not-exist" not in agent._loaded_skill_bodies

    def test_skills_read_still_appends_even_on_dedup(self) -> None:
        # The cross-iteration skills_read list is used by the harness to tell
        # the evolve step what the agent consulted.  Dedupped re-reads still
        # count as "the agent asked for this skill" — otherwise the evolve
        # step underestimates skill usage.
        agent = self._mk_agent_with_stub_skill_manager()
        agent._read_skill("regional-control")
        agent._read_skill("regional-control")
        assert agent.skills_read.count("regional-control") == 2


# ---------------------------------------------------------------------------
# Item 4 — Cheaper decompose model
# ---------------------------------------------------------------------------


class TestP2CheaperDecomposeModel:
    """``_decompose_prompt`` is a pure-text LLM call (prompt -> yes/no
    questions).  It doesn't need the flagship VLM; benchmarks can swap in a
    cheap model via ``decompose_model``.  If not set, falls back to the main
    verifier model."""

    def test_default_decompose_model_falls_back_to_main_model(self) -> None:
        v = ClawVerifier(api_key="", model="anthropic/claude-sonnet-4-5")
        assert v.decompose_model == "anthropic/claude-sonnet-4-5"

    def test_explicit_decompose_model_is_honoured(self) -> None:
        v = ClawVerifier(
            api_key="",
            model="anthropic/claude-sonnet-4-5",
            decompose_model="openai/gpt-4o-mini",
        )
        assert v.decompose_model == "openai/gpt-4o-mini"

    def test_none_decompose_model_falls_back(self) -> None:
        v = ClawVerifier(
            api_key="",
            model="anthropic/claude-sonnet-4-5",
            decompose_model=None,
        )
        assert v.decompose_model == "anthropic/claude-sonnet-4-5"

    def test_decompose_routes_through_decompose_model(self) -> None:
        v = ClawVerifier(
            api_key="",
            model="anthropic/claude-sonnet-4-5",
            decompose_model="openai/gpt-4o-mini",
        )

        captured: dict = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = '["Is there a fox?"]'
            return resp

        with patch("comfyclaw.verifier.litellm.completion", side_effect=fake_completion):
            out = v._decompose_prompt("a red fox in a forest")

        assert out == ["Is there a fox?"]
        assert captured["model"] == "openai/gpt-4o-mini", (
            "Decompose must NOT fall back to the main VLM model when a "
            "cheaper decompose_model was configured."
        )

    def test_decompose_cache_still_works_across_calls(self) -> None:
        v = ClawVerifier(
            api_key="",
            model="anthropic/claude-sonnet-4-5",
            decompose_model="openai/gpt-4o-mini",
        )
        n_calls = {"n": 0}

        def fake_completion(**kwargs):
            n_calls["n"] += 1
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = '["Q1?"]'
            return resp

        with patch("comfyclaw.verifier.litellm.completion", side_effect=fake_completion):
            v._decompose_prompt("same prompt")
            v._decompose_prompt("same prompt")
            v._decompose_prompt("same prompt")

        assert n_calls["n"] == 1, "Per-prompt decomposition cache broke."

    def test_harness_config_default_is_none(self) -> None:
        cfg = HarnessConfig(api_key="")
        assert cfg.decompose_model is None

    def test_harness_config_accepts_decompose_model(self) -> None:
        cfg = HarnessConfig(api_key="", decompose_model="openai/gpt-4o-mini")
        assert cfg.decompose_model == "openai/gpt-4o-mini"
