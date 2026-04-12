"""Unit tests for ClawAgent tool dispatch (litellm mocked)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from comfyclaw.agent import ClawAgent
from comfyclaw.workflow import WorkflowManager

# ---------------------------------------------------------------------------
# Helpers — build LiteLLM / OpenAI-format mock responses
# ---------------------------------------------------------------------------


def _make_agent(
    server_address: str = "127.0.0.1:8188",
    pinned_image_model: str | None = None,
) -> ClawAgent:
    agent = ClawAgent.__new__(ClawAgent)
    agent.model = "anthropic/claude-test"
    agent.server_address = server_address
    from comfyclaw.skill_manager import SkillManager

    agent.skill_manager = SkillManager(None)  # built-in skills
    agent.on_change = None
    agent.on_agent_event = None
    agent.max_tool_rounds = 10
    agent.pinned_image_model = pinned_image_model
    return agent


def _litellm_tool_call(name: str, inputs: dict, call_id: str = "call_1") -> SimpleNamespace:
    """Build a single tool_call object in OpenAI/LiteLLM format."""
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(inputs)),
    )


def _litellm_tool_response(*tool_calls: SimpleNamespace) -> MagicMock:
    """Build a litellm.completion response that requests tool calls."""
    message = MagicMock()
    message.content = None
    message.tool_calls = list(tool_calls)

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "tool_calls"

    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _litellm_stop_response(text: str = "Done.") -> MagicMock:
    """Build a litellm.completion response that signals end of turn."""
    message = MagicMock()
    message.content = text
    message.tool_calls = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# Direct tool dispatch tests (unit-level, no LLM call needed)
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_set_param_mutates_workflow(self, wm: WorkflowManager) -> None:
        agent = _make_agent()
        result, stop = agent._dispatch(
            "set_param", {"node_id": "3", "param_name": "steps", "value": 50}, wm
        )
        assert wm.workflow["3"]["inputs"]["steps"] == 50
        assert "✅" in result
        assert stop is False

    def test_add_node_returns_id(self, wm: WorkflowManager) -> None:
        agent = _make_agent()
        result, stop = agent._dispatch("add_node", {"class_type": "VAEDecode", "inputs": {}}, wm)
        assert "VAEDecode" in result
        assert stop is False
        # A new node should now exist
        assert any(n["class_type"] == "VAEDecode" for n in wm.workflow.values())

    def test_connect_nodes(self, wm: WorkflowManager) -> None:
        nid = wm.add_node("VAEDecode")
        agent = _make_agent()
        result, stop = agent._dispatch(
            "connect_nodes",
            {
                "src_node_id": "3",
                "src_output_index": 0,
                "dst_node_id": nid,
                "dst_input_name": "samples",
            },
            wm,
        )
        assert wm.workflow[nid]["inputs"]["samples"] == ["3", 0]
        assert stop is False

    def test_delete_node(self, wm: WorkflowManager) -> None:
        agent = _make_agent()
        result, stop = agent._dispatch("delete_node", {"node_id": "2"}, wm)
        assert "2" not in wm.workflow
        assert stop is False

    def test_finalize_stops_loop(self, wm: WorkflowManager) -> None:
        agent = _make_agent()
        wm.add_node("SaveImage", "Save", images=["3", 0])
        result, stop = agent._dispatch("finalize_workflow", {"rationale": "All done."}, wm)
        assert stop is True

    def test_finalize_blocks_on_invalid_workflow(self, wm: WorkflowManager) -> None:
        agent = _make_agent()
        result, stop = agent._dispatch("finalize_workflow", {"rationale": "done"}, wm)
        assert stop is False
        assert "⚠️" in result

    def test_validate_workflow(self, wm: WorkflowManager) -> None:
        agent = _make_agent()
        result, stop = agent._dispatch("validate_workflow", {}, wm)
        assert stop is False
        assert "SaveImage" in result
        wm.add_node("SaveImage", "Save", images=["3", 0])
        result2, _ = agent._dispatch("validate_workflow", {}, wm)
        assert "✅" in result2

    def test_unknown_tool_returns_error(self, wm: WorkflowManager) -> None:
        agent = _make_agent()
        result, stop = agent._dispatch("does_not_exist", {}, wm)
        assert "❌" in result
        assert stop is False

    def test_tool_error_returns_error_string(self, wm: WorkflowManager) -> None:
        agent = _make_agent()
        # set_param on a non-existent node should produce a tool error, not raise
        result, stop = agent._dispatch(
            "set_param", {"node_id": "999", "param_name": "x", "value": 1}, wm
        )
        assert "❌" in result
        assert stop is False


# ---------------------------------------------------------------------------
# LoRA rewiring
# ---------------------------------------------------------------------------


class TestAddLora:
    def test_rewires_model_and_clip_consumers(self, wm: WorkflowManager) -> None:
        """After add_lora_loader, KSampler should take model from LoraLoader[0]."""
        agent = _make_agent()
        result, _ = agent._dispatch(
            "add_lora_loader",
            {
                "lora_name": "detail-tweaker.safetensors",
                "model_node_id": "1",
                "clip_node_id": "1",
                "strength_model": 0.8,
                "strength_clip": 0.8,
            },
            wm,
        )
        assert "LoraLoader" in result
        # Find the new LoraLoader node
        lora_nids = wm.get_nodes_by_class("LoraLoader")
        assert len(lora_nids) == 1
        lora_nid = lora_nids[0]
        # KSampler (node "3") model should now come from LoraLoader
        assert wm.workflow["3"]["inputs"]["model"] == [lora_nid, 0]


# ---------------------------------------------------------------------------
# Regional attention (bug fix: no _meta KeyError)
# ---------------------------------------------------------------------------


class TestRegionalAttention:
    def test_no_meta_keyerror(self, wm: WorkflowManager) -> None:
        """Nodes without _meta should not raise KeyError."""
        wm.workflow["2"].pop("_meta", None)  # remove _meta from node 2
        agent = _make_agent()
        # Should not raise
        result, _ = agent._dispatch(
            "add_regional_attention",
            {
                "positive_node_id": "2",
                "clip_node_id": "1",
                "foreground_prompt": "cat",
                "background_prompt": "forest",
            },
            wm,
        )
        assert "✅" in result

    def test_ksampler_rewired(self, wm: WorkflowManager) -> None:
        agent = _make_agent()
        agent._dispatch(
            "add_regional_attention",
            {
                "positive_node_id": "2",
                "clip_node_id": "1",
                "foreground_prompt": "subject",
                "background_prompt": "background",
            },
            wm,
        )
        # KSampler positive should now point to a ConditioningCombine node
        pos_link = wm.workflow["3"]["inputs"]["positive"]
        combine_nids = wm.get_nodes_by_class("ConditioningCombine")
        assert len(combine_nids) == 1
        assert pos_link[0] == combine_nids[0]


# ---------------------------------------------------------------------------
# Hires fix
# ---------------------------------------------------------------------------


class TestHiresFix:
    def test_adds_upscale_and_second_sampler(self, wm: WorkflowManager) -> None:
        # Add a VAEDecode and SaveImage first so hires fix has something to re-wire
        decode_nid = wm.add_node("VAEDecode", samples=["3", 0], vae=["1", 2])
        save_nid = wm.add_node("SaveImage", images=[decode_nid, 0])
        agent = _make_agent()
        agent._dispatch(
            "add_hires_fix",
            {
                "base_ksampler_node_id": "3",
                "vae_node_id": "1",
                "scale_by": 1.5,
                "hires_steps": 10,
                "hires_denoise": 0.4,
                "save_image_node_id": save_nid,
            },
            wm,
        )
        upscale_nids = wm.get_nodes_by_class("LatentUpscaleBy")
        hires_ks_nids = [
            nid
            for nid, n in wm.workflow.items()
            if n.get("class_type") == "KSampler" and nid != "3"
        ]
        assert len(upscale_nids) == 1
        assert len(hires_ks_nids) == 1


# ---------------------------------------------------------------------------
# Query models (offline)
# ---------------------------------------------------------------------------


class TestQueryModels:
    def test_offline_returns_error_string(self, wm: WorkflowManager) -> None:
        # Use an address that won't respond
        agent = _make_agent(server_address="127.0.0.1:19999")
        result, stop = agent._dispatch("query_available_models", {"model_type": "loras"}, wm)
        assert "❌" in result or "No" in result or "Could not" in result
        assert stop is False

    def test_unknown_model_type_returns_error(self, wm: WorkflowManager) -> None:
        agent = _make_agent()
        result, stop = agent._dispatch("query_available_models", {"model_type": "unknown_type"}, wm)
        assert "❌" in result
        assert stop is False


# ---------------------------------------------------------------------------
# plan_and_patch integration (litellm.completion mocked)
# ---------------------------------------------------------------------------


class TestPlanAndPatch:
    def test_finalize_returns_rationale(self, wm: WorkflowManager) -> None:
        wm.add_node("SaveImage", "Save", images=["3", 0])
        strategy_resp = _litellm_tool_response(
            _litellm_tool_call(
                "report_evolution_strategy",
                {"strategy": "optimize", "top_issue": "quality"},
                "call_1",
            )
        )
        finalize_resp = _litellm_tool_response(
            _litellm_tool_call("finalize_workflow", {"rationale": "All done."}, "call_2")
        )

        agent = _make_agent()
        with patch("litellm.completion", side_effect=[strategy_resp, finalize_resp]):
            rationale = agent.plan_and_patch(wm, "a red fox")

        assert rationale == "All done."

    def test_end_turn_stops_loop(self, wm: WorkflowManager) -> None:
        stop_resp = _litellm_stop_response("Done.")

        agent = _make_agent()
        with patch("litellm.completion", return_value=stop_resp) as mock_completion:
            rationale = agent.plan_and_patch(wm, "a prompt")

        assert isinstance(rationale, str)
        assert mock_completion.call_count == 1
