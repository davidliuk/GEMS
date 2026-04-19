"""Comprehensive LoRA tests — wiring, stacking, rewiring, skill discovery, and edge cases.

Complements test_lora_controlnet_archs.py (which covers arch detection + basic dispatch)
with deeper coverage of the LoRA data-flow graph after tool execution.
"""

from __future__ import annotations

import pytest

from comfyclaw.agent import ARCH_REGISTRY, ClawAgent
from comfyclaw.skill_manager import SkillManager
from comfyclaw.stage_router import StageRouter
from comfyclaw.workflow import WorkflowManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(skills_dir: str | None = None) -> ClawAgent:
    agent = ClawAgent.__new__(ClawAgent)
    agent.model = "anthropic/claude-test"
    agent.server_address = "127.0.0.1:8188"
    agent.skill_manager = SkillManager(skills_dir)
    agent.on_change = None
    agent.on_agent_event = None
    agent.max_tool_rounds = 10
    agent.pinned_image_model = None
    agent.stage_router = StageRouter(enabled=False)
    agent.skills_read = []
    return agent


def _dispatch(agent: ClawAgent, wm: WorkflowManager, name: str, inputs: dict) -> str:
    result, _ = agent._dispatch(name, inputs, wm)
    return result


# ---------------------------------------------------------------------------
# Workflow fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sd_wm() -> WorkflowManager:
    """SD 1.5 workflow: CheckpointLoaderSimple → CLIPTextEncode → KSampler."""
    return WorkflowManager({
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"},
            "inputs": {"ckpt_name": "v1-5-pruned.ckpt"},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive"},
            "inputs": {"clip": ["1", 1], "text": "a red fox"},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Negative"},
            "inputs": {"clip": ["1", 1], "text": "blurry"},
        },
        "4": {
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"},
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "seed": 42, "steps": 20, "cfg": 7.0,
                "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
            },
        },
    })


@pytest.fixture()
def qwen_wm() -> WorkflowManager:
    """Qwen-Image-2512 with ModelSamplingAuraFlow between UNET and KSampler."""
    return WorkflowManager({
        "1": {
            "class_type": "UNETLoader",
            "_meta": {"title": "Qwen UNET"},
            "inputs": {"unet_name": "qwen_image_2512_fp8.safetensors", "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "_meta": {"title": "Qwen CLIP"},
            "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image"},
        },
        "3": {
            "class_type": "ModelSamplingAuraFlow",
            "_meta": {"title": "AuraFlow"},
            "inputs": {"model": ["1", 0], "shift": 3.1},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive"},
            "inputs": {"clip": ["2", 0], "text": "a red fox"},
        },
        "5": {
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"},
            "inputs": {
                "model": ["3", 0],
                "positive": ["4", 0],
                "seed": 42, "steps": 4, "cfg": 1.0,
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
            },
        },
    })


@pytest.fixture()
def zimage_wm() -> WorkflowManager:
    """Z-Image-Turbo with ModelSamplingAuraFlow and ConditioningZeroOut."""
    return WorkflowManager({
        "1": {
            "class_type": "UNETLoader",
            "_meta": {"title": "Z-Image UNET"},
            "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "_meta": {"title": "Z-Image CLIP"},
            "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2"},
        },
        "3": {
            "class_type": "ModelSamplingAuraFlow",
            "_meta": {"title": "AuraFlow"},
            "inputs": {"model": ["1", 0], "shift": 3},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive"},
            "inputs": {"clip": ["2", 0], "text": "a cat"},
        },
        "5": {
            "class_type": "ConditioningZeroOut",
            "_meta": {"title": "NegativeZero"},
            "inputs": {"conditioning": ["4", 0]},
        },
        "6": {
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"},
            "inputs": {
                "model": ["3", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "seed": 42, "steps": 8, "cfg": 1,
                "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0,
            },
        },
    })


@pytest.fixture()
def longcat_wm() -> WorkflowManager:
    """LongCat-Image with custom pipeline nodes (no standard MODEL tensor)."""
    return WorkflowManager({
        "1": {
            "class_type": "LongCatImageModelLoader",
            "_meta": {"title": "LongCat Loader"},
            "inputs": {"model_name": "longcat_image_bf16.safetensors", "precision": "bfloat16"},
        },
        "2": {
            "class_type": "LongCatImageTextToImage",
            "_meta": {"title": "LongCat Generate"},
            "inputs": {
                "model": ["1", 0],
                "positive_text": "a fox", "negative_text": "blurry",
                "steps": 28, "guidance_scale": 4.5, "seed": 42,
                "width": 1024, "height": 1024,
            },
        },
    })


# ===========================================================================
# 1. SD/SDXL LoRA — full MODEL + CLIP rewiring
# ===========================================================================


class TestSDLoraRewiring:
    """Verify standard LoRA (LoraLoader) correctly rewires both MODEL and CLIP chains."""

    def test_model_rewired_to_lora_output(self, sd_wm: WorkflowManager) -> None:
        agent = _make_agent()
        _dispatch(agent, sd_wm, "add_lora_loader", {
            "lora_name": "detail.safetensors",
            "model_node_id": "1", "clip_node_id": "1",
            "strength_model": 0.8, "strength_clip": 0.75,
        })
        lora_nids = sd_wm.get_nodes_by_class("LoraLoader")
        assert len(lora_nids) == 1
        lora_nid = lora_nids[0]
        assert sd_wm.workflow["4"]["inputs"]["model"] == [lora_nid, 0]

    def test_clip_rewired_to_lora_output(self, sd_wm: WorkflowManager) -> None:
        agent = _make_agent()
        _dispatch(agent, sd_wm, "add_lora_loader", {
            "lora_name": "detail.safetensors",
            "model_node_id": "1", "clip_node_id": "1",
            "strength_model": 0.8, "strength_clip": 0.75,
        })
        lora_nid = sd_wm.get_nodes_by_class("LoraLoader")[0]
        # Both CLIPTextEncode nodes consumed clip from node "1" slot 1.
        # They should now consume from the LoRA node slot 1.
        assert sd_wm.workflow["2"]["inputs"]["clip"] == [lora_nid, 1]
        assert sd_wm.workflow["3"]["inputs"]["clip"] == [lora_nid, 1]

    def test_lora_node_inputs_correct(self, sd_wm: WorkflowManager) -> None:
        agent = _make_agent()
        _dispatch(agent, sd_wm, "add_lora_loader", {
            "lora_name": "detail.safetensors",
            "model_node_id": "1", "clip_node_id": "1",
            "strength_model": 0.7, "strength_clip": 0.6,
        })
        lora_nid = sd_wm.get_nodes_by_class("LoraLoader")[0]
        lora = sd_wm.workflow[lora_nid]
        assert lora["inputs"]["model"] == ["1", 0]
        assert lora["inputs"]["clip"] == ["1", 1]
        assert lora["inputs"]["lora_name"] == "detail.safetensors"
        assert lora["inputs"]["strength_model"] == 0.7
        assert lora["inputs"]["strength_clip"] == 0.6

    def test_missing_clip_node_id_returns_error(self, sd_wm: WorkflowManager) -> None:
        agent = _make_agent()
        result = _dispatch(agent, sd_wm, "add_lora_loader", {
            "lora_name": "detail.safetensors",
            "model_node_id": "1",
        })
        assert "clip_node_id" in result.lower()
        assert sd_wm.get_nodes_by_class("LoraLoader") == []

    def test_empty_clip_node_id_returns_error(self, sd_wm: WorkflowManager) -> None:
        agent = _make_agent()
        result = _dispatch(agent, sd_wm, "add_lora_loader", {
            "lora_name": "detail.safetensors",
            "model_node_id": "1",
            "clip_node_id": "",
        })
        assert "clip_node_id" in result.lower()
        assert sd_wm.get_nodes_by_class("LoraLoader") == []


# ===========================================================================
# 2. Qwen LoRA — model-only, no CLIP
# ===========================================================================


class TestQwenLoraRewiring:
    """Verify Qwen uses LoraLoaderModelOnly and correctly rewires the MODEL chain."""

    def test_uses_model_only_loader(self, qwen_wm: WorkflowManager) -> None:
        agent = _make_agent()
        result = _dispatch(agent, qwen_wm, "add_lora_loader", {
            "lora_name": "lightning.safetensors",
            "model_node_id": "1",
            "strength_model": 1.0,
        })
        assert "LoraLoaderModelOnly" in result
        assert qwen_wm.get_nodes_by_class("LoraLoader") == []

    def test_model_chain_rewired(self, qwen_wm: WorkflowManager) -> None:
        """LoRA inserted between UNETLoader and ModelSamplingAuraFlow."""
        agent = _make_agent()
        _dispatch(agent, qwen_wm, "add_lora_loader", {
            "lora_name": "lightning.safetensors",
            "model_node_id": "1",
            "strength_model": 1.0,
        })
        lora_nid = qwen_wm.get_nodes_by_class("LoraLoaderModelOnly")[0]
        # ModelSamplingAuraFlow previously read model from ["1", 0];
        # should now read from LoRA
        assert qwen_wm.workflow["3"]["inputs"]["model"] == [lora_nid, 0]
        # LoRA reads from UNETLoader
        assert qwen_wm.workflow[lora_nid]["inputs"]["model"] == ["1", 0]

    def test_clip_untouched(self, qwen_wm: WorkflowManager) -> None:
        """Qwen LoRA should not modify any CLIP connections."""
        agent = _make_agent()
        _dispatch(agent, qwen_wm, "add_lora_loader", {
            "lora_name": "lightning.safetensors",
            "model_node_id": "1",
            "strength_model": 1.0,
        })
        assert qwen_wm.workflow["4"]["inputs"]["clip"] == ["2", 0]

    def test_no_clip_node_id_needed(self, qwen_wm: WorkflowManager) -> None:
        agent = _make_agent()
        result = _dispatch(agent, qwen_wm, "add_lora_loader", {
            "lora_name": "lightning.safetensors",
            "model_node_id": "1",
        })
        assert "✅" in result


# ===========================================================================
# 3. Z-Image LoRA — model-only, with ModelSamplingAuraFlow
# ===========================================================================


class TestZImageLoraRewiring:
    def test_uses_model_only_loader(self, zimage_wm: WorkflowManager) -> None:
        agent = _make_agent()
        result = _dispatch(agent, zimage_wm, "add_lora_loader", {
            "lora_name": "z_realism.safetensors",
            "model_node_id": "1",
            "strength_model": 0.8,
        })
        assert "LoraLoaderModelOnly" in result

    def test_model_chain_rewired(self, zimage_wm: WorkflowManager) -> None:
        agent = _make_agent()
        _dispatch(agent, zimage_wm, "add_lora_loader", {
            "lora_name": "z_realism.safetensors",
            "model_node_id": "1",
            "strength_model": 0.8,
        })
        lora_nid = zimage_wm.get_nodes_by_class("LoraLoaderModelOnly")[0]
        # ModelSamplingAuraFlow previously read from ["1", 0]; now from LoRA
        assert zimage_wm.workflow["3"]["inputs"]["model"] == [lora_nid, 0]


# ===========================================================================
# 4. LongCat — LoRA blocked
# ===========================================================================


class TestLongCatLoraBlocked:
    def test_returns_not_supported(self, longcat_wm: WorkflowManager) -> None:
        agent = _make_agent()
        result = _dispatch(agent, longcat_wm, "add_lora_loader", {
            "lora_name": "any.safetensors",
            "model_node_id": "1",
        })
        assert "not supported" in result.lower()

    def test_no_nodes_added(self, longcat_wm: WorkflowManager) -> None:
        agent = _make_agent()
        node_count_before = len(longcat_wm.workflow)
        _dispatch(agent, longcat_wm, "add_lora_loader", {
            "lora_name": "any.safetensors",
            "model_node_id": "1",
        })
        assert len(longcat_wm.workflow) == node_count_before

    def test_workflow_unchanged(self, longcat_wm: WorkflowManager) -> None:
        """No wiring should be modified on a blocked arch."""
        import copy
        agent = _make_agent()
        before = copy.deepcopy(longcat_wm.workflow)
        _dispatch(agent, longcat_wm, "add_lora_loader", {
            "lora_name": "any.safetensors",
            "model_node_id": "1",
        })
        assert longcat_wm.workflow == before


# ===========================================================================
# 5. LoRA stacking — chaining multiple LoRAs
# ===========================================================================


class TestLoraStacking:
    def test_qwen_two_lora_stack(self, qwen_wm: WorkflowManager) -> None:
        """Speed LoRA → Style LoRA chain for Qwen."""
        agent = _make_agent()

        # First LoRA: speed, attached to UNETLoader
        result1 = _dispatch(agent, qwen_wm, "add_lora_loader", {
            "lora_name": "lightning-4step.safetensors",
            "model_node_id": "1",
            "strength_model": 1.0,
        })
        assert "✅" in result1
        lora1_nid = qwen_wm.get_nodes_by_class("LoraLoaderModelOnly")[0]

        # Second LoRA: style, attached to first LoRA output
        result2 = _dispatch(agent, qwen_wm, "add_lora_loader", {
            "lora_name": "anime-style.safetensors",
            "model_node_id": lora1_nid,
            "strength_model": 0.75,
        })
        assert "✅" in result2
        all_loras = qwen_wm.get_nodes_by_class("LoraLoaderModelOnly")
        assert len(all_loras) == 2
        lora2_nid = [n for n in all_loras if n != lora1_nid][0]

        # Chain: UNET → LoRA1 → LoRA2 → ModelSamplingAuraFlow → KSampler
        assert qwen_wm.workflow[lora1_nid]["inputs"]["model"] == ["1", 0]
        assert qwen_wm.workflow[lora2_nid]["inputs"]["model"] == [lora1_nid, 0]
        # ModelSamplingAuraFlow should point to LoRA2
        assert qwen_wm.workflow["3"]["inputs"]["model"] == [lora2_nid, 0]

    def test_sd_two_lora_stack(self, sd_wm: WorkflowManager) -> None:
        """Two standard LoRAs on SD pipeline."""
        agent = _make_agent()

        _dispatch(agent, sd_wm, "add_lora_loader", {
            "lora_name": "detail.safetensors",
            "model_node_id": "1", "clip_node_id": "1",
            "strength_model": 0.8, "strength_clip": 0.8,
        })
        lora1_nid = sd_wm.get_nodes_by_class("LoraLoader")[0]

        _dispatch(agent, sd_wm, "add_lora_loader", {
            "lora_name": "style.safetensors",
            "model_node_id": lora1_nid, "clip_node_id": lora1_nid,
            "strength_model": 0.6, "strength_clip": 0.6,
        })
        all_loras = sd_wm.get_nodes_by_class("LoraLoader")
        assert len(all_loras) == 2
        lora2_nid = [n for n in all_loras if n != lora1_nid][0]

        # KSampler model should point to the last LoRA
        assert sd_wm.workflow["4"]["inputs"]["model"] == [lora2_nid, 0]
        # CLIP chain: both CLIPTextEncode nodes should point to LoRA2 clip out
        assert sd_wm.workflow["2"]["inputs"]["clip"] == [lora2_nid, 1]
        assert sd_wm.workflow["3"]["inputs"]["clip"] == [lora2_nid, 1]


# ===========================================================================
# 6. CLIP slot detection
# ===========================================================================


class TestClipSlotDetection:
    def test_checkpoint_loader_clip_is_slot_1(self) -> None:
        wm = WorkflowManager({
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x.ckpt"}},
        })
        assert ClawAgent._detect_clip_slot(wm, "1") == 1

    def test_clip_loader_clip_is_slot_0(self) -> None:
        wm = WorkflowManager({
            "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "x.safetensors", "type": "qwen_image"}},
        })
        assert ClawAgent._detect_clip_slot(wm, "1") == 0

    def test_dual_clip_loader_clip_is_slot_0(self) -> None:
        wm = WorkflowManager({
            "1": {"class_type": "DualCLIPLoader", "inputs": {}},
        })
        assert ClawAgent._detect_clip_slot(wm, "1") == 0

    def test_lora_loader_clip_is_slot_1(self) -> None:
        wm = WorkflowManager({
            "1": {"class_type": "LoraLoader", "inputs": {}},
        })
        assert ClawAgent._detect_clip_slot(wm, "1") == 1

    def test_unknown_node_uses_heuristic(self) -> None:
        """Falls back to scanning existing consumers for evidence."""
        wm = WorkflowManager({
            "1": {"class_type": "SomeCustomLoader", "inputs": {}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 2], "text": "hi"}},
        })
        # Node "2" reads clip from ["1", 2], so slot 2 should be detected
        assert ClawAgent._detect_clip_slot(wm, "1") == 2


# ===========================================================================
# 7. Architecture detection
# ===========================================================================


class TestArchDetection:
    def test_qwen_detected_by_unet_name(self, qwen_wm: WorkflowManager) -> None:
        arch = ClawAgent._detect_arch(qwen_wm)
        assert arch is not None
        assert arch.skill_name == "qwen-image-2512"
        assert arch.lora_node == "LoraLoaderModelOnly"

    def test_zimage_detected_by_clip_type(self, zimage_wm: WorkflowManager) -> None:
        arch = ClawAgent._detect_arch(zimage_wm)
        assert arch is not None
        assert arch.skill_name == "z-image-turbo"

    def test_longcat_detected_by_class_type(self, longcat_wm: WorkflowManager) -> None:
        arch = ClawAgent._detect_arch(longcat_wm)
        assert arch is not None
        assert arch.lora_supported is False

    def test_sd_workflow_returns_none(self, sd_wm: WorkflowManager) -> None:
        assert ClawAgent._detect_arch(sd_wm) is None

    def test_empty_workflow_returns_none(self) -> None:
        assert ClawAgent._detect_arch(WorkflowManager({})) is None


# ===========================================================================
# 8. Skill discovery for LoRA scenarios
# ===========================================================================


class TestLoraSkillDiscovery:
    """Verify the lora-enhancement skill is discoverable and loadable."""

    def test_lora_skill_exists(self) -> None:
        sm = SkillManager(None)
        assert "lora-enhancement" in sm.skill_names

    def test_lora_skill_has_agent_tag(self) -> None:
        sm = SkillManager(None)
        props = sm.get_properties("lora-enhancement")
        assert "agent" in (props.tags or [])

    def test_lora_skill_appears_in_xml(self) -> None:
        sm = SkillManager(None)
        xml = sm.build_available_skills_xml(include_tags={"agent"})
        assert "<name>lora-enhancement</name>" in xml

    def test_lora_skill_body_loadable(self) -> None:
        sm = SkillManager(None)
        body = sm.get_body("lora-enhancement")
        assert "add_lora_loader" in body
        assert "query_available_models" in body

    def test_model_skills_have_lora_guidance(self) -> None:
        """Each model-specific skill should document LoRA support/non-support."""
        sm = SkillManager(None)
        for name in ["qwen-image-2512", "z-image-turbo", "longcat-image"]:
            body = sm.get_body(name)
            assert "lora" in body.lower(), f"{name} skill should mention LoRA"


# ===========================================================================
# 9. Structural hint extraction from verifier feedback
# ===========================================================================


class TestStructuralHints:
    """Verify verifier fix_strategy keywords map to LoRA skill + tool directives."""

    def test_inject_lora_detail_maps_to_skill(self) -> None:
        hints = ClawAgent._extract_structural_hints(
            "Score: 6/10. fix_strategy: inject_lora_detail. Skin looks plasticky."
        )
        assert "lora-enhancement" in hints
        assert "add_lora_loader" in hints

    def test_inject_lora_style_maps_to_skill(self) -> None:
        hints = ClawAgent._extract_structural_hints(
            "fix_strategy: inject_lora_style"
        )
        assert "lora-enhancement" in hints

    def test_inject_lora_anatomy_maps_to_skill(self) -> None:
        hints = ClawAgent._extract_structural_hints(
            "fix_strategy: inject_lora_anatomy"
        )
        assert "lora-enhancement" in hints

    def test_no_lora_keywords_returns_empty(self) -> None:
        hints = ClawAgent._extract_structural_hints(
            "Score: 9/10. Good image quality. fix_strategy: refine_positive_prompt"
        )
        assert "lora" not in hints.lower()

    def test_multiple_strategies_all_captured(self) -> None:
        hints = ClawAgent._extract_structural_hints(
            "fix_strategy: inject_lora_detail, inject_lora_lighting, add_hires_fix"
        )
        assert "detail" in hints
        assert "lighting" in hints
        assert "hires" in hints


# ===========================================================================
# 10. Stage router — LoRA tool availability
# ===========================================================================


class TestStageRouterLora:
    """Verify add_lora_loader is available at the right stages."""

    def test_lora_available_in_enhancement_stage(self) -> None:
        router = StageRouter(enabled=True)
        router.transition_to("enhancement")
        names = router.get_current_tool_names()
        assert "add_lora_loader" in names
        assert "query_available_models" in names

    def test_lora_not_available_in_planning_stage(self) -> None:
        router = StageRouter(enabled=True)
        assert "add_lora_loader" not in router.get_current_tool_names()

    def test_lora_not_available_in_conditioning_stage(self) -> None:
        router = StageRouter(enabled=True)
        router.transition_to("conditioning")
        assert "add_lora_loader" not in router.get_current_tool_names()

    def test_lora_not_available_in_finalization_stage(self) -> None:
        router = StageRouter(enabled=True)
        router.transition_to("finalization")
        assert "add_lora_loader" not in router.get_current_tool_names()

    def test_all_tools_available_when_router_disabled(self) -> None:
        from comfyclaw.agent import _TOOLS
        router = StageRouter(enabled=False)
        filtered = router.filter_tools(_TOOLS)
        tool_names = [t["function"]["name"] for t in filtered]
        assert "add_lora_loader" in tool_names


# ===========================================================================
# 11. Default strength values
# ===========================================================================


class TestDefaultStrengths:
    def test_sd_default_strength_model(self, sd_wm: WorkflowManager) -> None:
        agent = _make_agent()
        _dispatch(agent, sd_wm, "add_lora_loader", {
            "lora_name": "x.safetensors",
            "model_node_id": "1", "clip_node_id": "1",
        })
        lora_nid = sd_wm.get_nodes_by_class("LoraLoader")[0]
        assert sd_wm.workflow[lora_nid]["inputs"]["strength_model"] == 0.8
        assert sd_wm.workflow[lora_nid]["inputs"]["strength_clip"] == 0.8

    def test_qwen_default_strength_model(self, qwen_wm: WorkflowManager) -> None:
        agent = _make_agent()
        _dispatch(agent, qwen_wm, "add_lora_loader", {
            "lora_name": "x.safetensors",
            "model_node_id": "1",
        })
        lora_nid = qwen_wm.get_nodes_by_class("LoraLoaderModelOnly")[0]
        assert qwen_wm.workflow[lora_nid]["inputs"]["strength_model"] == 0.8


# ===========================================================================
# 12. Arch registry consistency
# ===========================================================================


class TestArchRegistryConsistency:
    def test_all_arches_have_required_fields(self) -> None:
        for name, cfg in ARCH_REGISTRY.items():
            assert cfg.skill_name, f"{name} missing skill_name"
            assert cfg.description, f"{name} missing description"
            if cfg.lora_supported:
                assert cfg.lora_node, f"{name} supports LoRA but has empty lora_node"

    def test_lora_unsupported_arch_has_empty_node(self) -> None:
        for name, cfg in ARCH_REGISTRY.items():
            if not cfg.lora_supported:
                assert cfg.lora_node == "", f"{name} doesn't support LoRA but has lora_node={cfg.lora_node!r}"
