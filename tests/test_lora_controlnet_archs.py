"""Tests for LoRA/ControlNet arch-specific wiring.

Covers the ARCH_REGISTRY-driven routing in ClawAgent._add_lora and
ClawAgent._add_controlnet for:

- Standard SD/SDXL/Flux:   LoraLoader + ControlNetApplyAdvanced
- Qwen-Image-2512:         LoraLoaderModelOnly + QwenImageFunControlNet(Loader|Apply)
- Z-Image-Turbo:           LoraLoaderModelOnly + ZImageFunControlNet(Loader|Apply)
- LongCat-Image:           LoRA/CN gracefully blocked with informative error
"""

from __future__ import annotations

import pytest

from comfyclaw.agent import ARCH_REGISTRY, ClawAgent
from comfyclaw.workflow import WorkflowManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent() -> ClawAgent:
    """Build a minimal ClawAgent instance without touching litellm / network."""
    agent = ClawAgent.__new__(ClawAgent)
    agent.model = "anthropic/claude-test"
    agent.server_address = "127.0.0.1:8188"
    from comfyclaw.skill_manager import SkillManager

    agent.skill_manager = SkillManager(None)
    agent.on_change = None
    agent.on_agent_event = None
    agent.max_tool_rounds = 10
    agent.pinned_image_model = None
    from comfyclaw.stage_router import StageRouter

    agent.stage_router = StageRouter(enabled=False)
    return agent


# ---------------------------------------------------------------------------
# Arch-specific workflow fixtures (minimal but faithful to each arch)
# ---------------------------------------------------------------------------


@pytest.fixture()
def qwen_wm() -> WorkflowManager:
    """Qwen-Image-2512: UNETLoader with 'qwen_image' name + CLIPLoader type."""
    return WorkflowManager(
        {
            "1": {
                "class_type": "UNETLoader",
                "_meta": {"title": "Qwen UNET"},
                "inputs": {
                    "unet_name": "qwen_image_2512_fp8_e4m3fn.safetensors",
                    "weight_dtype": "default",
                },
            },
            "2": {
                "class_type": "CLIPLoader",
                "_meta": {"title": "Qwen CLIP"},
                "inputs": {
                    "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                    "type": "qwen_image",
                },
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive"},
                "inputs": {"clip": ["2", 0], "text": "a red fox"},
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Negative"},
                "inputs": {"clip": ["2", 0], "text": "blurry"},
            },
            "5": {
                "class_type": "KSampler",
                "_meta": {"title": "KSampler"},
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "seed": 42,
                    "steps": 50,
                    "cfg": 4.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                },
            },
        }
    )


@pytest.fixture()
def zimage_wm() -> WorkflowManager:
    """Z-Image-Turbo: UNETLoader with 'z_image' + CLIPLoader type 'qwen3_4b'."""
    return WorkflowManager(
        {
            "1": {
                "class_type": "UNETLoader",
                "_meta": {"title": "Z-Image UNET"},
                "inputs": {
                    "unet_name": "z_image_turbo_bf16.safetensors",
                    "weight_dtype": "default",
                },
            },
            "2": {
                "class_type": "CLIPLoader",
                "_meta": {"title": "Z-Image CLIP"},
                "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "qwen3_4b"},
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive"},
                "inputs": {"clip": ["2", 0], "text": "a cat"},
            },
            "4": {
                "class_type": "ConditioningZeroOut",
                "_meta": {"title": "NegativeZero"},
                "inputs": {"conditioning": ["3", 0]},
            },
            "5": {
                "class_type": "KSampler",
                "_meta": {"title": "KSampler"},
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "seed": 42,
                    "steps": 8,
                    "cfg": 1.0,
                    "sampler_name": "res_multistep",
                    "scheduler": "simple",
                    "denoise": 1.0,
                },
            },
        }
    )


@pytest.fixture()
def longcat_wm() -> WorkflowManager:
    """LongCat-Image: UNETLoader with 'longcat' name (standard KSampler arch)."""
    return WorkflowManager(
        {
            "1": {
                "class_type": "UNETLoader",
                "_meta": {"title": "LongCat UNET"},
                "inputs": {
                    "unet_name": "longcat_image_bf16.safetensors",
                    "weight_dtype": "default",
                },
            },
            "2": {
                "class_type": "CLIPLoader",
                "_meta": {"title": "LongCat CLIP"},
                "inputs": {
                    "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                    "type": "longcat_image",
                },
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive"},
                "inputs": {"clip": ["2", 0], "text": "a teahouse"},
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Negative"},
                "inputs": {"clip": ["2", 0], "text": "blurry"},
            },
            "5": {
                "class_type": "KSampler",
                "_meta": {"title": "KSampler"},
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "seed": 42,
                    "steps": 20,
                    "cfg": 4.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                },
            },
        }
    )


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------


class TestArchRegistry:
    def test_registry_contains_three_archs(self) -> None:
        assert set(ARCH_REGISTRY) == {"qwen_image", "z_image", "longcat_image"}

    def test_qwen_uses_model_only_lora_and_patch_cn(self) -> None:
        cfg = ARCH_REGISTRY["qwen_image"]
        assert cfg.lora_node == "LoraLoaderModelOnly"
        assert cfg.lora_needs_clip is False
        assert cfg.cn_style == "model_patch"
        assert cfg.cn_loader_node == "ModelPatchLoader"
        assert cfg.cn_apply_node == "QwenImageDiffsynthControlnet"
        assert cfg.cn_has_union_type is False
        assert cfg.lora_supported and cfg.cn_supported

    def test_zimage_uses_model_only_lora_and_patch_cn(self) -> None:
        cfg = ARCH_REGISTRY["z_image"]
        assert cfg.lora_node == "LoraLoaderModelOnly"
        assert cfg.lora_needs_clip is False
        assert cfg.cn_style == "model_patch"
        assert cfg.cn_loader_node == "ModelPatchLoader"
        assert cfg.cn_apply_node == "ZImageFunControlnet"
        assert cfg.cn_has_union_type is False
        assert cfg.lora_supported and cfg.cn_supported

    def test_longcat_blocks_lora_and_cn(self) -> None:
        cfg = ARCH_REGISTRY["longcat_image"]
        assert cfg.lora_supported is False
        assert cfg.cn_supported is False


# ---------------------------------------------------------------------------
# _detect_arch
# ---------------------------------------------------------------------------


class TestDetectArch:
    def test_detects_qwen_by_unet_name(self, qwen_wm: WorkflowManager) -> None:
        cfg = ClawAgent._detect_arch(qwen_wm)
        assert cfg is not None and cfg.skill_name == "qwen-image-2512"

    def test_detects_zimage_by_clip_type(self, zimage_wm: WorkflowManager) -> None:
        cfg = ClawAgent._detect_arch(zimage_wm)
        assert cfg is not None and cfg.skill_name == "z-image-turbo"

    def test_detects_longcat_by_unet_name(self, longcat_wm: WorkflowManager) -> None:
        cfg = ClawAgent._detect_arch(longcat_wm)
        assert cfg is not None and cfg.skill_name == "longcat-image"

    def test_standard_sd_returns_none(self, wm: WorkflowManager) -> None:
        """wm fixture from conftest.py is an SD1.5 CheckpointLoaderSimple workflow."""
        assert ClawAgent._detect_arch(wm) is None

    def test_detects_qwen_via_plugin_node(self) -> None:
        wm = WorkflowManager(
            {
                "1": {
                    "class_type": "QwenImageModelLoader",
                    "inputs": {"ckpt_name": "qwen-image"},
                    "_meta": {"title": "Qwen Loader"},
                },
            }
        )
        cfg = ClawAgent._detect_arch(wm)
        assert cfg is not None and cfg.skill_name == "qwen-image-2512"


# ---------------------------------------------------------------------------
# LoRA — per-arch wiring
# ---------------------------------------------------------------------------


class TestAddLoraQwen:
    def test_uses_lora_loader_model_only(self, qwen_wm: WorkflowManager) -> None:
        agent = _make_agent()
        result, _ = agent._dispatch(
            "add_lora_loader",
            {
                "lora_name": "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors",
                "model_node_id": "1",
                "strength_model": 0.8,
            },
            qwen_wm,
        )
        assert "LoraLoaderModelOnly" in result
        model_only_nids = qwen_wm.get_nodes_by_class("LoraLoaderModelOnly")
        plain_nids = qwen_wm.get_nodes_by_class("LoraLoader")
        assert len(model_only_nids) == 1
        assert plain_nids == []
        # KSampler (node 5) model input should be rewired to the new LoRA node
        assert qwen_wm.workflow["5"]["inputs"]["model"] == [model_only_nids[0], 0]

    def test_accepts_missing_clip_node_id(self, qwen_wm: WorkflowManager) -> None:
        """clip_node_id is no longer required for MMDiT archs."""
        agent = _make_agent()
        result, _ = agent._dispatch(
            "add_lora_loader",
            {"lora_name": "lightning.safetensors", "model_node_id": "1"},
            qwen_wm,
        )
        assert "LoraLoaderModelOnly" in result
        assert "⚠️" not in result and "❌" not in result


class TestAddLoraZImage:
    def test_uses_lora_loader_model_only(self, zimage_wm: WorkflowManager) -> None:
        agent = _make_agent()
        result, _ = agent._dispatch(
            "add_lora_loader",
            {
                "lora_name": "z_image_style.safetensors",
                "model_node_id": "1",
                "strength_model": 0.75,
            },
            zimage_wm,
        )
        assert "LoraLoaderModelOnly" in result
        model_only_nids = zimage_wm.get_nodes_by_class("LoraLoaderModelOnly")
        assert len(model_only_nids) == 1
        assert zimage_wm.workflow["5"]["inputs"]["model"] == [model_only_nids[0], 0]


class TestAddLoraLongCat:
    def test_returns_blocked_error(self, longcat_wm: WorkflowManager) -> None:
        agent = _make_agent()
        result, stop = agent._dispatch(
            "add_lora_loader",
            {"lora_name": "whatever.safetensors", "model_node_id": "1"},
            longcat_wm,
        )
        assert stop is False
        assert "longcat-image" in result.lower()
        assert "not supported" in result.lower()
        # No loader nodes should have been added
        assert longcat_wm.get_nodes_by_class("LoraLoader") == []
        assert longcat_wm.get_nodes_by_class("LoraLoaderModelOnly") == []


# ---------------------------------------------------------------------------
# ControlNet — per-arch wiring
# ---------------------------------------------------------------------------


class TestAddControlNetQwen:
    def test_uses_model_patch_nodes(self, qwen_wm: WorkflowManager) -> None:
        agent = _make_agent()
        # The Qwen fixture needs a VAELoader for the model-patch CN; add it.
        vae_nid = qwen_wm.add_node("VAELoader", vae_name="qwen_image_vae.safetensors")
        img_nid = qwen_wm.add_node("LoadImage", image="ref.png")
        result, _ = agent._dispatch(
            "add_controlnet",
            {
                "controlnet_name": "qwen_image_diffsynth_canny.safetensors",
                "positive_node_id": "3",
                "negative_node_id": "4",
                "image_node_id": img_nid,
                "strength": 0.75,
            },
            qwen_wm,
        )
        loader_nids = qwen_wm.get_nodes_by_class("ModelPatchLoader")
        patch_nids = qwen_wm.get_nodes_by_class("QwenImageDiffsynthControlnet")
        assert len(loader_nids) == 1
        assert len(patch_nids) == 1
        # Patch node has correct wiring.
        p_in = qwen_wm.workflow[patch_nids[0]]["inputs"]
        assert p_in["model"] == ["1", 0]  # was UNETLoader
        assert p_in["model_patch"] == [loader_nids[0], 0]
        assert p_in["vae"] == [vae_nid, 0]
        assert p_in["image"] == [img_nid, 0]
        assert p_in["strength"] == 0.75
        assert "union_type" not in p_in
        # KSampler.model should now come from the patch node.
        assert qwen_wm.workflow["5"]["inputs"]["model"] == [patch_nids[0], 0]
        # Conditioning untouched.
        assert qwen_wm.workflow["5"]["inputs"]["positive"] == ["3", 0]
        assert qwen_wm.workflow["5"]["inputs"]["negative"] == ["4", 0]
        # No legacy nodes should have been emitted.
        assert qwen_wm.get_nodes_by_class("QwenImageFunControlNetLoader") == []
        assert qwen_wm.get_nodes_by_class("QwenImageFunControlNetApply") == []
        assert qwen_wm.get_nodes_by_class("ControlNetLoader") == []
        assert qwen_wm.get_nodes_by_class("ControlNetApplyAdvanced") == []
        assert "qwen-image-2512" in result.lower()

    def test_requires_image_node_id(self, qwen_wm: WorkflowManager) -> None:
        """model-patch CN requires a reference image; without it, error out."""
        agent = _make_agent()
        qwen_wm.add_node("VAELoader", vae_name="qwen_image_vae.safetensors")
        result, _ = agent._dispatch(
            "add_controlnet",
            {
                "controlnet_name": "x.safetensors",
                "positive_node_id": "3",
                "negative_node_id": "4",
                "strength": 0.7,
            },
            qwen_wm,
        )
        assert "⚠️" in result
        assert "image_node_id" in result
        assert qwen_wm.get_nodes_by_class("ModelPatchLoader") == []

    def test_requires_vae_loader(self, qwen_wm: WorkflowManager) -> None:
        """model-patch CN requires a VAE source."""
        agent = _make_agent()
        img_nid = qwen_wm.add_node("LoadImage", image="ref.png")
        result, _ = agent._dispatch(
            "add_controlnet",
            {
                "controlnet_name": "x.safetensors",
                "positive_node_id": "3",
                "negative_node_id": "4",
                "image_node_id": img_nid,
                "strength": 0.7,
            },
            qwen_wm,
        )
        assert "⚠️" in result
        assert "VAELoader" in result
        assert qwen_wm.get_nodes_by_class("ModelPatchLoader") == []


class TestAddControlNetZImage:
    def test_uses_model_patch_nodes(self, zimage_wm: WorkflowManager) -> None:
        agent = _make_agent()
        vae_nid = zimage_wm.add_node("VAELoader", vae_name="ae.safetensors")
        img_nid = zimage_wm.add_node("LoadImage", image="ref.png")
        result, _ = agent._dispatch(
            "add_controlnet",
            {
                "controlnet_name": "z_image_fun_canny.safetensors",
                "positive_node_id": "3",
                "negative_node_id": "4",
                "image_node_id": img_nid,
                "strength": 0.80,
            },
            zimage_wm,
        )
        loader_nids = zimage_wm.get_nodes_by_class("ModelPatchLoader")
        patch_nids = zimage_wm.get_nodes_by_class("ZImageFunControlnet")
        assert len(loader_nids) == 1
        assert len(patch_nids) == 1
        p_in = zimage_wm.workflow[patch_nids[0]]["inputs"]
        assert p_in["model"] == ["1", 0]
        assert p_in["model_patch"] == [loader_nids[0], 0]
        assert p_in["vae"] == [vae_nid, 0]
        assert p_in["image"] == [img_nid, 0]
        assert p_in["strength"] == 0.80
        assert "union_type" not in p_in
        # KSampler.model now routes through the patch node.
        assert zimage_wm.workflow["5"]["inputs"]["model"] == [patch_nids[0], 0]
        # Conditioning untouched.
        assert zimage_wm.workflow["5"]["inputs"]["positive"] == ["3", 0]
        assert zimage_wm.workflow["5"]["inputs"]["negative"] == ["4", 0]
        # No legacy Fun CN loader/apply nodes.
        assert zimage_wm.get_nodes_by_class("ZImageFunControlNetLoader") == []
        assert zimage_wm.get_nodes_by_class("ZImageFunControlNetApply") == []
        assert "z-image-turbo" in result.lower()


class TestAddControlNetLongCat:
    def test_returns_blocked_error(self, longcat_wm: WorkflowManager) -> None:
        agent = _make_agent()
        result, stop = agent._dispatch(
            "add_controlnet",
            {
                "controlnet_name": "whatever.safetensors",
                "positive_node_id": "3",
                "negative_node_id": "4",
                "strength": 0.7,
            },
            longcat_wm,
        )
        assert stop is False
        assert "longcat-image" in result.lower()
        assert "not supported" in result.lower()
        assert longcat_wm.get_nodes_by_class("ControlNetLoader") == []
        assert longcat_wm.get_nodes_by_class("ModelPatchLoader") == []
        assert longcat_wm.get_nodes_by_class("QwenImageDiffsynthControlnet") == []
        assert longcat_wm.get_nodes_by_class("ZImageFunControlnet") == []


# ---------------------------------------------------------------------------
# Standard SD workflow still uses the legacy code paths
# ---------------------------------------------------------------------------


class TestStandardArchStillWorks:
    def test_lora_uses_plain_lora_loader(self, wm: WorkflowManager) -> None:
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
        assert wm.get_nodes_by_class("LoraLoaderModelOnly") == []
        assert len(wm.get_nodes_by_class("LoraLoader")) == 1

    def test_controlnet_uses_standard_apply_advanced(self, wm: WorkflowManager) -> None:
        agent = _make_agent()
        # SD workflow needs a negative prompt for CN wiring; extend the minimal fixture.
        neg_nid = wm.add_node(
            "CLIPTextEncode", "Negative", clip=["1", 1], text="blurry"
        )
        wm.workflow["3"]["inputs"]["negative"] = [neg_nid, 0]
        img_nid = wm.add_node("LoadImage", image="ref.png")
        result, _ = agent._dispatch(
            "add_controlnet",
            {
                "controlnet_name": "control_v11p_sd15_canny.pth",
                "positive_node_id": "2",
                "negative_node_id": neg_nid,
                "image_node_id": img_nid,
                "preprocessor_class": "CannyEdgePreprocessor",
                "strength": 0.7,
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
            wm,
        )
        assert len(wm.get_nodes_by_class("ControlNetLoader")) == 1
        assert len(wm.get_nodes_by_class("ControlNetApplyAdvanced")) == 1
        assert wm.get_nodes_by_class("ModelPatchLoader") == []
        assert wm.get_nodes_by_class("QwenImageDiffsynthControlnet") == []
        assert wm.get_nodes_by_class("ZImageFunControlnet") == []
