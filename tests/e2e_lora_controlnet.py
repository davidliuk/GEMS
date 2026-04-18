"""End-to-end verification of LoRA/ControlNet wiring against a live ComfyUI.

This script builds workflows with ClawAgent's LoRA/ControlNet tools for each
supported architecture (Qwen-Image-2512, Z-Image-Turbo, LongCat-Image, standard
SD), then:

  1. Asserts the generated graph uses the expected node classes.
  2. POSTs the prompt to the live ComfyUI server and captures the server's
     validation response (so we can see whether ComfyUI's own schema check
     accepts the graph structure, independent of whether the model weight
     files are actually present on disk).

Run from repo root:

    source .venv/bin/activate
    python tests/e2e_lora_controlnet.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

from comfyclaw.agent import ClawAgent
from comfyclaw.workflow import WorkflowManager

COMFY_ADDR = os.environ.get("COMFYUI_ADDR", "127.0.0.1:8188")
SERVER_URL = f"http://{COMFY_ADDR}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent() -> ClawAgent:
    agent = ClawAgent.__new__(ClawAgent)
    agent.model = "anthropic/claude-test"
    agent.server_address = COMFY_ADDR
    from comfyclaw.skill_manager import SkillManager

    agent.skill_manager = SkillManager(None)
    agent.on_change = None
    agent.on_agent_event = None
    agent.max_tool_rounds = 10
    agent.pinned_image_model = None
    from comfyclaw.stage_router import StageRouter

    agent.stage_router = StageRouter(enabled=False)
    return agent


def _post_prompt(workflow: dict) -> dict[str, Any]:
    """POST workflow to /api/prompt and return the parsed response body."""
    payload = json.dumps({"prompt": workflow, "client_id": "e2e_test"}).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return {"status": resp.status, "body": json.loads(resp.read())}
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            body_parsed = json.loads(body)
        except json.JSONDecodeError:
            body_parsed = body
        return {"status": e.code, "body": body_parsed}


def _object_info() -> dict[str, Any]:
    with urllib.request.urlopen(f"{SERVER_URL}/object_info", timeout=10) as r:
        return json.loads(r.read())


def _banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _summarize(label: str, ok: bool, detail: str = "") -> None:
    sym = "PASS" if ok else "FAIL"
    line = f"  [{sym}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)


def _print_prompt_errors(resp: dict, case: str) -> None:
    """Pretty-print the server's per-node validation errors for diagnosis.

    A graph whose only errors are 'value not in list' (for missing model files)
    means the graph STRUCTURE is correct — the server just can't find the
    weights. That's the expected state here (empty loras/, empty controlnets/).
    """
    body = resp.get("body")
    if not isinstance(body, dict):
        return
    top_err = body.get("error") or {}
    err_type = top_err.get("type") if isinstance(top_err, dict) else None
    _summarize(
        f"case {case}: no missing_node_type (graph structurally valid)",
        err_type != "missing_node_type",
        f"err type={err_type}",
    )
    node_errors = body.get("node_errors") or {}
    if not node_errors:
        return
    print("    per-node errors:")
    all_structural = True
    for nid, ne in node_errors.items():
        ct = ne.get("class_type", "?")
        errs = ne.get("errors", [])
        for e in errs:
            etype = e.get("type")
            msg = e.get("message", "")
            details = e.get("details", "")
            print(f"      node {nid} ({ct}): {etype} — {msg} | {details[:120]}")
            # "value_not_in_list" means the filename isn't present on disk but
            # the graph shape is fine.
            if etype not in {"value_not_in_list", "invalid_input_type"}:
                all_structural = False
    _summarize(
        f"case {case}: all errors are missing-file (not structural)",
        all_structural,
    )


# ---------------------------------------------------------------------------
# Workflow builders (minimal but schema-faithful per architecture)
# ---------------------------------------------------------------------------


def _qwen_baseline() -> dict:
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "qwen_image_bf16.safetensors",
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_2.5_vl_7b.safetensors",
                "type": "qwen_image",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_vae.safetensors"},
        },
        "4": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["1", 0], "shift": 3.0},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": "a red fox in a forest"},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": "blurry, low quality"},
        },
        "7": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0],
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0],
                "seed": 42,
                "steps": 20,
                "cfg": 4.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["3", 0]},
        },
        "10": {
            "class_type": "SaveImage",
            "inputs": {"images": ["9", 0], "filename_prefix": "e2e_qwen"},
        },
    }


def _zimage_baseline() -> dict:
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "z_image_turbo_bf16.safetensors",
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "ae.safetensors"},
        },
        "4": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["1", 0], "shift": 3.0},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": "a sunlit beach at golden hour"},
        },
        "6": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["5", 0]},
        },
        "7": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0],
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0],
                "seed": 42,
                "steps": 8,
                "cfg": 1.0,
                "sampler_name": "res_multistep",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["3", 0]},
        },
        "10": {
            "class_type": "SaveImage",
            "inputs": {"images": ["9", 0], "filename_prefix": "e2e_zimg"},
        },
    }


def _longcat_baseline() -> dict:
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "longcat_image_bf16.safetensors",
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_2.5_vl_7b.safetensors",
                "type": "longcat_image",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "ae.safetensors"},
        },
        "4": {
            "class_type": "CFGNorm",
            "inputs": {"model": ["1", 0], "strength": 1.0},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": "a vintage teahouse"},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": "low quality"},
        },
        "7": {
            "class_type": "FluxGuidance",
            "inputs": {"conditioning": ["5", 0], "guidance": 4.0},
        },
        "8": {
            "class_type": "FluxGuidance",
            "inputs": {"conditioning": ["6", 0], "guidance": 4.0},
        },
        "9": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        },
        "10": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0],
                "positive": ["7", 0],
                "negative": ["8", 0],
                "latent_image": ["9", 0],
                "seed": 42,
                "steps": 20,
                "cfg": 4.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "11": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["10", 0], "vae": ["3", 0]},
        },
        "12": {
            "class_type": "SaveImage",
            "inputs": {"images": ["11", 0], "filename_prefix": "e2e_longcat"},
        },
    }


# ---------------------------------------------------------------------------
# Diagnostic: show which Qwen/Z-Image CN node classes the server exposes
# ---------------------------------------------------------------------------


def probe_server() -> set[str]:
    _banner("1. Probing ComfyUI server for required node classes")
    info = _object_info()
    classes = set(info.keys())

    wanted = [
        # Standard
        "LoraLoader",
        "LoraLoaderModelOnly",
        "ControlNetLoader",
        "ControlNetApplyAdvanced",
        # Agent's assumed Fun CN class names
        "QwenImageFunControlNetLoader",
        "QwenImageFunControlNetApply",
        "ZImageFunControlNetLoader",
        "ZImageFunControlNetApply",
        # Actual Fun CN classes in this ComfyUI
        "ZImageFunControlnet",
        "QwenImageDiffsynthControlnet",
        "ModelPatchLoader",
        "SetUnionControlNetType",
    ]
    for name in wanted:
        print(f"  {'OK  ' if name in classes else 'MISS'}  {name}")
    return classes


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_qwen_lora() -> None:
    _banner("2. Qwen-Image-2512 + LoRA (LoraLoaderModelOnly)")
    wm = WorkflowManager(_qwen_baseline())
    agent = _make_agent()
    # Use a placeholder filename — the LoRAs folder is empty, so ComfyUI will
    # reject on file presence, not on graph structure.
    result, _ = agent._dispatch(
        "add_lora_loader",
        {
            "lora_name": "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors",
            "model_node_id": "1",
            "strength_model": 0.8,
        },
        wm,
    )
    print("  agent:", result.splitlines()[0])

    lora_nids = wm.get_nodes_by_class("LoraLoaderModelOnly")
    _summarize(
        "exactly 1 LoraLoaderModelOnly node inserted",
        len(lora_nids) == 1,
        f"count={len(lora_nids)}",
    )
    # The ModelSamplingAuraFlow input 'model' should now come from the LoRA.
    sampling = wm.workflow["4"]["inputs"]["model"]
    _summarize(
        "ModelSamplingAuraFlow.model rewired to LoRA",
        sampling == [lora_nids[0], 0],
        f"model={sampling}",
    )
    _summarize("no plain LoraLoader used", wm.get_nodes_by_class("LoraLoader") == [])

    # Submit to /prompt — expect a specific error about missing lora file, NOT
    # about graph structure.
    resp = _post_prompt(wm.workflow)
    print(f"  /prompt → HTTP {resp['status']}")
    _print_prompt_errors(resp, "2")


def test_zimage_lora() -> None:
    _banner("3. Z-Image-Turbo + LoRA (LoraLoaderModelOnly)")
    wm = WorkflowManager(_zimage_baseline())
    agent = _make_agent()
    result, _ = agent._dispatch(
        "add_lora_loader",
        {
            "lora_name": "z_image_realism.safetensors",
            "model_node_id": "1",
            "strength_model": 0.75,
        },
        wm,
    )
    print("  agent:", result.splitlines()[0])
    lora_nids = wm.get_nodes_by_class("LoraLoaderModelOnly")
    _summarize(
        "exactly 1 LoraLoaderModelOnly node inserted",
        len(lora_nids) == 1,
        f"count={len(lora_nids)}",
    )
    sampling = wm.workflow["4"]["inputs"]["model"]
    _summarize(
        "ModelSamplingAuraFlow.model rewired to LoRA",
        sampling == [lora_nids[0], 0],
        f"model={sampling}",
    )
    resp = _post_prompt(wm.workflow)
    print(f"  /prompt → HTTP {resp['status']}")
    _print_prompt_errors(resp, "3")


def test_longcat_lora_blocked() -> None:
    _banner("4. LongCat-Image + LoRA (MUST be blocked)")
    wm = WorkflowManager(_longcat_baseline())
    agent = _make_agent()
    result, _ = agent._dispatch(
        "add_lora_loader",
        {"lora_name": "whatever.safetensors", "model_node_id": "1"},
        wm,
    )
    print("  agent:", result.splitlines()[0])
    _summarize(
        "workflow unchanged (no LoraLoader* inserted)",
        wm.get_nodes_by_class("LoraLoader") == []
        and wm.get_nodes_by_class("LoraLoaderModelOnly") == [],
    )
    _summarize(
        "error mentions 'not supported'",
        "not supported" in result.lower() and "longcat" in result.lower(),
    )


def _add_loadimage(wm: WorkflowManager) -> str:
    """Add a LoadImage node pointing at a real image on the ComfyUI server."""
    return wm.add_node("LoadImage", image="example.png")


def test_qwen_controlnet(server_classes: set[str]) -> None:
    _banner("5. Qwen-Image-2512 + ControlNet (model-patch)")
    wm = WorkflowManager(_qwen_baseline())
    agent = _make_agent()
    img_nid = _add_loadimage(wm)
    result, _ = agent._dispatch(
        "add_controlnet",
        {
            "controlnet_name": "qwen_image_diffsynth_canny.safetensors",
            "positive_node_id": "5",
            "negative_node_id": "6",
            "image_node_id": img_nid,
            "strength": 0.75,
        },
        wm,
    )
    print("  agent:", result.splitlines()[0])

    _summarize(
        "agent generated ModelPatchLoader",
        len(wm.get_nodes_by_class("ModelPatchLoader")) == 1,
    )
    patch_nids = wm.get_nodes_by_class("QwenImageDiffsynthControlnet")
    _summarize(
        "agent generated QwenImageDiffsynthControlnet",
        len(patch_nids) == 1,
    )
    _summarize(
        "KSampler.model rewired to patch output",
        patch_nids and wm.workflow["8"]["inputs"]["model"] == [patch_nids[0], 0],
    )
    _summarize(
        "positive/negative conditioning unchanged",
        wm.workflow["8"]["inputs"]["positive"] == ["5", 0]
        and wm.workflow["8"]["inputs"]["negative"] == ["6", 0],
    )
    print(
        f"  server has ModelPatchLoader={('ModelPatchLoader' in server_classes)}, "
        f"QwenImageDiffsynthControlnet={('QwenImageDiffsynthControlnet' in server_classes)}"
    )

    resp = _post_prompt(wm.workflow)
    print(f"  /prompt → HTTP {resp['status']}")
    _print_prompt_errors(resp, "5")


def test_zimage_controlnet(server_classes: set[str]) -> None:
    _banner("6. Z-Image-Turbo + ControlNet (model-patch)")
    wm = WorkflowManager(_zimage_baseline())
    agent = _make_agent()
    img_nid = _add_loadimage(wm)
    result, _ = agent._dispatch(
        "add_controlnet",
        {
            "controlnet_name": "z_image_fun_canny.safetensors",
            "positive_node_id": "5",
            "negative_node_id": "6",
            "image_node_id": img_nid,
            "strength": 0.80,
        },
        wm,
    )
    print("  agent:", result.splitlines()[0])
    _summarize(
        "agent generated ModelPatchLoader",
        len(wm.get_nodes_by_class("ModelPatchLoader")) == 1,
    )
    patch_nids = wm.get_nodes_by_class("ZImageFunControlnet")
    _summarize(
        "agent generated ZImageFunControlnet",
        len(patch_nids) == 1,
    )
    _summarize(
        "KSampler.model rewired to patch output",
        patch_nids and wm.workflow["8"]["inputs"]["model"] == [patch_nids[0], 0],
    )
    _summarize(
        "positive/negative conditioning unchanged",
        wm.workflow["8"]["inputs"]["positive"] == ["5", 0]
        and wm.workflow["8"]["inputs"]["negative"] == ["6", 0],
    )
    print(
        f"  server has ModelPatchLoader={('ModelPatchLoader' in server_classes)}, "
        f"ZImageFunControlnet={('ZImageFunControlnet' in server_classes)}"
    )

    resp = _post_prompt(wm.workflow)
    print(f"  /prompt → HTTP {resp['status']}")
    _print_prompt_errors(resp, "6")


def test_longcat_controlnet_blocked() -> None:
    _banner("7. LongCat-Image + ControlNet (MUST be blocked)")
    wm = WorkflowManager(_longcat_baseline())
    agent = _make_agent()
    result, _ = agent._dispatch(
        "add_controlnet",
        {
            "controlnet_name": "whatever.safetensors",
            "positive_node_id": "5",
            "negative_node_id": "6",
            "strength": 0.7,
        },
        wm,
    )
    print("  agent:", result.splitlines()[0])
    _summarize(
        "workflow unchanged",
        wm.get_nodes_by_class("ControlNetLoader") == []
        and wm.get_nodes_by_class("ModelPatchLoader") == []
        and wm.get_nodes_by_class("QwenImageDiffsynthControlnet") == []
        and wm.get_nodes_by_class("ZImageFunControlnet") == [],
    )
    _summarize(
        "error mentions 'not supported'",
        "not supported" in result.lower() and "longcat" in result.lower(),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _wait_for_prompt(prompt_id: str, timeout_s: int = 300) -> dict | None:
    """Poll /history/<id> until the prompt completes or times out."""
    import time as _t

    deadline = _t.time() + timeout_s
    while _t.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"{SERVER_URL}/history/{prompt_id}", timeout=10
            ) as r:
                hist = json.loads(r.read())
                if prompt_id in hist:
                    return hist[prompt_id]
        except Exception:
            pass
        _t.sleep(2)
    return None


def test_baseline(label: str, builder, poll: bool = False) -> None:
    _banner(f"0.{label} Baseline sanity — {label} (no LoRA / no CN)")
    wm = WorkflowManager(builder())
    resp = _post_prompt(wm.workflow)
    print(f"  /prompt → HTTP {resp['status']}")
    body = resp["body"]
    if isinstance(body, dict):
        queued = resp["status"] == 200 and "prompt_id" in body
        _summarize(
            f"{label} baseline accepted (prompt_id returned)",
            queued,
            f"body keys={list(body.keys())[:4]}",
        )
        _print_prompt_errors(resp, f"0.{label}")
        if queued and poll:
            pid = body["prompt_id"]
            print(f"  polling /history/{pid} (up to 300s)...")
            hist = _wait_for_prompt(pid)
            if hist is None:
                _summarize(f"{label} image generated", False, "timed out")
                return
            status = hist.get("status", {})
            outputs = hist.get("outputs", {})
            completed = status.get("completed") or any(
                "images" in v for v in outputs.values()
            )
            _summarize(
                f"{label} image generated",
                bool(completed),
                f"status={status.get('status_str', '?')}",
            )
            for nid, out in outputs.items():
                if "images" in out:
                    for im in out["images"]:
                        print(f"    saved: {im.get('filename')} ({im.get('type')})")


def main() -> int:
    print(f"ComfyUI server: {SERVER_URL}")
    server_classes = probe_server()
    # Baseline sanity — do the arch-specific plain workflows even parse?
    # Poll Z-Image (fast, ~8 steps) to confirm actual image generation.
    test_baseline("qwen", _qwen_baseline)
    test_baseline("zimage", _zimage_baseline, poll=True)
    test_baseline("longcat", _longcat_baseline)
    # LoRA
    test_qwen_lora()
    test_zimage_lora()
    test_longcat_lora_blocked()
    # ControlNet
    test_qwen_controlnet(server_classes)
    test_zimage_controlnet(server_classes)
    test_longcat_controlnet_blocked()
    return 0


if __name__ == "__main__":
    sys.exit(main())
