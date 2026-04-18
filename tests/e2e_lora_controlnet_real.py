"""Full end-to-end test: submit real LoRA + ControlNet workflows to ComfyUI
and verify images are actually generated.

Uses the weights downloaded by the operator into:
  - /workspace/ComfyUI/models/loras/
  - /workspace/ComfyUI/models/model_patches/

Unlike tests/e2e_lora_controlnet.py (which only checks graph structure),
this script polls /history and confirms actual images land on disk.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

from comfyclaw.agent import ClawAgent
from comfyclaw.workflow import WorkflowManager

COMFY_ADDR = os.environ.get("COMFYUI_ADDR", "127.0.0.1:8188")
SERVER_URL = f"http://{COMFY_ADDR}"


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


def _post_prompt(workflow: dict) -> dict:
    payload = json.dumps({"prompt": workflow, "client_id": "e2e_real"}).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return {"status": resp.status, "body": json.loads(resp.read())}
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            body_parsed = json.loads(body)
        except json.JSONDecodeError:
            body_parsed = body
        return {"status": e.code, "body": body_parsed}


def _wait(prompt_id: str, timeout_s: int = 420) -> dict | None:
    deadline = time.time() + timeout_s
    last_msg = 0.0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"{SERVER_URL}/history/{prompt_id}", timeout=10
            ) as r:
                hist = json.loads(r.read())
                if prompt_id in hist:
                    return hist[prompt_id]
        except Exception:
            pass
        if time.time() - last_msg > 15:
            print(f"    ...still waiting ({int(deadline - time.time())}s left)")
            last_msg = time.time()
        time.sleep(2)
    return None


def _run(label: str, wm: WorkflowManager) -> bool:
    print(f"\n=== {label} ===")
    resp = _post_prompt(wm.workflow)
    print(f"  /prompt → HTTP {resp['status']}")
    if resp["status"] != 200:
        print(f"  ERR: {json.dumps(resp['body'], ensure_ascii=False)[:600]}")
        return False
    pid = resp["body"]["prompt_id"]
    print(f"  prompt_id={pid}; polling...")
    hist = _wait(pid)
    if hist is None:
        print("  FAIL: timeout")
        return False
    status = hist.get("status", {})
    outputs = hist.get("outputs", {})
    ok = status.get("status_str") == "success" or any(
        "images" in v for v in outputs.values()
    )
    print(f"  status={status.get('status_str')}  messages={len(status.get('messages', []))}")
    for nid, out in outputs.items():
        for im in out.get("images", []):
            print(
                f"  SAVED: {im.get('filename')} ({im.get('type')}, "
                f"subfolder={im.get('subfolder') or '-'})"
            )
    return ok


# ---------------------------------------------------------------------------
# Workflow builders (match what's available on this ComfyUI install)
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
            "inputs": {
                "clip": ["2", 0],
                "text": "a red fox in a sunlit autumn forest, highly detailed, cinematic lighting",
            },
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": "blurry, low quality, watermark"},
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
                # Lightning-4steps LoRA enables 4-step inference; keep 6 for CN headroom.
                "steps": 8,
                "cfg": 1.0,
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
            "inputs": {"images": ["9", 0], "filename_prefix": "e2e_qwen_real"},
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
            "inputs": {
                "clip": ["2", 0],
                "text": "a realistic portrait of a young woman in soft sunset light",
            },
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
            "inputs": {"images": ["9", 0], "filename_prefix": "e2e_zimg_real"},
        },
    }


def main() -> int:
    agent = _make_agent()
    results: dict[str, bool] = {}

    # --- 1. Qwen + LoRA ---
    wm = WorkflowManager(_qwen_baseline())
    r, _ = agent._dispatch(
        "add_lora_loader",
        {
            "lora_name": "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
            "model_node_id": "1",
            "strength_model": 1.0,
        },
        wm,
    )
    print("Qwen LoRA insert:", r.splitlines()[0])
    results["qwen+LoRA"] = _run("Qwen-Image-2512 + Lightning 4-step LoRA", wm)

    # --- 2. Qwen + ControlNet (canny, no preprocessor — raw image as control) ---
    wm = WorkflowManager(_qwen_baseline())
    img_nid = wm.add_node("LoadImage", image="example.png")
    r, _ = agent._dispatch(
        "add_controlnet",
        {
            "controlnet_name": "qwen_image_canny_diffsynth_controlnet.safetensors",
            "positive_node_id": "5",
            "negative_node_id": "6",
            "image_node_id": img_nid,
            "strength": 0.75,
        },
        wm,
    )
    print("Qwen CN insert:", r.splitlines()[0])
    results["qwen+CN"] = _run("Qwen-Image-2512 + Canny DiffSynth ControlNet", wm)

    # --- 3. Qwen + LoRA + ControlNet (combined) ---
    wm = WorkflowManager(_qwen_baseline())
    agent._dispatch(
        "add_lora_loader",
        {
            "lora_name": "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
            "model_node_id": "1",
            "strength_model": 1.0,
        },
        wm,
    )
    img_nid = wm.add_node("LoadImage", image="example.png")
    agent._dispatch(
        "add_controlnet",
        {
            "controlnet_name": "qwen_image_depth_diffsynth_controlnet.safetensors",
            "positive_node_id": "5",
            "negative_node_id": "6",
            "image_node_id": img_nid,
            "strength": 0.60,
        },
        wm,
    )
    results["qwen+LoRA+CN"] = _run("Qwen-Image-2512 + LoRA + Depth CN", wm)

    # --- 4. Z-Image + LoRA ---
    wm = WorkflowManager(_zimage_baseline())
    r, _ = agent._dispatch(
        "add_lora_loader",
        {
            "lora_name": "Z-Image-Turbo-Radiant-Realism-Pro.safetensors",
            "model_node_id": "1",
            "strength_model": 0.9,
        },
        wm,
    )
    print("Z-Image LoRA insert:", r.splitlines()[0])
    results["zimage+LoRA"] = _run("Z-Image-Turbo + Radiant Realism LoRA", wm)

    # --- 5. Z-Image + ControlNet (no preprocessor) ---
    wm = WorkflowManager(_zimage_baseline())
    img_nid = wm.add_node("LoadImage", image="example.png")
    r, _ = agent._dispatch(
        "add_controlnet",
        {
            "controlnet_name": "Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors",
            "positive_node_id": "5",
            "negative_node_id": "6",
            "image_node_id": img_nid,
            "strength": 0.75,
        },
        wm,
    )
    print("Z-Image CN insert:", r.splitlines()[0])
    results["zimage+CN"] = _run("Z-Image-Turbo + Fun ControlNet Union", wm)

    # --- 6. Z-Image + LoRA + ControlNet (combined) ---
    wm = WorkflowManager(_zimage_baseline())
    agent._dispatch(
        "add_lora_loader",
        {
            "lora_name": "Z-Image-Turbo-Realism-LoRA.safetensors",
            "model_node_id": "1",
            "strength_model": 0.7,
        },
        wm,
    )
    img_nid = wm.add_node("LoadImage", image="example.png")
    agent._dispatch(
        "add_controlnet",
        {
            "controlnet_name": "Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors",
            "positive_node_id": "5",
            "negative_node_id": "6",
            "image_node_id": img_nid,
            "strength": 0.60,
        },
        wm,
    )
    results["zimage+LoRA+CN"] = _run("Z-Image-Turbo + LoRA + ControlNet combined", wm)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
