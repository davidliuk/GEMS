# LoRA & ControlNet — usage guide

ComfyClaw's agent can insert LoRA adapters and ControlNet branches into any
workflow it builds or evolves, across four architecture families:

| Family | LoRA tool | LoRA node | ControlNet tool | ControlNet wiring |
|---|---|---|---|---|
| **SD 1.5 / SDXL / Flux** (standard) | `add_lora_loader` | `LoraLoader` (MODEL + CLIP) | `add_controlnet` | `ControlNetLoader` + `ControlNetApplyAdvanced` (wraps positive/negative) |
| **Qwen-Image-2512** (20B MMDiT) | `add_lora_loader` | `LoraLoaderModelOnly` | `add_controlnet` | `ModelPatchLoader` + `QwenImageDiffsynthControlnet` (patches MODEL) |
| **Z-Image-Turbo** (6B S3-DiT) | `add_lora_loader` | `LoraLoaderModelOnly` | `add_controlnet` | `ModelPatchLoader` + `ZImageFunControlnet` (patches MODEL) |
| **LongCat-Image** (6B pipeline) | — | *(blocked)* | — | *(blocked — no MODEL tensor)* |

You always call the **same two agent tools** (`add_lora_loader`,
`add_controlnet`). The harness detects the active architecture from the
workflow graph (checkpoint filename, `CLIPLoader.type`, custom loader
`class_type`) and dispatches to the correct nodes automatically. See
`ARCH_REGISTRY` in `comfyclaw/agent.py` for the detection rules.

---

## 1. When to reach for each tool

Both tools are most useful when the VLM verifier reports a structural defect
that prompt-tuning alone can't fix.

| Verifier complaint | Best fix | Tool |
|---|---|---|
| Plasticky skin, flat lighting, soft overall quality | Add a detail / realism / lighting LoRA | `add_lora_loader` |
| Wrong style, anime / oil paint intent | Style LoRA | `add_lora_loader` |
| Model too slow (30+ steps) | Speed / acceleration LoRA (Lightning, Turbo) | `add_lora_loader` |
| Wrong 3D layout / no depth separation | Depth ControlNet | `add_controlnet` |
| Structural composition needs to be locked | Canny / HED ControlNet | `add_controlnet` |
| Wrong human pose (SD / SDXL / Flux only) | Pose ControlNet | `add_controlnet` |

The agent's `lora-enhancement` and `controlnet-control` skills also suggest
these fixes based on verifier `fix_strategy` tokens.

---

## 2. Where to put the weights

The agent discovers files via `query_available_models("loras")` and
`query_available_models("controlnets")`. Both tools surface the standard
ComfyUI folders:

```
ComfyUI/models/
├── loras/                   LoRA safetensors for all archs
├── controlnets/             Standard SD/SDXL/Flux ControlNets
└── model_patches/           Qwen / Z-Image DiffSynth and Fun CN patch files
                             (what ModelPatchLoader reads)
```

> Qwen-Image-2512 and Z-Image-Turbo ControlNets **must live in
> `model_patches/`, not `controlnets/`** — they are loaded through
> `ModelPatchLoader`.

Recommended starter set (all verified to run end-to-end against ComfyUI):

| Arch | Role | Filename | Size | Source |
|---|---|---|---|---|
| Qwen-Image-2512 | LoRA (speed, 4-step) | `Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors` | 811 MB | `lightx2v/Qwen-Image-2512-Lightning` |
| Qwen-Image-2512 | ControlNet (canny) | `qwen_image_canny_diffsynth_controlnet.safetensors` | 2.2 GB | `Comfy-Org/Qwen-Image_ComfyUI` |
| Qwen-Image-2512 | ControlNet (depth) | `qwen_image_depth_diffsynth_controlnet.safetensors` | 2.2 GB | `Comfy-Org/Qwen-Image_ComfyUI` |
| Z-Image-Turbo   | LoRA (realism) | `Z-Image-Turbo-Radiant-Realism-Pro.safetensors` | 163 MB | community |
| Z-Image-Turbo   | LoRA (realism, lighter) | `Z-Image-Turbo-Realism-LoRA.safetensors` | 82 MB | community |
| Z-Image-Turbo   | ControlNet (union) | `Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors` | 6.3 GB | `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union` |

Download with `huggingface-cli download <repo> <file> --local-dir <target>`
or `wget` on the `resolve/main/…` URL.

---

## 3. Letting the agent do it (recommended)

Just tell the agent what you want in the ComfyClaw panel or on the CLI. The
agent reads the relevant skill (`lora-enhancement`, `controlnet-control`,
`qwen-image-2512`, `z-image-turbo`), queries installed weights, and calls
`add_lora_loader` / `add_controlnet` with the right arguments.

Serve mode:
```bash
comfyclaw serve
```

Then in the ComfyUI panel, try prompts like:

- _"photoreal portrait at golden hour — use a realism LoRA if one is
  available"_
- _"match the composition of this depth map; use a depth ControlNet"_
- _"fast generation, Lightning 4-step"_

CLI one-shot:
```bash
comfyclaw run \
  --workflow qwen_workflow_api.json \
  --prompt "a red fox at dawn, photorealistic, DSLR" \
  --iterations 2
```

The agent also repairs automatically: if the patch file is missing, the
CLIPLoader `type` is wrong, or a required `VAELoader` is absent, it'll try
to fix the graph up to `--max-repair-attempts` times.

---

## 4. Calling the tools directly (Python API)

If you want deterministic control, drive the agent's dispatcher yourself.

```python
from comfyclaw.agent import ClawAgent
from comfyclaw.workflow import WorkflowManager
from comfyclaw.memory import ClawMemory

wm    = WorkflowManager.from_file("qwen_workflow_api.json")
agent = ClawAgent(api_key="", model="anthropic/claude-sonnet-4-5")
mem   = ClawMemory()

agent._dispatch_tool(
    "add_lora_loader",
    {
        "lora_name":      "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
        "model_node_id":  "37",       # UNETLoader node ID
        "strength_model": 1.0,
    },
    wm, mem,
)

agent._dispatch_tool(
    "add_controlnet",
    {
        "controlnet_name":    "qwen_image_canny_diffsynth_controlnet.safetensors",
        "preprocessor_class": "",      # pass a pre-computed canny PNG
        "image_node_id":      "50",    # LoadImage node ID
        "positive_node_id":   "6",     # required for API symmetry
        "negative_node_id":   "7",
        "strength":           0.8,
    },
    wm, mem,
)

wm.to_file("qwen_with_lora_and_cn.json")
```

### Tool arguments

`add_lora_loader`:

| Arg | Required | SD / SDXL / Flux | Qwen / Z-Image | LongCat |
|---|---|---|---|---|
| `lora_name` | yes | filename from `loras/` | filename from `loras/` | *(blocked)* |
| `model_node_id` | yes | `CheckpointLoaderSimple` or `UNETLoader` (or previous LoRA) | `UNETLoader` (or previous LoRA) | — |
| `clip_node_id` | yes here, ignored elsewhere | `CheckpointLoaderSimple` or `CLIPLoader` | omit / ignored | — |
| `strength_model` | optional (default 1.0) | 0.4–1.0 typical | 0.4–1.0 typical | — |
| `strength_clip` | optional | 0.4–1.0 typical | ignored | — |

`add_controlnet`:

| Arg | SD / SDXL / Flux | Qwen / Z-Image | LongCat |
|---|---|---|---|
| `controlnet_name` | file in `controlnets/` | file in `model_patches/` | *(blocked)* |
| `preprocessor_class` | e.g. `CannyEdgePreprocessor`, or `""` | same — or `""` to pass a pre-computed map | — |
| `image_node_id` | optional | **required** (patch needs an image) | — |
| `positive_node_id`, `negative_node_id` | used to wrap conditioning | required for API symmetry, not wired | — |
| `strength` | 0.5–0.7 typical | 0.7–0.85 typical | — |
| `start_percent` / `end_percent` | applied | ignored | — |
| `union_type` | only for ControlNet-Union | ignored | — |

For Qwen / Z-Image, the workflow must already contain a `VAELoader`; the
tool will error otherwise. For all archs, call
`query_available_models("loras" | "controlnets")` first so you use an exact
filename that is actually installed.

---

## 5. End-to-end test on a live ComfyUI

Two scripts under `tests/` validate the full stack against a running server.
They are not part of the default pytest suite (offline-only).

```bash
# 1. Graph-shape / validation E2E (doesn't require the real weights to run,
#    errors out cleanly with "value_not_in_list" if files are missing).
python tests/e2e_lora_controlnet.py

# 2. Real-weight E2E — requires the starter weight set from § 2 to be
#    installed. Generates actual PNGs in ComfyUI/output/.
python tests/e2e_lora_controlnet_real.py
```

Both scripts honour `COMFYUI_ADDR` from `.env` (default `127.0.0.1:8188`).

Expected output from `e2e_lora_controlnet_real.py`:

```
=== Qwen-Image-2512 + Lightning 4-step LoRA ===
  SAVED: e2e_qwen_real_00006_.png

=== Qwen-Image-2512 + Canny DiffSynth ControlNet ===
  SAVED: e2e_qwen_real_00007_.png

…
SUMMARY
  PASS  qwen+LoRA
  PASS  qwen+CN
  PASS  qwen+LoRA+CN
  PASS  zimage+LoRA
  PASS  zimage+CN
  PASS  zimage+LoRA+CN
```

---

## 6. Troubleshooting

| Symptom | Diagnosis | Fix |
|---|---|---|
| `Node 'QwenImageFunControlNetLoader' not found on server` | Your agent is still on the old Fun-CN wiring (pre-`ARCH_REGISTRY`) | Pull latest `nips` branch — Qwen now uses `ModelPatchLoader` + `QwenImageDiffsynthControlnet` |
| `ControlNet requires an image_node_id for Qwen/Z-Image` | You called `add_controlnet` without `image_node_id` | Add a `LoadImage` to the workflow and pass its ID |
| `ControlNet requires a VAELoader somewhere upstream` | No `VAELoader` in the workflow | Add one (the decode side usually has one already — just connect it) |
| `value_not_in_list: … not in [...]` from `/prompt` | The filename isn't present on the ComfyUI server | `ls ComfyUI/models/{loras,controlnets,model_patches}/` and match exactly — names are case-sensitive |
| `missing_node_type: CannyEdgePreprocessor` | Preprocessor custom_node not installed | Install `comfyui_controlnet_aux` (Fannovel16), or pass `preprocessor_class=""` and supply a pre-computed canny map |
| LoRA silently ignored | Wrong arch (e.g. using an SDXL LoRA with Qwen) | LoRAs are **not** cross-compatible across archs — match the base model |
| Qwen image comes out over-saturated at cfg=1 | Too strong a speed LoRA | Reduce `strength_model` to 0.75, or switch to 8-step Lightning |
| LongCat complains that LoRA/CN isn't supported | Expected — LongCat pipeline nodes expose no MODEL tensor | Use prompt-level guidance or switch to Qwen / Z-Image |

---

## 7. Reference: where this is implemented

- `comfyclaw/agent.py` — `ARCH_REGISTRY`, `_detect_arch`, `_add_lora`,
  `_add_controlnet`
- `comfyclaw/skills/lora-enhancement/SKILL.md` — agent-facing playbook
- `comfyclaw/skills/controlnet-control/SKILL.md` — agent-facing playbook
- `comfyclaw/skills/qwen-image-2512/SKILL.md` — full Qwen recipe
- `comfyclaw/skills/z-image-turbo/SKILL.md` — full Z-Image recipe
- `comfyclaw/skills/longcat-image/SKILL.md` — LongCat limitations
- `tests/test_lora_controlnet_archs.py` — offline unit tests for the
  dispatcher
- `tests/e2e_lora_controlnet.py` — live-server graph-shape E2E
- `tests/e2e_lora_controlnet_real.py` — live-server real-weight E2E
