---
name: planner-playbook
description: >-
  Decision table mapping the current workflow / model / verifier feedback to the
  right next action (which built-in skill to read, which structural upgrade to
  apply, which parameters are architecture-specific).  Read this FIRST on any
  iteration where you are not sure which built-in skill to consult — it is the
  authoritative index of built-in skills and model-specific gotchas and is much
  cheaper than guessing wrong and burning an iteration.
license: MIT
metadata:
  author: "comfyclaw"
  version: "1.0.0"
tags: [agent]
---

# planner-playbook

## Purpose
A compact decision table that maps **what you see in the workflow / verifier
feedback** to **which built-in skill to read next** and **which architecture
gotchas to watch for**.  Consult this whenever the planning step is ambiguous
— it is cheaper than picking the wrong upgrade and wasting an iteration.

## Decision table — pick one row, then `read_skill` the listed skill

### By workflow state

| Signal                                            | Action                                                                                          |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Workflow is EMPTY (no nodes)                      | `read_skill("workflow-builder")` FIRST, then `query_available_models` to pick an architecture.  |

### By active model (detected via `inspect_workflow`)

| Active model contains…                            | Read this skill FIRST                                                                           |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `QwenImageModelLoader`                            | `qwen-image-2512` — LoRA = `LoraLoaderModelOnly`.                                               |
| `longcat`                                         | `longcat-image` — uses CFGNorm + FluxGuidance, not ModelSamplingAuraFlow.  LoRA **not** supported via standard tools. |
| `z_image` / `z-image`                             | `z-image-turbo` — cfg=1, sampler=`res_multistep`, `ConditioningZeroOut` for negative.  Do NOT change cfg/sampler.  LoRA = `LoraLoaderModelOnly`. |
| `lcm`                                             | `dreamshaper8-lcm` — different steps/cfg/sampler; read BEFORE any sampler tuning.               |

### By verifier feedback / region issue

| Verifier symptom                                  | Read this skill                                                                                 |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Plasticky skin / poor texture                     | `lora-enhancement` → detail LoRA.                                                               |
| Wrong anatomy (hands, fingers)                    | `lora-enhancement` → anatomy LoRA.                                                              |
| Style inconsistency                               | `lora-enhancement` → style LoRA.                                                                |
| Subject / background bleed or multi-object dropout| `regional-control`.                                                                             |
| Multiple objects with spatial layout              | `spatial`.                                                                                      |
| Low resolution / soft fine detail                 | `hires-fix`.                                                                                    |
| Localised artifact in one area                    | `add_inpaint_pass` (tool) — no skill read needed.                                               |
| Text / sign / label in the image                  | `text-rendering`.                                                                               |

### By user intent

| User asks for…                                    | Read this skill                                                                                 |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Photorealistic image                              | `photorealistic`.                                                                               |
| High quality / sharp                              | `high-quality`.                                                                                 |
| Creative / artistic                               | `creative`.                                                                                     |
| Flat prompt that needs artistic depth             | `prompt-artist`.                                                                                |

## Structural-upgrade priority (iteration ≥ 2 with verifier feedback)

- Do **not** rely on prompt-only tweaks — they plateau within 1–2 iterations.
- **Prefer** a structural upgrade: LoRA / hires-fix / regional-control / inpaint.
- If any `region_issue.fix_strategies` entry contains `inject_lora_*`, you **must**
  attempt that LoRA upgrade.  Call `query_available_models` first; if a matching
  LoRA exists, read `lora-enhancement` and apply it.
- Always **combine** a prompt refinement with the structural upgrade in the
  same iteration.
- Fall back to prompt-only only when no LoRA / inpaint models are installed
  or fix strategies are exclusively prompt-related.

## Node-parameter constraints (violations cause ComfyUI HTTP 400)

- `UNETLoader.weight_dtype` ∈ `{"default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"}`.
  **Never** `"fp16"` / `"fp32"`.
- Apple MPS cannot run FP8 models.  On `Float8_e4m3fn` MPS error, set
  `weight_dtype="default"` and stop tuning dtype.
- LoRA class is `LoraLoader` for SD / SDXL / Flux, `LoraLoaderModelOnly` for
  MMDiT / S3-DiT (Qwen-Image-2512, Z-Image-Turbo).  The `add_lora_loader` tool
  selects the right class automatically.
- LongCat-Image does **not** support LoRA via standard tools — use `set_param`
  to tune steps / guidance_scale instead, and consult `longcat-image` for
  enhancement options.
- `ConditioningAverage.conditioning_to_strength` is clamped to `[0.0, 1.0]`;
  the `add_regional_attention` tool enforces this for you, but if you wire it
  manually, keep `foreground_weight` in range.
