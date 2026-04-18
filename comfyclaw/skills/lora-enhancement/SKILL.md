---
name: lora-enhancement
description: >-
  Insert LoRA adapter nodes to correct defects the base model cannot fix
  through prompting alone — plasticky texture, flat lighting, distorted anatomy,
  style inconsistency, or soft overall quality. Use when the verifier reports
  any of these issues or when fix_strategy contains "inject_lora". Always query
  available LoRAs first; skip this skill if none are installed.
license: MIT
compatibility: ComfyClaw agent — call query_available_models("loras") before use.
allowed-tools: query_available_models add_lora_loader
metadata:
  author: davidliuk
  version: "0.2.0"
tags: [agent]
---

LoRA adapters inject targeted weight corrections that base-model prompting
cannot achieve. They are the most reliable fix for texture, anatomy, and
style issues — but only when a relevant LoRA is installed.

## Step 1 — Query first, always

```
query_available_models("loras")
```

If the list is empty or contains no relevant LoRA, stop here and try a
different fix strategy.

## Step 2 — Pick the right LoRA type

| Verifier issue | LoRA category | Filename keywords to look for |
|---|---|---|
| Plasticky / waxy skin or surfaces | Detail / texture | `add-detail`, `detail-tweaker`, `texture` |
| Distorted hands, fingers, or faces | Anatomy fix | `hand`, `better-hands`, `face-fix` |
| Flat or artificial lighting | Lighting | `lighting`, `illumination`, `cinematic-light` |
| Style inconsistency or genre mismatch | Style | `oil-paint`, `photorealism`, `anime`, `3d` |
| Soft or blurry overall quality | Quality booster | `xl-detail`, `quality`, `sharpness` |

Pick the single best match for the top verifier issue. Adding more than two
LoRAs in one iteration rarely helps and often introduces new artifacts.

## Step 3 — Add with `add_lora_loader`

**For SD / SDXL / Flux pipelines** (CheckpointLoaderSimple or UNETLoader without Qwen):

```
add_lora_loader(
  lora_name       = "<exact filename from query>",
  model_node_id   = "<UNETLoader or CheckpointLoaderSimple node ID>",
  clip_node_id    = "<CLIPLoader or CheckpointLoaderSimple node ID>",
  strength_model  = 0.75,
  strength_clip   = 0.75,
)
```

**For Qwen-Image-2512 (MMDiT) pipelines** — omit `clip_node_id`; the backend
automatically uses `LoraLoaderModelOnly` (model-only, no CLIP tower):

```
add_lora_loader(
  lora_name      = "<exact filename from query>",
  model_node_id  = "<UNETLoader or existing LoraLoaderModelOnly node ID>",
  strength_model = 0.75,
)
```

LoRA loaders chain in series between the model loader and the sampler.

## Strength guidelines

| LoRA type | Starting strength | Reduce if… |
|---|---|---|
| Detail / texture | 0.70 | Background starts looking over-processed |
| Style | 0.55 | Style overwhelms the subject |
| Anatomy fix | 0.80 | New anatomy artifacts appear |
| Quality booster | 0.65 | Output looks overly "HD-remastered" |

## Step 4 — Update the positive prompt

Some LoRAs require trigger tokens to activate (e.g. `<lora:hand_fix:0.8>`
or just `detailed hands`). Check the LoRA filename for hints; if you know
the trigger word, add it to the positive CLIPTextEncode text.
