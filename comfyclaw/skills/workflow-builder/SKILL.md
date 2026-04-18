---
name: workflow-builder
description: >-
  Build a complete ComfyUI workflow from scratch. Use when the starting workflow
  is empty, when you need to construct a new pipeline topology, or when the user
  asks to "create a workflow", "build from scratch", "start fresh", or
  "generate a new workflow". Covers SD 1.5, SDXL, Flux, and DiT (Qwen/HunyuanDiT)
  architectures with node-by-node construction recipes and wiring patterns.
  Always call query_available_models('checkpoints') first to detect which
  architecture to use.
license: MIT
metadata:
  author: comfyclaw
  version: "1.0.0"
tags: [agent]
---

# Workflow Builder — Constructing ComfyUI Pipelines From Scratch

This skill teaches you to build complete, working ComfyUI workflows using
`add_node` and `connect_nodes`. **Every recipe below is tested and produces
a valid, queueable graph.**

## Output Slot Reference (CRITICAL — memorize these)

When wiring nodes via `add_node(... input_name=[src_node_id, OUTPUT_SLOT])`,
you must use the correct output slot index. Wrong slot = ComfyUI 400 error.

| Node class | Slot 0 | Slot 1 | Slot 2 |
|---|---|---|---|
| **CheckpointLoaderSimple** | MODEL | CLIP | VAE |
| **UNETLoader** | MODEL | — | — |
| **CLIPLoader** | CLIP | — | — |
| **DualCLIPLoader** | CLIP | — | — |
| **VAELoader** | VAE | — | — |
| **LoraLoader** | MODEL | CLIP | — |
| **LoraLoaderModelOnly** | MODEL | — | — |
| **CLIPTextEncode** | CONDITIONING | — | — |
| **EmptyLatentImage** | LATENT | — | — |
| **EmptySD3LatentImage** | LATENT | — | — |
| **KSampler** | LATENT | — | — |
| **VAEDecode** | IMAGE | — | — |
| **ModelSamplingAuraFlow** | MODEL | — | — |
| **FluxGuidance** | CONDITIONING | — | — |
| **ControlNetLoader** | CONTROL_NET | — | — |
| **ControlNetApplyAdvanced** | positive (CONDITIONING) | negative (CONDITIONING) | — |
| **LatentUpscaleBy** | LATENT | — | — |
| **LongCatImageModelLoader** | LONGCAT_PIPE | — | — |
| **LongCatImageTextToImage** | IMAGE | — | — |

**Common wiring mistakes to avoid:**
- CheckpointLoaderSimple: CLIP is slot **1** (not 0), VAE is slot **2** (not 1)
- UNETLoader: only has slot **0** (MODEL) — no CLIP or VAE outputs
- KSampler output is LATENT at slot **0** — feed this to VAEDecode's `samples` input
- VAEDecode output is IMAGE at slot **0** — feed this to SaveImage's `images` input

## Step 0 — Detect the Architecture

Before building anything, discover what models the server has:

```
query_available_models("checkpoints")
query_available_models("diffusion_models")
```

Then match the filename to an architecture:

| Filename pattern | Architecture | CLIPLoader type | Go to |
|---|---|---|---|
| `*sd15*`, `*sd_1*`, `dreamshaper*`, `realistic*`, `deliberate*` | SD 1.5 | — | Recipe A |
| `*sdxl*`, `*sd_xl*`, `juggernaut*`, `zavychroma*` | SDXL | — | Recipe B |
| `*flux*` | Flux | — | Recipe C |
| `*qwen_image*` | Qwen-Image-2512 | `qwen_image` | Recipe D |
| `*z_image*` | Z-Image-Turbo | `qwen3_4b` | Recipe G |
| `*longcat*` | LongCat-Image | — (custom nodes) | Recipe H |
| `*hunyuan*` | HunyuanDiT | — | Recipe E |
| `*sd3*`, `*sd_3*` | SD 3.x | — | Recipe F |

If multiple models are available, prefer: Z-Image-Turbo > Qwen > LongCat-Image > Flux > SDXL > SD 1.5
(unless the user specifies a model or the pinned image model overrides).

**LongCat-Image note:** This model uses custom pipeline nodes — no UNETLoader, no KSampler.
Use Recipe H. LoRA and ControlNet are NOT available for this arch; use prompt tuning + parameter
tuning (guidance_scale, steps) instead.

If checkpoints list is empty, also check `diffusion_models` — DiT-based models
(Flux, Qwen, Z-Image, HunyuanDiT) use UNETLoader with files in `diffusion_models/`.

---

## Recipe A — SD 1.5 (txt2img)

Classic Stable Diffusion 1.5 pipeline. 7 nodes.

```
CheckpointLoaderSimple ──MODEL──► KSampler ──LATENT──► VAEDecode ──IMAGE──► SaveImage
        ├──CLIP──► CLIPTextEncode (positive) ──►──┘(positive)
        ├──CLIP──► CLIPTextEncode (negative) ──►──┘(negative)
        └──VAE───► VAEDecode ◄──────────────────┘(vae)
EmptyLatentImage ──LATENT──► KSampler (latent_image)
```

### Node-by-node construction

```python
# 1. Checkpoint loader
n1 = add_node("CheckpointLoaderSimple", "Load Checkpoint",
              ckpt_name="<exact filename from query_available_models>")

# 2. Positive prompt
n2 = add_node("CLIPTextEncode", "Positive Prompt",
              clip=[n1, 1], text="<detailed positive prompt>")

# 3. Negative prompt
n3 = add_node("CLIPTextEncode", "Negative Prompt",
              clip=[n1, 1], text="<negative prompt>")

# 4. Empty latent
n4 = add_node("EmptyLatentImage", "Empty Latent",
              width=512, height=512, batch_size=1)

# 5. KSampler
n5 = add_node("KSampler", "KSampler",
              model=[n1, 0], positive=[n2, 0], negative=[n3, 0],
              latent_image=[n4, 0],
              seed=42, steps=20, cfg=7.0,
              sampler_name="euler_ancestral", scheduler="normal",
              denoise=1.0)

# 6. VAE Decode
n6 = add_node("VAEDecode", "VAE Decode",
              samples=[n5, 0], vae=[n1, 2])

# 7. Save Image
n7 = add_node("SaveImage", "Save Image",
              images=[n6, 0], filename_prefix="ComfyClaw")
```

### Default parameters

| Parameter | Value | Notes |
|---|---|---|
| Resolution | 512×512 | SD 1.5 native; 768×768 max without quality loss |
| Steps | 20 | Good quality/speed balance |
| CFG | 7.0 | Standard; 5-9 range |
| Sampler | `euler_ancestral` | Good general choice |
| Scheduler | `normal` | Standard for SD 1.5 |

### LCM variant (DreamShaper-LCM, etc.)

If the checkpoint name contains `lcm`:
- steps=6, cfg=2.0, sampler_name="lcm", scheduler="sgm_uniform"
- **Read skill "dreamshaper8-lcm" for detailed guidance**

---

## Recipe B — SDXL (txt2img)

Stable Diffusion XL. Same topology as SD 1.5, but higher resolution.

### Node-by-node construction

```python
n1 = add_node("CheckpointLoaderSimple", "Load SDXL",
              ckpt_name="<sdxl checkpoint>")

n2 = add_node("CLIPTextEncode", "Positive Prompt",
              clip=[n1, 1], text="<detailed positive prompt>")

n3 = add_node("CLIPTextEncode", "Negative Prompt",
              clip=[n1, 1], text="<negative prompt>")

n4 = add_node("EmptyLatentImage", "Empty Latent",
              width=1024, height=1024, batch_size=1)

n5 = add_node("KSampler", "KSampler",
              model=[n1, 0], positive=[n2, 0], negative=[n3, 0],
              latent_image=[n4, 0],
              seed=42, steps=25, cfg=7.0,
              sampler_name="dpmpp_2m", scheduler="karras",
              denoise=1.0)

n6 = add_node("VAEDecode", "VAE Decode",
              samples=[n5, 0], vae=[n1, 2])

n7 = add_node("SaveImage", "Save Image",
              images=[n6, 0], filename_prefix="ComfyClaw")
```

### Default parameters

| Parameter | Value | Notes |
|---|---|---|
| Resolution | 1024×1024 | SDXL native; supports 768×1344, 1344×768, 896×1152, etc. |
| Steps | 25 | Slightly more than SD 1.5 for quality |
| CFG | 7.0 | Same as SD 1.5 |
| Sampler | `dpmpp_2m` | Best for SDXL |
| Scheduler | `karras` | Smoother noise schedule |

### SDXL resolution buckets

| Aspect | Width | Height |
|---|---|---|
| 1:1 | 1024 | 1024 |
| 16:9 | 1344 | 768 |
| 9:16 | 768 | 1344 |
| 4:3 | 1152 | 896 |
| 3:4 | 896 | 1152 |
| 21:9 | 1536 | 640 |

---

## Recipe C — Flux (txt2img)

Flux uses UNETLoader (separate diffusion model) with dual CLIP encoders.
The graph structure differs significantly from SD/SDXL.

**First check which Flux files exist:**
```
query_available_models("diffusion_models")  # for the UNET
query_available_models("clip")              # for text encoders
query_available_models("vae")               # for VAE
```

### Node-by-node construction

```python
# 1. UNET Loader
n1 = add_node("UNETLoader", "Load Flux UNET",
              unet_name="<flux model file>",
              weight_dtype="fp8_e4m3fn")

# 2. Dual CLIP Loader (Flux uses CLIP-L + T5-XXL)
n2 = add_node("DualCLIPLoader", "Load CLIP",
              clip_name1="<clip_l file>",
              clip_name2="<t5xxl file>",
              type="flux")

# 3. VAE Loader
n3 = add_node("VAELoader", "Load VAE",
              vae_name="<flux vae file>")

# 4. Positive prompt
n4 = add_node("CLIPTextEncode", "Positive Prompt",
              clip=[n2, 0], text="<detailed positive prompt>")

# 5. Empty latent (use EmptySD3LatentImage for Flux)
n5 = add_node("EmptySD3LatentImage", "Empty Latent",
              width=1024, height=1024, batch_size=1)

# 6. Flux guidance scale
n6 = add_node("FluxGuidance", "Flux Guidance",
              conditioning=[n4, 0], guidance=3.5)

# 7. KSampler
n7 = add_node("KSampler", "KSampler",
              model=[n1, 0], positive=[n6, 0],
              negative=[n4, 0],
              latent_image=[n5, 0],
              seed=42, steps=20, cfg=1.0,
              sampler_name="euler", scheduler="simple",
              denoise=1.0)

# 8. VAE Decode
n8 = add_node("VAEDecode", "VAE Decode",
              samples=[n7, 0], vae=[n3, 0])

# 9. Save Image
n9 = add_node("SaveImage", "Save Image",
              images=[n8, 0], filename_prefix="ComfyClaw")
```

### Default parameters

| Parameter | Value | Notes |
|---|---|---|
| Resolution | 1024×1024 | Flux supports 512–2048 range |
| Steps | 20 | 20-30 for quality |
| CFG | 1.0 | Flux uses FluxGuidance node instead of CFG |
| Guidance | 3.5 | Via FluxGuidance node; range 1.0–10.0 |
| Sampler | `euler` | Best for Flux |
| Scheduler | `simple` | Standard for Flux |
| weight_dtype | `fp8_e4m3fn` | For VRAM efficiency; use `default` on MPS |

### Important Flux differences
- Flux does NOT use a negative prompt in the traditional sense (set it to empty or same as positive)
- Guidance is controlled via the `FluxGuidance` node, NOT cfg on KSampler
- Uses `EmptySD3LatentImage` (not `EmptyLatentImage`)
- `DualCLIPLoader` with type="flux" (not regular CLIPLoader)

---

## Recipe D — Qwen-Image-2512

**Read skill "qwen-image-2512" for detailed guidance** — it covers this model's
unique architecture, Lightning LoRA mode, resolution buckets, and prompt style.

Quick construction reference (use EXACT filenames from query_available_models):

```python
# Loaders — each outputs ONE type on slot 0
n1 = add_node("UNETLoader", "Load Qwen UNET",
              unet_name="<from diffusion_models query>",  # → slot 0: MODEL
              weight_dtype="fp8_e4m3fn")
n2 = add_node("CLIPLoader", "Load CLIP",
              clip_name="<from clip query>",               # → slot 0: CLIP
              type="qwen_image", device="default")
n3 = add_node("VAELoader", "Load VAE",
              vae_name="<from vae query>")                 # → slot 0: VAE

# Optional Lightning LoRA for 4-step fast inference
n_lora = add_node("LoraLoaderModelOnly", "Lightning LoRA",
                  model=[n1, 0],                           # ← MODEL from UNETLoader
                  lora_name="<from loras query>",          # → slot 0: MODEL
                  strength_model=1.0)

n4 = add_node("ModelSamplingAuraFlow", "AuraFlow Sampling",
              model=[n_lora, 0],                           # ← MODEL from LoRA
              shift=3.1)                                   # → slot 0: MODEL

# Prompts — CLIPTextEncode outputs CONDITIONING on slot 0
n5 = add_node("CLIPTextEncode", "Positive Prompt",
              clip=[n2, 0],                                # ← CLIP from CLIPLoader
              text="<natural language prompt>")            # → slot 0: CONDITIONING
n6 = add_node("CLIPTextEncode", "Negative Prompt",
              clip=[n2, 0],                                # ← CLIP from CLIPLoader
              text="低分辨率，低画质，肢体畸形，画面具有AI感")  # → slot 0: CONDITIONING

# Latent
n7 = add_node("EmptySD3LatentImage", "Empty Latent",
              width=1328, height=1328, batch_size=1)       # → slot 0: LATENT

# Sampler — all inputs wired, output is LATENT on slot 0
n8 = add_node("KSampler", "KSampler",
              model=[n4, 0],                               # ← MODEL from AuraFlow
              positive=[n5, 0],                            # ← CONDITIONING from pos prompt
              negative=[n6, 0],                            # ← CONDITIONING from neg prompt
              latent_image=[n7, 0],                        # ← LATENT from empty latent
              seed=42, steps=4, cfg=1.0,
              sampler_name="euler", scheduler="simple",
              denoise=1.0)                                 # → slot 0: LATENT

# Decode — samples MUST come from KSampler slot 0, vae from VAELoader slot 0
n9 = add_node("VAEDecode", "VAE Decode",
              samples=[n8, 0],                             # ← LATENT from KSampler
              vae=[n3, 0])                                 # ← VAE from VAELoader
                                                           # → slot 0: IMAGE

# Save — images MUST come from VAEDecode slot 0
n10 = add_node("SaveImage", "Save Image",
               images=[n9, 0],                             # ← IMAGE from VAEDecode
               filename_prefix="ComfyClaw")
```

---

## Recipe G — Z-Image-Turbo

**Read skill "z-image-turbo" for detailed guidance** — it covers sampler settings,
LoRA stacking, and ControlNet usage for this model.

Z-Image-Turbo: 6B S3-DiT, BF16, 16 GB VRAM. Separate UNETLoader + CLIPLoader + VAELoader.
**No ModelSamplingAuraFlow needed. CFG must be 0.0. Steps = 8.**

```python
# 1. UNET loader
n1 = add_node("UNETLoader", "Z-Image UNET",
              unet_name="z_image_turbo_bf16.safetensors",
              weight_dtype="bfloat16")

# 2. Text encoder (Qwen3-4B)
n2 = add_node("CLIPLoader", "Z-Image CLIP",
              clip_name="qwen_3_4b.safetensors",
              type="qwen3_4b")

# 3. VAE
n3 = add_node("VAELoader", "VAE",
              vae_name="ae.safetensors")

# 4. Positive prompt
n4 = add_node("CLIPTextEncode", "Positive Prompt",
              clip=[n2, 0], text="<detailed natural-language description>")

# 5. Negative prompt
n5 = add_node("CLIPTextEncode", "Negative Prompt",
              clip=[n2, 0], text="low resolution, blurry, distorted, watermark")

# 6. Latent (1024×1024 native)
n6 = add_node("EmptySD3LatentImage", "Empty Latent",
              width=1024, height=1024, batch_size=1)

# 7. KSampler — cfg=0.0 is mandatory for Turbo
n7 = add_node("KSampler", "KSampler",
              model=[n1, 0],
              positive=[n4, 0], negative=[n5, 0],
              latent_image=[n6, 0],
              seed=42, steps=8, cfg=0.0,
              sampler_name="euler", scheduler="simple",
              denoise=1.0)

# 8. VAE Decode
n8 = add_node("VAEDecode", "VAE Decode",
              samples=[n7, 0], vae=[n3, 0])

# 9. Save
n9 = add_node("SaveImage", "Save Image",
              images=[n8, 0], filename_prefix="ComfyClaw")
```

### Default parameters

| Parameter | Value | Notes |
|---|---|---|
| Resolution | 1024×1024 | Native; 16:9 → 1280×720 |
| Steps | 8 | Turbo-optimised |
| CFG | **0.0** | Mandatory — guidance distillation |
| Sampler | `euler` | |
| Scheduler | `simple` | |

---

## Recipe H — LongCat-Image

**Read skill "longcat-image" for detailed guidance** — it covers Chinese text rendering,
parameter tuning, and iteration strategy.

LongCat-Image: 6B model by Meituan, BF16, ~17 GB VRAM. Uses custom pipeline nodes —
NOT UNETLoader/KSampler. **No LoRA or ControlNet support.** Key strength: Chinese text rendering.

```python
# 1. Load model — outputs LONGCAT_PIPE on slot 0
n1 = add_node("LongCatImageModelLoader", "LongCat-Image Loader",
              model_name="<exact filename from query_available_models>",
              precision="bfloat16")

# 2. All-in-one generator — takes LONGCAT_PIPE, outputs IMAGE on slot 0
n2 = add_node("LongCatImageTextToImage", "LongCat Generate",
              model=[n1, 0],
              positive_text="<detailed natural-language prompt>",
              negative_text="low quality, blurry, watermark, text artifacts",
              steps=28,
              guidance_scale=4.5,
              seed=42,
              width=1344,
              height=768,
              enable_cfg_renorm=True,
              enable_prompt_rewrite=True)

# 3. Save Image — takes IMAGE on slot 0
n3 = add_node("SaveImage", "Save Image",
              images=[n2, 0],
              filename_prefix="ComfyClaw")
```

### Default parameters

| Parameter | Photorealism | Stylized | Notes |
|---|---|---|---|
| `steps` | 24–32 | 14–24 | Start at 28 |
| `guidance_scale` | 4.5–6.0 | 3.0–4.0 | Start at 4.5 |
| `enable_cfg_renorm` | `true` | `true` | Always enable |
| `enable_prompt_rewrite` | `true` | `true` | Always enable |

### Resolution buckets

| Aspect | Width | Height |
|---|---|---|
| 16:9 | 1344 | 768 |
| 9:16 | 768 | 1344 |
| 1:1 | 1024 | 1024 |

### Chinese text rendering

Wrap any text to be rendered in quotation marks inside the prompt:
```
A neon sign reading "欢迎光临" above a Beijing street food stall at night.
```

---

## Recipe E — HunyuanDiT

HunyuanDiT is a bilingual (Chinese/English) DiT model.

```python
n1 = add_node("UNETLoader", "Load HunyuanDiT",
              unet_name="<hunyuan model file>",
              weight_dtype="fp8_e4m3fn")
n2 = add_node("DualCLIPLoader", "Load CLIP",
              clip_name1="<clip_l file>",
              clip_name2="<mt5_xl file>",
              type="hunyuan_dit")
n3 = add_node("VAELoader", "Load VAE",
              vae_name="<vae file>")

n4 = add_node("CLIPTextEncode", "Positive", clip=[n2, 0], text="<prompt>")
n5 = add_node("CLIPTextEncode", "Negative", clip=[n2, 0], text="<negative>")
n6 = add_node("EmptySD3LatentImage", "Empty Latent",
              width=1024, height=1024, batch_size=1)

n7 = add_node("KSampler", "KSampler",
              model=[n1, 0], positive=[n4, 0], negative=[n5, 0],
              latent_image=[n6, 0],
              seed=42, steps=30, cfg=6.0,
              sampler_name="euler", scheduler="normal", denoise=1.0)

n8 = add_node("VAEDecode", "VAE Decode", samples=[n7, 0], vae=[n3, 0])
n9 = add_node("SaveImage", "Save Image", images=[n8, 0], filename_prefix="ComfyClaw")
```

---

## Recipe F — SD 3.x

SD3 uses triple text encoders (CLIP-L, CLIP-G, T5-XXL).

```python
n1 = add_node("CheckpointLoaderSimple", "Load SD3",
              ckpt_name="<sd3 checkpoint>")

n2 = add_node("CLIPTextEncode", "Positive", clip=[n1, 1], text="<prompt>")
n3 = add_node("CLIPTextEncode", "Negative", clip=[n1, 1], text="<negative>")
n4 = add_node("EmptySD3LatentImage", "Empty Latent",
              width=1024, height=1024, batch_size=1)

n5 = add_node("KSampler", "KSampler",
              model=[n1, 0], positive=[n2, 0], negative=[n3, 0],
              latent_image=[n4, 0],
              seed=42, steps=28, cfg=4.5,
              sampler_name="euler", scheduler="sgm_uniform", denoise=1.0)

n6 = add_node("VAEDecode", "VAE Decode", samples=[n5, 0], vae=[n1, 2])
n7 = add_node("SaveImage", "Save Image", images=[n6, 0], filename_prefix="ComfyClaw")
```

| Parameter | Value |
|---|---|
| Resolution | 1024×1024 |
| Steps | 28 |
| CFG | 4.5 |
| Sampler | `euler` |
| Scheduler | `sgm_uniform` |

---

## Common Enhancements (apply after base pipeline)

After building the base pipeline, enhance with these patterns:

### Add LoRA
Call `add_lora_loader` composite tool — it auto-wires model+clip chains.

### Add ControlNet
Call `add_controlnet` composite tool — it adds loader + preprocessor + apply node.

### Add Hires Fix (upscale pass)
Call `add_hires_fix` composite tool — adds LatentUpscaleBy + second KSampler + VAEDecode.

### Add Regional Prompting
Call `add_regional_attention` composite tool — splits positive conditioning
into foreground/background weighted regions.

---

## Critical Rules

1. **Always query_available_models FIRST** — never guess filenames.
2. **Use exact filenames** from the query results — a single typo causes HTTP 400.
3. **Add nodes ONE AT A TIME** — each `add_node` returns the node ID for wiring.
4. **Connect via `add_node` inputs** — pass `input_name=[source_node_id, output_slot]`.
5. **Every workflow needs SaveImage** — without it, ComfyUI returns `prompt_no_outputs`.
6. **Check UNETLoader weight_dtype** — use `"default"` on Apple MPS, `"fp8_e4m3fn"` on CUDA.
7. **Different latent types** — SD 1.5/SDXL use `EmptyLatentImage`; Flux/SD3/Qwen/HunyuanDiT use `EmptySD3LatentImage`.
