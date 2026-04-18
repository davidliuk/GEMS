---
name: longcat-image
description: >-
  Configuration guide for LongCat-Image, Meituan's 6B text-to-image model (Dec 2025).
  Detect when UNETLoader contains "longcat_image" in the model name, or CLIPLoader
  type is "longcat_image". Uses standard ComfyUI nodes (UNETLoader, CLIPLoader
  type "longcat_image", VAELoader, KSampler) with FluxGuidance and CFGNorm.
  Key strength: superior Chinese text rendering. LoRA and ControlNet are not supported.
license: Apache-2.0
metadata:
  author: davidliuk
  version: "2.0.0"
  base_arch: LongCat-Image (6B DiT, BF16)
  diffusion_model: longcat_image_bf16.safetensors       → ComfyUI/models/diffusion_models/
  text_encoder:    qwen_2.5_vl_7b_fp8_scaled.safetensors → ComfyUI/models/text_encoders/
  vae:             ae.safetensors                         → ComfyUI/models/vae/
  variants: LongCat-Image, LongCat-Image-Dev, LongCat-Image-Edit, LongCat-Image-Edit-Turbo
tags: [agent, "model:longcat"]
---

LongCat-Image is Meituan's 6B text-to-image model released December 2025. Its defining
strength is precise Chinese text rendering — when you place Chinese (or Latin) text inside
quotation marks in the prompt, the model renders that text accurately inside the image.
It requires ~17 GB VRAM (BF16) or CPU offload.

---

## ⚠️ Architecture differences vs SD/SDXL and other models

| Feature | SD 1.5 / SDXL | Qwen-Image-2512 | Z-Image-Turbo | LongCat-Image |
|---|---|---|---|---|
| Model loader | `CheckpointLoaderSimple` | `UNETLoader` | `UNETLoader` | **`UNETLoader`** |
| Text encoder type | built-in | `qwen_image` | `lumina2` | **`longcat_image`** |
| Text encoder file | built-in | qwen_2.5_vl_7b | qwen_3_4b | **qwen_2.5_vl_7b_fp8_scaled** |
| VAE | built-in | qwen_image_vae | ae | **ae** |
| Model conditioning | — | `ModelSamplingAuraFlow` | `ModelSamplingAuraFlow` | **`CFGNorm` (strength=1)** |
| Guidance node | — | — | — | **`FluxGuidance` (guidance=4)** |
| Steps | 20–30 | 50 | 8 | **20** |
| CFG | 7.0 | 4.0 | 1 | **4** |
| Sampler | `euler_ancestral` | `euler` | `res_multistep` | **`euler`** |
| Resolution | 512/1024 | 1328×1328 | 1024×1024 | **1024×1024** |
| LoRA | yes | yes | yes | **not supported** |
| ControlNet | yes | yes | yes | **not supported** |

> **Note on LoRA / ControlNet tools:** Calling `add_lora_loader` or `add_controlnet`
> while LongCat-Image is the active model returns an informative error and does NOT
> modify the workflow. Use `set_param` and prompt engineering instead — see §6/§7.

---

## 1. Node graph structure

### Standard text-to-image

```
UNETLoader ("longcat_image_bf16.safetensors", weight_dtype="default")
    └──► CFGNorm (strength=1)
             └──► KSampler
                      ◄── FluxGuidance (guidance=4) ◄── CLIPTextEncode (positive) ← CLIPLoader (type="longcat_image")
                      ◄── FluxGuidance (guidance=4) ◄── CLIPTextEncode (negative) ← CLIPLoader (type="longcat_image")
                      ◄── EmptySD3LatentImage (1024×1024)
                 └──► VAEDecode ◄── VAELoader ("ae.safetensors")
                          └──► SaveImage
```

Key points:
- **`CFGNorm` (strength=1)** is required between UNETLoader and KSampler
- **`FluxGuidance` (guidance=4)** is applied to BOTH positive and negative conditioning
- CLIPLoader type must be `"longcat_image"`
- Both positive and negative prompts use `CLIPTextEncode` (unlike Z-Image-Turbo)
- VAE is `ae.safetensors` (same as Z-Image-Turbo)

---

## 2. KSampler settings

| param | value | notes |
|---|---|---|
| steps | **20** | official default; try 50 for max quality |
| cfg | **4** | |
| sampler_name | **`euler`** | |
| scheduler | `simple` | |
| denoise | 1.0 | |

```
set_param("<KSampler ID>", "steps", 20)
set_param("<KSampler ID>", "cfg",   4)
set_param("<KSampler ID>", "sampler_name", "euler")
set_param("<KSampler ID>", "scheduler",    "simple")
```

---

## 3. Resolutions (EmptySD3LatentImage)

| Aspect | Width | Height | Best use case |
|---|---|---|---|
| 1:1 (default) | 1024 | 1024 | Social media, square |
| 16:9 | 1344 | 768 | Landscape, cinematic |
| 9:16 | 768 | 1344 | Portrait, mobile |
| 4:3 | 1152 | 864 | Standard print |
| 3:4 | 864 | 1152 | Portrait print |

---

## 4. Prompt engineering

LongCat-Image uses a powerful bilingual text encoder (Qwen2.5-VL-7B). Write
**descriptive natural language** in English or Chinese — not comma-separated tags.

### Standard prompt structure

```
[Subject and action], [scene/environment], [visual style], [lighting], [camera details].
```

Example:
```
A young woman reading a book in a sunlit library. Warm golden afternoon light,
towering bookshelves, shallow depth of field, cinematic composition, 35mm lens,
photorealistic, high resolution.
```

### Chinese text rendering (key strength)

To render text **inside the image**, wrap the text in quotation marks within the prompt:

```
# Single Chinese label
A street sign showing "欢迎光临" in bright red characters.

# Multiple text elements
A storefront with "咖啡馆" on a neon sign and "OPEN" on the door.
```

**Rules for text rendering:**
- Wrap target text in double quotes (`"text"`) inside the prompt
- Keep the surrounding description clear about the text's physical location
- Limit to 2–3 text elements per image for best results

### Negative prompt

```
blurry, low resolution, oversaturated, harsh lighting, messy composition,
distorted face, extra fingers, bad anatomy, cheap jewelry, plastic texture,
cartoon, illustration, anime, watermark, text, logo
```

---

## 5. Iteration strategy (no LoRA / ControlNet available)

Since LoRA and ControlNet are not supported, quality improvements come from:

| Verifier issue | Fix strategy |
|---|---|
| Dull / low contrast | Increase FluxGuidance guidance to 5.0–6.0 |
| Wrong composition | Add explicit spatial language ("in the foreground", "upper-left") |
| Too stylized / painterly | Increase steps to 50; add "photorealistic, 4k, sharp focus" |
| Blurry / soft | Increase steps; add "sharp focus, ultra-detailed" to prompt |
| Wrong lighting | Add specific lighting terms ("golden hour", "dramatic rim light") |
| Text rendering errors | Move text to a clearer part of the scene; fewer text elements |
| Complete failure | Check CFGNorm and FluxGuidance nodes are present |

**NEVER remove CFGNorm or FluxGuidance nodes** — they are essential for correct generation.

---

## 6. Example: standard photorealistic generation

```
inspect_workflow()

set_param("<positive CLIPTextEncode ID>", "text",
    "A modern coffee shop interior at golden hour. "
    "Warm amber light through large windows, wooden furniture, "
    "latte art on the counter, cozy atmosphere, "
    "photorealistic, shallow depth of field, 35mm lens, 4k.")
set_param("<negative CLIPTextEncode ID>", "text",
    "low quality, blurry, watermark, oversaturated, flat lighting")

# Keep default steps=20, cfg=4
finalize_workflow()
```

## 7. Example: Chinese text rendering

```
inspect_workflow()

set_param("<positive CLIPTextEncode ID>", "text",
    'A vintage Chinese tea shop with a wooden sign reading "茶馆" '
    'above the door, a handwritten menu board showing "龙井 ¥38" '
    'inside, warm lantern lighting, photorealistic, high detail.')
set_param("<KSampler ID>", "steps", 50)

finalize_workflow()
```