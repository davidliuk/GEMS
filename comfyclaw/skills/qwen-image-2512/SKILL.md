---
name: qwen-image-2512
description: >-
  Configuration guide for Qwen-Image-2512, Alibaba's top-ranked open-source
  text-to-image model (Dec 2025). Detect this model when the workflow contains
  UNETLoader with "qwen_image" in the model name, or CLIPLoader with
  type "qwen_image". Uses standard ComfyUI nodes (UNETLoader, CLIPLoader,
  VAELoader, KSampler, EmptySD3LatentImage) with FP8 quantized weights.
license: Apache-2.0
metadata:
  author: davidliuk
  version: "3.0.0"
  base_arch: Qwen-Image-2512 (20B MMDiT + Qwen2.5-VL-7B text encoder, FP8 quantized)
  diffusion_model: qwen_image_2512_fp8_e4m3fn.safetensors  → ComfyUI/models/diffusion_models/
  text_encoder:    qwen_2.5_vl_7b_fp8_scaled.safetensors   → ComfyUI/models/text_encoders/
  vae:             qwen_image_vae.safetensors                → ComfyUI/models/vae/
  optional_lora:   Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors → ComfyUI/models/loras/
  optional_controlnet: qwen_image_canny_diffsynth_controlnet.safetensors, qwen_image_depth_diffsynth_controlnet.safetensors → ComfyUI/models/model_patches/
tags: [agent, "model:qwen"]
---

Qwen-Image-2512 is Alibaba's #1-ranked open-source T2I model on AI Arena (Dec 2025).
20B MMDiT architecture with Qwen2.5-VL-7B as text encoder.
FP8 quantized weights (~28 GB total, fits in 45 GB VRAM).

---

## ⚠️ Architecture differences vs SD/SDXL

| Feature | SD 1.5 / SDXL | Qwen-Image-2512 |
|---|---|---|
| Model loader | `CheckpointLoaderSimple` | `UNETLoader` + `CLIPLoader` + `VAELoader` |
| Latent space | `EmptyLatentImage` | `EmptySD3LatentImage` |
| Model conditioning | — | `ModelSamplingAuraFlow` (required, `shift`≈3.1) |
| LoRA node | `LoraLoader` (MODEL+CLIP) | `LoraLoaderModelOnly` (MODEL only) |
| ControlNet loader | `ControlNetLoader` | `ModelPatchLoader` (loads the CN patch file) |
| ControlNet apply | `ControlNetApplyAdvanced` (patches conditioning) | `QwenImageDiffsynthControlnet` (patches the MODEL tensor) |
| Typical steps | 20–30 | **4** (speed LoRA) / **8** (turbo) / **50** (standard) |
| Typical CFG | 7.0 | **1.0** (speed LoRA) / **4.0** (standard) |
| Sampler | `euler_ancestral` | `euler` |
| Scheduler | `karras` | `simple` |
| Native resolution | 512 or 1024 | **1328 × 1328** (see aspect ratios below) |

---

## 1. Node graph structure

### Standard (no LoRA)

```
UNETLoader ("qwen_image_2512_fp8_e4m3fn.safetensors")
    └──► ModelSamplingAuraFlow (shift=3.1)
             └──► KSampler ◄── CLIPTextEncode (positive) ← CLIPLoader (type="qwen_image")
                           ◄── CLIPTextEncode (negative) ← CLIPLoader (type="qwen_image")
                           ◄── EmptySD3LatentImage (1328×1328)
                      └──► VAEDecode ◄── VAELoader
                               └──► SaveImage
```

### With LoRA

```
UNETLoader
    └──► LoraLoaderModelOnly (speed LoRA, strength=1.0)
             └──► LoraLoaderModelOnly (style LoRA, strength=0.7)  ← optional second LoRA
                      └──► ModelSamplingAuraFlow (shift=3.1)
                               └──► KSampler ...
```

### With DiffSynth ControlNet (model-patch style)

Qwen-Image-2512 ControlNet does **not** wrap positive/negative conditioning.
Instead it patches the MODEL tensor itself with a reference image:

```
UNETLoader ──► [LoRA chain] ──► QwenImageDiffsynthControlnet ──► ModelSamplingAuraFlow ──► KSampler
                                       ▲    ▲    ▲
                                       │    │    └── LoadImage (control reference, e.g. canny / depth map)
                                       │    └────── VAELoader
                                       └──────────  ModelPatchLoader (qwen_image_*_diffsynth_controlnet.safetensors)

CLIPTextEncode (positive) ──► KSampler.positive   (unchanged — no conditioning wrap)
CLIPTextEncode (negative) ──► KSampler.negative   (unchanged)
```

Because the MODEL is patched, `add_controlnet` **requires an `image_node_id`**
(LoadImage / VAEDecode) and the workflow **must contain a VAELoader**.

---

## 2. KSampler settings

### Mode A — Speed LoRA (4 steps, Lightning or Turbo)

| param | value |
|---|---|
| steps | 4 |
| cfg | 1.0 |
| sampler_name | `euler` |
| scheduler | `simple` |
| denoise | 1.0 |

### Mode B — Speed LoRA (8 steps, higher quality)

| param | value |
|---|---|
| steps | 8 |
| cfg | 1.0 |
| sampler_name | `euler` |
| scheduler | `simple` |
| denoise | 1.0 |

### Mode C — Standard (no speed LoRA, 50 steps)

| param | value |
|---|---|
| steps | 50 |
| cfg | 4.0 |
| sampler_name | `euler` |
| scheduler | `simple` |
| denoise | 1.0 |

---

## 3. ModelSamplingAuraFlow shift

Required node — removing it causes pure noise output.

`shift` controls the noise schedule (range 1.0–5.0, default 3.1):
- Higher (4.0–5.0) → more structured composition, stronger global layout
- Lower (1.5–2.5) → finer details, more creative freedom
- Default 3.1 is well-calibrated for most prompts

```
set_param("<node_id>", "shift", 3.1)
```

---

## 4. Resolutions (EmptySD3LatentImage)

Qwen-Image-2512 trains on fixed aspect-ratio buckets. Use only these:

| aspect | width | height | use case |
|---|---|---|---|
| 1:1 | 1328 | 1328 | portraits, square |
| 16:9 | 1664 | 928 | landscape, cinematic |
| 9:16 | 928 | 1664 | portrait/mobile |
| 4:3 | 1472 | 1104 | standard landscape |
| 3:4 | 1104 | 1472 | standard portrait |
| 3:2 | 1584 | 1056 | wide landscape |
| 2:3 | 1056 | 1584 | tall portrait |

---

## 5. Prompt engineering

Use **natural language sentences**, not comma-separated keywords. Qwen2.5-VL
understands descriptive prose far better than SD-style tag lists.

```
Good:
"A young woman standing in a sunlit wheat field at golden hour.
 Loose cream linen dress, long auburn hair catching the breeze.
 Shallow depth of field, soft bokeh background, film-grain texture.
 Warm tones, photorealistic, high detail."

Poor:  "woman, wheat field, golden hour, 8k, masterpiece, highly detailed"
```

Do NOT use `masterpiece`, `8k uhd`, `best quality` — these tokens have no
positive effect on Qwen and may reduce coherence.

**Negative prompt (Chinese preferred — more effective):**
```
低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感，构图混乱，文字模糊，扭曲
```

English fallback:
```
low resolution, low quality, deformed limbs, distorted fingers, oversaturated,
waxy skin, no facial detail, over-smoothed, AI-generated look, chaotic composition,
blurry text, distorted text
```

---

## 6. LoRA — complete guide

### Architecture note

Qwen-Image-2512 is a **MMDiT** — text and image are co-processed in the same
transformer. There is **no separate CLIP tower**. Every LoRA for this model is
**model-only**: only `strength_model` matters; `strength_clip` is ignored.

Always use `LoraLoaderModelOnly`, never `LoraLoader`.
The harness handles this automatically when it detects a Qwen pipeline.

---

### 6.1 Speed / acceleration LoRAs

Use these to reduce step count from 50 → 4–8 without quality loss.
Apply at **strength_model = 1.0** — do not lower it.

| LoRA | Steps | CFG | Filename | Notes |
|---|---|---|---|---|
| **Lightning 4-step** (lightx2v) | 4 | 1.0 | `Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors` (recommended, ~811 MB) / `…-fp32.safetensors` (~1.6 GB reference weight) | Official; BF16 is the default and matches Qwen's BF16 base; it is what the bundled E2E test uses |
| **Lightning 8-step** | 8 | 1.0 | `Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors` | More detail than 4-step |
| **Lightning 8-step V2** | 8 | 1.0 | `Qwen-Image-2512-Lightning-8steps-V2.0.safetensors` | Improved version |
| **Turbo 4-step V3** (Wuli-art) | 4 | 1.0 | `Wuli-Qwen-Image-2512-Turbo-LoRA-4steps-V3.0-bf16.safetensors` | Stronger contrast, cinematic; latest |
| **Turbo 2-step** (Wuli-art) | 2 | 1.0 | `Wuli-Qwen-Image-2512-Turbo-LoRA-2steps-V1.0-bf16.safetensors` | Extreme speed; text may degrade |

**Lightning vs Turbo:**
- Lightning (lightx2v): official, works cleanly with style LoRA stacking
- Turbo (Wuli-art): stronger light/shadow contrast, more cinematic look;
  use V3.0 for latest improvements

**Sampler config for any speed LoRA:**
```
steps=4, cfg=1.0, sampler_name="euler", scheduler="simple"
```

---

### 6.2 Style / creative LoRAs

#### Anime & illustration

| LoRA | Trigger word | strength_model | Notes |
|---|---|---|---|
| Qwen 2512 Expressive Anime | `anime style` | 1.0 | Versatile dynamic anime, expressive faces |
| Qwen 2512 Illustria Vivid | `anime style` | 1.0 | Vivid illustration colours |
| Qwen2511+Z Illustria Anime 01 | — | 0.8–1.0 | 9:16 aspect recommended |
| Qwen 2512 Hyperreal Anime | — | 0.8–1.0 | Photorealistic + anime blend |
| Anime Flat Style QWEN 2512 | `flat style, minimal shading` | 0.9–1.0 | Bold lines, flat colour |
| Anime Koni R15 | Chinese char. prompts | 0.7–1.0 | Trained on Chinese captions |
| Anime IRL (flymy-ai) | `Real life Anime` | 0.8–1.0 | Anime → photorealistic bridge |

#### Photorealism & portrait

| LoRA | Trigger word | strength_model | Notes |
|---|---|---|---|
| Qwen-Image Realism (flymy-ai) | `realism` | 0.8–1.0 | Improved skin, facial detail; cfg=5.0 |
| NiceGirls UltraReal | — | 0.7–1.0 | European feminine portraiture |
| NSGIRL UltraRealistic | — | 0.8–1.0 | 10K-step training, BF16 |

#### Specialty / niche

| LoRA | Trigger word | strength_model | Notes |
|---|---|---|---|
| Pixel Art (prithivMLmods) | `Pixel Art` | 1.0 | Best at 1280×832; network dim 64 |
| SCI_FI_CORE | — | 0.8 | **Avoid FP8 — use BF16/GGUF** |
| FusionLoRA ByRemile | `hand-painted, thick brushstrokes` | 0.85–1.0 | cfg=1.0–1.3 |
| Qwen 360 Diffusion | `equirectangular, 360 image` | 1.0 | Panoramic output; network dim 128 |
| Advertisement (RayyanAhmed9477) | — | 1.0 | **Requires BF16 — DoRA tech, FP8 breaks it** |
| Eva Qwen (character) | `Eva_qwen` | 1.0 (0.86 complex) | Varied poses/body types |

#### FP8 compatibility warning

Some LoRAs (SCI_FI_CORE, Advertisement) use DoRA or other techniques that
degrade under FP8 quantisation. If you see visual artifacts with these LoRAs,
switch the base model to BF16 precision (`weight_dtype = "bf16"` on UNETLoader).

---

### 6.3 Adding a LoRA with `add_lora_loader`

**Call pattern:**
```
add_lora_loader(
    lora_name      = "<exact filename from query_available_models>",
    model_node_id  = "<UNETLoader node ID, or last LoraLoaderModelOnly if stacking>",
    strength_model = 0.8,
    # clip_node_id omitted — Qwen has no CLIP tower
)
```

The harness auto-detects Qwen and uses `LoraLoaderModelOnly`.

**Insertion position:** between the last model source and `ModelSamplingAuraFlow`.

---

### 6.4 Stacking LoRAs (speed + style)

Stack a speed LoRA first, then a style LoRA:

```
# Step 1: add speed LoRA (full strength)
add_lora_loader(
    lora_name      = "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
    model_node_id  = "<UNETLoader node ID>",
    strength_model = 1.0,
)
# → returns LoraLoaderModelOnly node ID, e.g. "101"

# Step 2: add style LoRA on top
add_lora_loader(
    lora_name      = "Wuli-Qwen-Image-2512-Turbo-LoRA-4steps-V3.0-bf16.safetensors",
    model_node_id  = "101",   # ← output of previous LoRA
    strength_model = 0.75,
)
```

**When stacking:**
- Speed LoRA: always 1.0
- Style LoRA: 0.5–0.85 (start at 0.75, reduce if style overwhelms)
- Limit to 2–3 LoRAs total; more causes competing modifications

**Sampler settings when stacking speed + style:**
```
steps=4 or 8, cfg=1.0, sampler_name="euler", scheduler="simple"
```

---

### 6.5 Trigger token placement

When a LoRA requires trigger tokens, prepend them to the **positive prompt**:
```
set_param("<CLIPTextEncode positive node ID>", "text",
    "anime style, <rest of your prompt here>")
```

---

## 7. ControlNet — complete guide

### Architecture note

Qwen-Image-2512 uses the **DiffSynth block-wise ControlNet** format, wired as a
**model-patch** rather than a conditioning-patch. This is **not compatible** with
standard SD/SDXL ControlNets or with Fun-style ControlNets for video models.

**ComfyUI nodes used** (both are built-in — no custom_nodes install needed):
- Loader: `ModelPatchLoader` (generic — loads any patch file from
  `ComfyUI/models/model_patches/`)
- Apply: `QwenImageDiffsynthControlnet`
  - Inputs: `model`, `model_patch`, `vae`, `image`, `strength`
  - Output: a patched `MODEL`
  - The output MODEL replaces the original MODEL at every downstream consumer
    (KSampler, ModelSamplingAuraFlow, etc.).

The harness auto-detects Qwen and uses these nodes when you call
`add_controlnet`. You do **not** pass `union_type` — DiffSynth ControlNets bake
the control mode into the file itself (one file per mode).

---

### 7.1 Available model files (place in `ComfyUI/models/model_patches/`)

| File | Size | Control mode | Source |
|---|---|---|---|
| `qwen_image_canny_diffsynth_controlnet.safetensors` | 2.2 GB | Canny edges | `Comfy-Org/Qwen-Image_ComfyUI` |
| `qwen_image_depth_diffsynth_controlnet.safetensors` | 2.2 GB | Depth map | `Comfy-Org/Qwen-Image_ComfyUI` |

Older `Qwen-Image-2512-Fun-Controlnet-Union-*.safetensors` files from
`alibaba-pai` are for video pipelines and are **not** loaded by
`ModelPatchLoader`; don't put them in `model_patches/`.

---

### 7.2 Control modes

The mode is fixed by the chosen file — there is no `union_type` switch.

| File | Control signal | Preprocessor class | Use case |
|---|---|---|---|
| `qwen_image_canny_diffsynth_controlnet.safetensors` | Edge map | `CannyEdgePreprocessor` (or `""` to use raw image) | Preserve structure/outlines |
| `qwen_image_depth_diffsynth_controlnet.safetensors` | Depth map | `MiDaS-DepthMapPreprocessor` / `DepthAnythingV2Preprocessor` (or `""`) | Fix 3D layout, foreground/bg |

**Preprocessor nodes** are provided by the `comfyui_controlnet_aux` custom
package (Fannovel16). If it isn't installed, pass `preprocessor_class=""` and
supply an already-processed image (e.g. a canny map PNG) to `image_node_id`.

---

### 7.3 Parameters

| Parameter | Node input name | Range | Default | Notes |
|---|---|---|---|---|
| Control strength | `strength` | 0.0–1.0 (typ.) | 0.8 | Direct multiplier on the patch contribution |
| `union_type` | — | — | — | **Not used** for DiffSynth CNs |
| `start_percent` / `end_percent` | — | — | — | Not applied by the patch node; schedule is baked in |

**Strength guidelines by use case:**

| Scenario | strength |
|---|---|
| Loose composition hint | 0.40–0.60 |
| Balanced (default) | 0.70–0.85 |
| Strict structure enforcement | 0.90–1.0 |

`start_percent` / `end_percent` are accepted by `add_controlnet` for API
symmetry but are **not forwarded** to `QwenImageDiffsynthControlnet` — it has no
schedule inputs.

---

### 7.4 Adding ControlNet with `add_controlnet`

```
# Step 1: check what patch files are installed
query_available_models("controlnets")     # surfaces both controlnets/ and model_patches/

# Step 2: add the DiffSynth ControlNet branch
add_controlnet(
    controlnet_name    = "qwen_image_canny_diffsynth_controlnet.safetensors",
    preprocessor_class = "CannyEdgePreprocessor",   # or "" to pass a pre-computed canny map
    image_node_id      = "<LoadImage or VAEDecode node ID>",   # REQUIRED
    positive_node_id   = "<CLIPTextEncode positive node ID>",  # required for API symmetry
    negative_node_id   = "<CLIPTextEncode negative node ID>",  # required for API symmetry
    strength           = 0.8,
)
```

The harness:
1. Adds `ModelPatchLoader(name=<controlnet_name>)` — loads the patch file
2. Adds the preprocessor node (if `preprocessor_class` is set)
3. Adds `QwenImageDiffsynthControlnet` with `(model, model_patch, vae, image, strength)`
4. **Rewires every downstream consumer of the source MODEL** (KSampler,
   ModelSamplingAuraFlow, …) to consume the patched MODEL instead
5. Leaves positive/negative CLIP conditioning untouched

> Requires a `VAELoader` already present in the workflow (the patch node
> consumes VAE). If none exists, the agent will error with a clear message —
> add one first.

---

### 7.5 Mode-specific examples

#### Canny (edge structure)
```
add_controlnet(
    controlnet_name    = "qwen_image_canny_diffsynth_controlnet.safetensors",
    preprocessor_class = "CannyEdgePreprocessor",
    image_node_id      = "<source image node>",
    positive_node_id   = "<pos node>",
    negative_node_id   = "<neg node>",
    strength           = 0.80,
)
```

#### Depth (3D layout)
```
add_controlnet(
    controlnet_name    = "qwen_image_depth_diffsynth_controlnet.safetensors",
    preprocessor_class = "MiDaS-DepthMapPreprocessor",
    image_node_id      = "<source image node>",
    positive_node_id   = "<pos node>",
    negative_node_id   = "<neg node>",
    strength           = 0.85,
)
```

#### Pre-processed control map (no preprocessor node installed)
```
add_controlnet(
    controlnet_name    = "qwen_image_canny_diffsynth_controlnet.safetensors",
    preprocessor_class = "",                       # pass the image unchanged
    image_node_id      = "<LoadImage node with pre-baked canny map>",
    positive_node_id   = "<pos node>",
    negative_node_id   = "<neg node>",
    strength           = 0.80,
)
```

> Qwen-Image DiffSynth ControlNet only ships canny and depth weights today.
> Pose / seg / normal are **not** available for Qwen-Image-2512 — prefer a
> LoRA-based fix for pose defects.

---

### 7.6 ControlNet + LoRA together

LoRA and ControlNet are independent patches on the MODEL branch — LoRA runs
first in the chain, then the DiffSynth patch:

```
# 1. Add speed LoRA (model side)
add_lora_loader(
    lora_name      = "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
    model_node_id  = "<UNETLoader ID>",
    strength_model = 1.0,
)

# 2. Add ControlNet (also on the MODEL branch — harness finds the current MODEL source)
add_controlnet(
    controlnet_name    = "qwen_image_depth_diffsynth_controlnet.safetensors",
    preprocessor_class = "MiDaS-DepthMapPreprocessor",
    image_node_id      = "<source image>",
    positive_node_id   = "<pos node>",
    negative_node_id   = "<neg node>",
    strength           = 0.80,
)

# 3. Set Lightning sampler settings
set_param("<KSampler ID>", "steps", 4)
set_param("<KSampler ID>", "cfg",   1.0)
```

---

## 8. Iteration strategy

| Verifier issue | Fix strategy |
|---|---|
| Dull / washed out colours | Raise cfg slightly (1.5–2.0); add detail in prompt |
| Oversaturated / garish | Lower cfg toward 1.0 |
| Soft / blurry output | Switch to Mode C (50 steps, cfg=4.0) without speed LoRA |
| Wrong global composition | Add spatial language: "in the foreground", "far right", "upper left" |
| AI-looking / waxy skin | Extend negative prompt with skin artefact terms |
| Wrong aspect ratio | Update `EmptySD3LatentImage` width/height to a supported bucket |
| Speed LoRA artefacts | Increase steps (4→8); switch from Lightning to Turbo V3 |
| Style LoRA too strong | Reduce `strength_model` to 0.5–0.6 |
| Style LoRA too weak | Raise to 0.85–1.0; add trigger word to positive prompt |
| FP8 LoRA artefacts | LoRA incompatible with FP8; switch UNETLoader weight_dtype to "bf16" |
| ControlNet too rigid | Lower `strength` to ~0.5–0.6 |
| ControlNet too weak | Raise `strength` to 0.9–1.0 |
| ControlNet wrong mode | Swap to the matching patch file (canny ↔ depth); no `union_type` on Qwen |
| ControlNet node error "needs a VAELoader" | Add a `VAELoader` in the workflow and connect it upstream of VAEDecode |

---

## 9. Examples

### Example A — Fast generation, Lightning LoRA only

```
inspect_workflow()

# Prompt
set_param("<pos node>", "text",
    "A wolf standing on a snowy mountain ridge at dusk. Dramatic orange and
     purple sky, blizzard beginning in the distance. Photorealistic, cinematic
     lighting, individual fur strands visible, shallow depth of field.")
set_param("<neg node>", "text",
    "低分辨率，低画质，肢体畸形，画面过饱和，画面具有AI感")

# Resolution (landscape)
set_param("<EmptySD3LatentImage>", "width",  1664)
set_param("<EmptySD3LatentImage>", "height",  928)

# Add Lightning LoRA
add_lora_loader(
    lora_name      = "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
    model_node_id  = "<UNETLoader ID>",
    strength_model = 1.0,
)

# 4-step Lightning settings
set_param("<KSampler>", "steps", 4)
set_param("<KSampler>", "cfg",   1.0)

finalize_workflow()
```

### Example B — Style LoRA stacking (speed + anime)

```
# Add Lightning speed LoRA first
add_lora_loader(
    lora_name      = "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
    model_node_id  = "<UNETLoader ID>",
    strength_model = 1.0,
)  # → returns node ID "101"

# Add anime style LoRA on top
add_lora_loader(
    lora_name      = "qwen_2512_expressive_anime.safetensors",
    model_node_id  = "101",
    strength_model = 0.75,
)

# Trigger word in positive prompt
set_param("<pos node>", "text",
    "anime style, a girl with silver hair standing in a moonlit garden.
     Cherry blossom petals falling, soft glow from paper lanterns.
     Flowing white kimono, detailed fabric, expressive eyes.")

set_param("<KSampler>", "steps", 4)
set_param("<KSampler>", "cfg",   1.0)

finalize_workflow()
```

### Example C — ControlNet depth + Lightning LoRA

```
# Add LoRA
add_lora_loader(
    lora_name      = "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
    model_node_id  = "<UNETLoader ID>",
    strength_model = 1.0,
)

# Add DiffSynth ControlNet (depth mode)
add_controlnet(
    controlnet_name    = "qwen_image_depth_diffsynth_controlnet.safetensors",
    preprocessor_class = "MiDaS-DepthMapPreprocessor",
    image_node_id      = "<LoadImage with reference photo>",
    positive_node_id   = "<pos CLIPTextEncode ID>",
    negative_node_id   = "<neg CLIPTextEncode ID>",
    strength           = 0.82,
)

set_param("<KSampler>", "steps", 4)
set_param("<KSampler>", "cfg",   1.0)

finalize_workflow()
```
