---
name: z-image-turbo
description: >-
  Configuration guide for Z-Image-Turbo, Alibaba Tongyi Lab's 6B S3-DiT
  text-to-image model (Nov 2025). Detect when UNETLoader contains "z_image"
  in the model name. Fits in 16 GB VRAM. Uses standard ComfyUI nodes
  (UNETLoader, CLIPLoader type "lumina2", VAELoader, ModelSamplingAuraFlow,
  KSampler with cfg=1 and sampler "res_multistep", EmptySD3LatentImage).
  Negative conditioning uses ConditioningZeroOut (NOT CLIPTextEncode).
license: Apache-2.0
metadata:
  author: davidliuk
  version: "2.0.0"
  base_arch: Z-Image-Turbo (6B S3-DiT, BF16)
  diffusion_model: z_image_turbo_bf16.safetensors     → ComfyUI/models/diffusion_models/
  text_encoder:    qwen_3_4b.safetensors               → ComfyUI/models/text_encoders/
  vae:             ae.safetensors                       → ComfyUI/models/vae/
---

Z-Image-Turbo is Alibaba Tongyi Lab's 6B S3-DiT (Single-Stream DiT) model released
November 2025. Unlike Qwen-Image-2512 (MMDiT), it concatenates text, visual semantic,
and image tokens into a single unified stream. Achieves sub-second inference on H800;
fits in 16 GB consumer VRAM in BF16.

---

## ⚠️ Architecture differences vs SD/SDXL and Qwen-Image-2512

| Feature | SD 1.5 / SDXL | Qwen-Image-2512 | Z-Image-Turbo |
|---|---|---|---|
| Architecture | UNet | 20B MMDiT | **6B S3-DiT** |
| Model loader | `CheckpointLoaderSimple` | `UNETLoader` | `UNETLoader` |
| Text encoder type | built-in | `qwen_image` | **`lumina2`** |
| Latent node | `EmptyLatentImage` | `EmptySD3LatentImage` | `EmptySD3LatentImage` |
| Mandatory cond. node | — | `ModelSamplingAuraFlow` | **`ModelSamplingAuraFlow` (shift=3)** |
| Negative conditioning | `CLIPTextEncode` | `CLIPTextEncode` | **`ConditioningZeroOut`** |
| Steps | 20–30 | 4–50 | **8** (turbo-optimised) |
| CFG | 7.0 | 1.0 / 4.0 | **1** |
| Sampler | `euler_ancestral` | `euler` | **`res_multistep`** |
| Scheduler | `karras` | `simple` | `simple` |
| Native resolution | 512/1024 | 1328×1328 | **1024×1024** |
| UNETLoader weight_dtype | varies | `default` | **`default`** |
| LoRA node | `LoraLoader` | `LoraLoaderModelOnly` | `LoraLoaderModelOnly` |
| ControlNet loader | `ControlNetLoader` | `QwenImageFunControlNetLoader` | `ZImageFunControlNetLoader` |
| ControlNet apply | `ControlNetApplyAdvanced` | `QwenImageFunControlNetApply` | `ZImageFunControlNetApply` |

---

## 1. Node graph structure

### Standard (no LoRA)

```
UNETLoader ("z_image_turbo_bf16.safetensors", weight_dtype="default")
    └──► ModelSamplingAuraFlow (shift=3)
             └──► KSampler ◄── CLIPTextEncode (positive) ← CLIPLoader (type="lumina2")
                           ◄── ConditioningZeroOut ← CLIPTextEncode (positive)
                           ◄── EmptySD3LatentImage (1024×1024)
                      └──► VAEDecode ◄── VAELoader ("ae.safetensors")
                               └──► SaveImage
```

Key differences from Qwen:
- CLIPLoader type is `"lumina2"` (not `"qwen_image"` or `"qwen3_4b"`)
- **`ModelSamplingAuraFlow` (shift=3) IS required** — placed between UNETLoader and KSampler
- Negative conditioning uses **`ConditioningZeroOut`** — NOT a CLIPTextEncode with text
- `ConditioningZeroOut` takes the positive CLIPTextEncode output and zeros it out
- Native resolution is 1024×1024, not 1328×1328

### With LoRA

```
UNETLoader
    └──► LoraLoaderModelOnly (style LoRA, strength=0.8)
             └──► ModelSamplingAuraFlow (shift=3)
                      └──► KSampler ...
```

### With Fun ControlNet

```
UNETLoader ──► [LoRA chain] ──► ModelSamplingAuraFlow ──► KSampler
CLIPTextEncode (positive) ──► ZImageFunControlNetApply ──► KSampler.positive
ConditioningZeroOut ──► ZImageFunControlNetApply ──► KSampler.negative
ZImageFunControlNetLoader ──► ZImageFunControlNetApply
LoadImage ──► [Preprocessor] ──► ZImageFunControlNetApply.image
```

---

## 2. KSampler settings

Z-Image-Turbo has a single recommended configuration:

| param | value | notes |
|---|---|---|
| steps | 8 | turbo-optimised NFE count |
| cfg | **1** | do NOT set to 0 |
| sampler_name | **`res_multistep`** | NOT euler |
| scheduler | `simple` | |
| denoise | 1.0 | |

```
set_param("<KSampler ID>", "steps", 8)
set_param("<KSampler ID>", "cfg",   1)
set_param("<KSampler ID>", "sampler_name", "res_multistep")
set_param("<KSampler ID>", "scheduler",    "simple")
```

**CRITICAL: Do NOT change cfg or sampler_name.** The official workflow uses cfg=1 and
res_multistep. Changing these will cause complete generation failure (noise/garbage output).

---

## 3. Resolutions (EmptySD3LatentImage)

Native resolution is 1024×1024. Supported aspect-ratio buckets:

| aspect | width | height |
|---|---|---|
| 1:1 (default) | 1024 | 1024 |
| 16:9 | 1280 | 720 |
| 9:16 | 720 | 1280 |
| 4:3 | 1152 | 864 |
| 3:4 | 864 | 1152 |

---

## 4. UNETLoader weight_dtype

Set to `"default"`. Do **not** use `"bfloat16"` — it is not in the allowed values list.

```
set_param("<UNETLoader ID>", "weight_dtype", "default")
```

---

## 5. Prompt engineering

Z-Image-Turbo uses Qwen3-4B as its text encoder (loaded via CLIPLoader type `lumina2`).
It understands natural language well. Use **descriptive sentences**, not keyword tag lists.
Keep prompts **concise and clear** — long verbose prompts degrade quality.

```
Good:
"A golden retriever sitting in a sunlit park. Green grass, shallow depth of field,
 warm afternoon light, photorealistic."

Poor:  "golden retriever, park, 8k, masterpiece, highly detailed"
Also poor: Very long prompts with excessive detail about lighting angles, camera lenses, etc.
```

**Negative prompt:** Not used — Z-Image-Turbo uses `ConditioningZeroOut` instead.
Do NOT add a CLIPTextEncode for negative conditioning.

---

## 6. LoRA — complete guide

### Architecture note

Z-Image-Turbo is a unified S3-DiT — all LoRA injection is **model-only**.
Use `add_lora_loader` without `clip_node_id`; the harness uses `LoraLoaderModelOnly`.

**Z-Image-Turbo LoRAs are NOT compatible with Qwen-Image-2512, and vice versa.**

### Adding a LoRA

```
add_lora_loader(
    lora_name      = "<exact filename from query_available_models>",
    model_node_id  = "<UNETLoader node ID>",
    strength_model = 0.8,
)
```

### Strength guidelines

| Use-case | strength_model |
|---|---|
| Style / texture | 0.60–0.80 |
| Subject / character | 0.70–0.90 |

### Stacking LoRAs

Chain LoRAs by passing the previous `LoraLoaderModelOnly` node ID as `model_node_id`:

```
# LoRA 1
add_lora_loader(lora_name="style_v1.safetensors", model_node_id="<UNETLoader>", strength_model=0.8)
# → returns node ID "101"

# LoRA 2 on top
add_lora_loader(lora_name="detail_v2.safetensors", model_node_id="101", strength_model=0.7)
```

Limit to 2–3 LoRAs; keep total strength sum ≤ 1.8 to avoid artifacts.

---

## 7. ControlNet — complete guide

### Architecture note

Z-Image-Turbo uses the **Fun ControlNet Union** format (same concept as Qwen-Image-2512
but with its own trained weights). Custom ComfyUI nodes are required:

- Loader: `ZImageFunControlNetLoader`
- Apply: `ZImageFunControlNetApply`

The harness automatically uses these when Z-Image-Turbo is detected.

**Z-Image ControlNet weights are NOT compatible with Qwen-Image-2512, and vice versa.**

### Available model

| Model file | Size | Source |
|---|---|---|
| `Z-Image-Turbo-Fun-Controlnet-Union.safetensors` | ~3.5 GB | `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union` |

Place in `ComfyUI/models/controlnets/`.

### Control modes (union_type)

| union_type | Preprocessor | Use case |
|---|---|---|
| `canny` | `CannyEdgePreprocessor` | Edge/structure preservation |
| `hed` | `HEDPreprocessor` | Soft structural guidance |
| `depth` | `MiDaS-DepthMapPreprocessor` | 3D layout, depth separation |
| `pose` | `DWPreprocessor` | Lock human skeleton |
| `mlsd` | `MLSDPreprocessor` | Architecture, straight lines |
| `scribble` | none (raw sketch) | Loose compositional sketch |
| `gray` | none (grayscale) | Luminosity/tone control |

### Parameters

| param | input name | range | default |
|---|---|---|---|
| Strength | `control_context_scale` | 0.0–1.0 | 0.80 |
| Start step | `start_percent` | 0.0–1.0 | 0.0 |
| End step | `end_percent` | 0.0–1.0 | 1.0 |
| Mode | `union_type` | see table | `canny` |

Recommended strength: 0.65–0.80.

---

## 8. Iteration strategy

| Verifier issue | Fix strategy |
|---|---|
| Dull / low contrast | Add detail to prompt; LoRA for texture/realism |
| Distorted anatomy | LoRA for anatomy correction |
| Wrong composition | Add spatial language; ControlNet depth or canny |
| Wrong pose | ControlNet pose at strength 0.90, end_percent=1.0 |
| Soft / blurry | More descriptive prompt; increase steps to 12 |
| ControlNet too rigid | Lower `control_context_scale` to 0.55–0.65 |
| LoRA artifacts | Reduce `strength_model` to 0.6; check BF16 compat. |
| Over-engineered prompt | Simplify prompt — shorter is better for this model |
| Complete generation failure | Check cfg=1, sampler=res_multistep, ModelSamplingAuraFlow present |

**NEVER change cfg or sampler_name** — doing so causes catastrophic failure.

---

## 9. Example: standard generation

```
inspect_workflow()

set_param("<pos node>", "text",
    "A young woman in a modern Tokyo street at night. Neon signs reflected
     in wet pavement, cinematic composition, shallow depth of field,
     photorealistic, warm amber and blue tones.")

set_param("<EmptySD3LatentImage>", "width",  1280)
set_param("<EmptySD3LatentImage>", "height",  720)

# DO NOT change cfg or sampler — leave at cfg=1, res_multistep
finalize_workflow()
```