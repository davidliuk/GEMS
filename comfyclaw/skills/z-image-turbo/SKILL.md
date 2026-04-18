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
tags: [agent, "model:z-image-turbo"]
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
| ControlNet loader | `ControlNetLoader` | `ModelPatchLoader` | `ModelPatchLoader` |
| ControlNet apply | `ControlNetApplyAdvanced` | `QwenImageDiffsynthControlnet` (model-patch) | `ZImageFunControlnet` (model-patch) |

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

### With Fun ControlNet (model-patch style)

Z-Image-Turbo ControlNet patches the MODEL tensor — positive / negative
conditioning are left untouched.

```
UNETLoader ──► [LoRA chain] ──► ZImageFunControlnet ──► ModelSamplingAuraFlow ──► KSampler
                                       ▲    ▲    ▲
                                       │    │    └── LoadImage (control reference)
                                       │    └────── VAELoader
                                       └──────────  ModelPatchLoader (Z-Image-Turbo-Fun-Controlnet-Union-*.safetensors)

CLIPTextEncode (positive) ──► KSampler.positive     (unchanged)
ConditioningZeroOut      ──► KSampler.negative     (unchanged)
```

`add_controlnet` **requires an `image_node_id`** (LoadImage / VAEDecode) and
the workflow must contain a `VAELoader`.

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

### Public LoRA weights (place in `ComfyUI/models/loras/`)

| File | Size | Focus |
|---|---|---|
| `Z-Image-Turbo-Radiant-Realism-Pro.safetensors` | ~163 MB | Realism + lighting polish |
| `Z-Image-Turbo-Realism-LoRA.safetensors` | ~82 MB | Lighter realism tweak |

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

Z-Image-Turbo uses the **Fun ControlNet Union** format — wired as a
**model-patch** (not conditioning-patch). Nodes used (both built-in):

- Loader: `ModelPatchLoader` (generic — loads from `ComfyUI/models/model_patches/`)
- Apply:  `ZImageFunControlnet`
  - Inputs: `model`, `model_patch`, `vae`, `image`, `strength`
  - Output: a patched `MODEL` that replaces the original MODEL at every
    downstream consumer (KSampler, ModelSamplingAuraFlow).

The harness auto-detects Z-Image and uses these when you call `add_controlnet`.
There is **no `union_type` input** on `ZImageFunControlnet` — the Union file
ships all modes together and the preprocessor (or a pre-baked control image)
selects which signal is used.

**Z-Image ControlNet weights are NOT compatible with Qwen-Image-2512, and vice versa.**

### Available model file (place in `ComfyUI/models/model_patches/`)

| File | Size | Source |
|---|---|---|
| `Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors` | ~6.3 GB | `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union` |

### Control modes (chosen by the preprocessor, not by `union_type`)

| Preprocessor | Use case |
|---|---|
| `CannyEdgePreprocessor` | Edge / structure preservation |
| `HEDPreprocessor` | Soft structural guidance |
| `MiDaS-DepthMapPreprocessor` / `DepthAnythingV2Preprocessor` | 3D layout, depth separation |
| `DWPreprocessor` | Lock human skeleton |
| `MLSDPreprocessor` | Architecture, straight lines |
| `""` (none) | Pass a pre-computed map (sketch, grayscale, canny PNG) directly |

### Parameters

| param | input name | range | default |
|---|---|---|---|
| Strength | `strength` | 0.0–1.0 | 0.80 |

`union_type`, `start_percent`, `end_percent` are accepted by `add_controlnet`
for API symmetry but are **not forwarded** to `ZImageFunControlnet` — it has
no schedule or mode switch.

Recommended strength: 0.65–0.85.

---

## 8. Iteration strategy

| Verifier issue | Fix strategy |
|---|---|
| Dull / low contrast | Add detail to prompt; LoRA for texture/realism |
| Distorted anatomy | LoRA for anatomy correction |
| Wrong composition | Add spatial language; ControlNet depth or canny |
| Wrong pose | ControlNet pose at strength 0.90, end_percent=1.0 |
| Soft / blurry | More descriptive prompt; increase steps to 12 |
| ControlNet too rigid | Lower `strength` to 0.50–0.65 |
| ControlNet node error "needs a VAELoader" | Add a `VAELoader` in the workflow and connect it upstream of VAEDecode |
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

## 10. Example: LoRA + ControlNet (depth)

```
# Add style LoRA (model-patch chain)
add_lora_loader(
    lora_name      = "Z-Image-Turbo-Radiant-Realism-Pro.safetensors",
    model_node_id  = "<UNETLoader ID>",
    strength_model = 0.9,
)

# Add depth ControlNet (also on the MODEL branch — harness finds current MODEL source)
add_controlnet(
    controlnet_name    = "Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors",
    preprocessor_class = "MiDaS-DepthMapPreprocessor",
    image_node_id      = "<reference photo node>",    # REQUIRED
    positive_node_id   = "<pos node>",
    negative_node_id   = "<neg node>",
    strength           = 0.75,
)

# DO NOT change cfg or sampler — leave at cfg=1, res_multistep
finalize_workflow()
```