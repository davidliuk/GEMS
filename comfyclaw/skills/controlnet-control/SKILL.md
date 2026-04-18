---
name: controlnet-control
description: >-
  Add a ControlNet branch to enforce spatial and structural constraints that
  text prompts alone cannot express. Use when the verifier reports flat
  background, wrong 3D layout, blurry edges, incorrect pose, poor surface
  texture, or when fix_strategy contains "add_controlnet". ControlNet is the
  right tool whenever structure needs to be locked in rather than coaxed
  through prompting.
license: MIT
compatibility: ComfyClaw agent — call query_available_models("controlnets") before use.
allowed-tools: query_available_models add_controlnet
metadata:
  author: davidliuk
  version: "0.2.0"
tags: [agent]
---

ControlNet solves structural problems that prompting cannot — the diffusion
model's spatial understanding is limited, but a depth map or pose skeleton
gives it a concrete blueprint to follow.

## Step 1 — Query installed ControlNets

```
query_available_models("controlnets")
```

If no suitable model is installed, stop here and try a different fix strategy.

> **Qwen-Image-2512 / Z-Image-Turbo note:** call the same `add_controlnet`
> tool, but the harness automatically switches to the **model-patch** wiring
> for these archs (`ModelPatchLoader` + `QwenImageDiffsynthControlnet` for
> Qwen, `ZImageFunControlnet` for Z-Image). Two consequences:
>
> 1. `image_node_id` is **mandatory** (the patch node takes an image input) —
>    wire in a `LoadImage` or a `VAEDecode` before calling.
> 2. The workflow **must contain a `VAELoader`** — the patch node consumes
>    VAE.
> 3. `union_type`, `start_percent`, `end_percent` are ignored — Qwen uses
>    per-file control modes (canny vs depth) and neither patch node exposes a
>    schedule. See the per-model skills for the exact filenames.
>
> SD/SDXL ControlNet weights are architecturally incompatible with Qwen /
> Z-Image and vice versa.
>
> **LongCat-Image**: ControlNet is not supported at all for this arch
> (pipeline-style `LongCatImageTextToImage` node exposes no MODEL tensor to
> patch). The tool call will fail with a clear message; use prompt-level or
> LoRA-equivalent guidance instead.

## Step 2 — Choose the right ControlNet type

| Verifier problem | ControlNet type | Preprocessor class | Model keywords |
|---|---|---|---|
| Flat background / no depth separation | Depth | `MiDaS-DepthMapPreprocessor` | `depth`, `depth-midas` |
| Blurry edges / lost structure | Canny | `CannyEdgePreprocessor` | `canny`, `lineart` |
| Wrong human pose or body layout | Pose | `DWPreprocessor` | `openpose`, `dw-pose` |
| Poor surface texture / low detail | Tile | `TilePreprocessor` | `tile` |
| Subject bleeding into background | Seg | `OneFormer-ADE20K` | `seg`, `ade20k` |
| Wrong surface normals / 3D feel | Normal | `NormalMapSimple` | `normal`, `normal-bae` |

Pick the type that most directly addresses the top verifier issue.

## Step 3 — Identify or create a control image

You need a **control image** node (`LoadImage`) as the source signal.

- **Depth / Canny**: the previously generated image works well — wire a
  VAEDecode output into the preprocessor.
- **Pose**: needs an external reference image showing the target pose.
  Skip this type if no reference image is available in the workflow.
- **Tile**: use the generated image itself (or a low-res version of it).

## Step 4 — Add with `add_controlnet`

```
add_controlnet(
  controlnet_name    = "<exact filename from query>",
  preprocessor_class = "<see table above, or '' to pass image directly>",
  image_node_id      = "<LoadImage or VAEDecode node ID>",
  positive_node_id   = "<existing positive CLIPTextEncode node ID>",
  negative_node_id   = "<existing negative CLIPTextEncode node ID>",
  strength           = 0.65,   # starting point
  start_percent      = 0.0,
  end_percent        = 0.7,    # stop before final steps to preserve creativity
)
```

`end_percent = 0.7` prevents the ControlNet from over-constraining the final
denoising steps where fine detail and colour are determined.

## Strength tuning

- **0.5–0.7** — balanced structure with creative freedom (default range)
- **0.7–1.0** — strong enforcement; use for pose or when structure is severely wrong
- **> 1.0** — the image becomes rigid and over-constrained; avoid
- **Tile ControlNet** — use 0.3–0.5; tile is sensitive and escalates quickly

## Chaining multiple ControlNets

Wire them in series: `ControlNetApplyAdvanced` outputs feed the next one's
conditioning inputs. The final output feeds KSampler. Limit to two per
iteration; more than two creates competing constraints.
