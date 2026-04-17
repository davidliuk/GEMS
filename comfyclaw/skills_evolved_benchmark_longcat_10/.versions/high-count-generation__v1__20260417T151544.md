---
name: high-count-generation
description: >-
  Generate 6-8 objects of the same type using latent tiling, iterative composition, and explicit enumeration to overcome diffusion models' count limitations
license: MIT
metadata:
  cluster: "high_count_object_generation"
  origin: "self-evolve"
---

# High Count Generation (6-8 Objects)

## When to Use
Trigger when the user requests 6, 7, or 8 objects of the same type ("seven green croissants", "six purple trucks", "eight cats"). Do NOT use for mixed categories — defer to cross-category-composition.

## Core Strategy
Diffusion models struggle with counts above 5 because attention disperses too widely. Use spatial decomposition and explicit enumeration.

## Prompt Structure
```
(exactly {N} {object}:1.4), {object} 1, {object} 2, {object} 3, {object} 4, {object} 5, {object} 6[, {object} 7][, {object} 8], arranged in [rows|grid|circle], separate and distinct, multiple {object}s, full scene visible
Negative: merged, overlapping, fused, duplicate, blurry
```

## Workflow Modifications

### Option A: Latent Tiling (Preferred for 6-7)
1. Use `EmptyLatentImage` with larger canvas: 1280×768 or 1536×768
2. Increase `batch_latent_count` in KSampler to 2-3 batches
3. Add `LatentUpscaleBy` node (factor 1.25) before sampling for more spatial resolution
4. Set sampler steps to 35-40 (higher for count stability)

### Option B: Regional Composition (for 7-8)
1. Query `list_custom_nodes()` for regional conditioning nodes
2. Divide canvas into 2-3 horizontal or grid regions
3. Apply same prompt to each region with explicit count distribution:
   - Region 1: "three {object}s"
   - Region 2: "three {object}s" (for 6 total)
   - Region 3: "one {object}" (for 7 total)

### Option C: Iterative Inpainting (fallback)
1. Generate base image with 4 objects using medium-count-generation
2. Use `InpaintModelConditioning` to add 2-3 more objects in empty regions
3. Mask should cover 30-40% of canvas for each iteration

## Key Parameters
- CFG: 8.0-9.0 (higher for count enforcement)
- Steps: 35-45
- Resolution: minimum 1280×768 for 6+, 1536×768 for 7-8
- Sampler: dpmpp_2m or euler_ancestral

## Verification
After generation, if the verifier reports wrong count, retry with Option B (regional) if Option A was used, or increase CFG by 1.0 and add "counted, numbered" to prompt.