---
name: multi-object-layout
description: >-
  Apply regional conditioning or latent composition to generate scenes with 3+ distinct objects where counting accuracy is critical
license: MIT
metadata:
  cluster: "counting_multiple_objects"
  origin: "self-evolve"
---

# Multi-Object Layout

## When to use
Trigger when:
- User requests 3 or more instances of the same object type ("seven croissants", "five bears")
- Prompt contains explicit counts >= 3 with object names
- Verifier reports wrong object count or missing objects in multi-object scenes
- fix_strategy contains "use_regional_layout" or "compositional_generation"

## Core problem
Diffusion models struggle with exact counts beyond 2-3 objects because attention mechanisms don't maintain discrete object identity. Repeating "cat cat cat cat" in a prompt doesn't guarantee 4 cats.

## Solution strategy

### Option 1: Regional prompt conditioning (for arranged objects)
1. Call `regional-control` skill to set up region-based prompting
2. Divide canvas into N spatial regions (grid or linear layout)
3. Assign one object instance per region with position-specific prompts:
   - "green croissant in top-left corner"
   - "green croissant in top-center"
   - etc.
4. Use ConditioningSetArea or regional-control nodes to bind each prompt to its region

### Option 2: Iterative composition (for scattered objects)
1. Generate objects individually or in small batches (1-2 at a time)
2. Use inpainting or latent blending to composite into single scene
3. Track object count across iterations
4. Requires LatentComposite or InpaintModelConditioning nodes

### Option 3: Attention masking (advanced)
1. If available, use attention manipulation nodes to enforce object separation
2. Apply cross-attention masks that prevent object feature bleeding
3. Requires custom nodes like AttentionCouple or ComposableLoRA

## Implementation steps
1. Parse the requested count from prompt (extract number)
2. Choose strategy based on count:
   - 3-4 objects: Try regional conditioning first
   - 5-7 objects: Use grid layout with tight regional bounds
   - 7+ objects: Warn user and suggest iterative generation
3. Restructure prompt to remove redundant object repetition
4. Apply spatial keywords: "arranged in a row", "grid pattern", "evenly spaced"
5. Increase resolution if needed (objects need minimum pixel budget)
6. Set CFG slightly higher (7.5-9.0) to maintain object distinctness

## Example transformation
Input: "seven green croissants"
Output workflow:
- Use EmptyLatentImage 1024x512 (wide canvas)
- Create 7 regional conditions in horizontal strip layout
- Each region: "single green croissant, detailed pastry, isolated on white"
- Negative: "multiple croissants, overlapping, blurry"
- Apply ConditioningSetArea for each region with width=146px, x offsets: 0, 146, 292, 438, 584, 730, 876
- Combine with ConditioningCombine before KSampler

## Fallback
If regional nodes unavailable, inject strong spatial language via `spatial` skill and increase step count to 35-40 to give model more refinement passes.