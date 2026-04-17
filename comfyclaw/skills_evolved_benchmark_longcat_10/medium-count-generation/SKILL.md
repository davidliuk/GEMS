---
name: medium-count-generation
description: >-
  Generate accurate counts of 4-6 objects using regional prompt splitting, per-object conditioning, and negative prompt count reinforcement to prevent undercounting
license: MIT
metadata:
  cluster: "medium_count_objects"
  origin: "self-evolve"
---

# Medium Count Generation (4-6 objects)

## When to use
- User requests exactly 4, 5, or 6 instances of the same object type
- Especially critical when objects have color/material modifiers ("four purple lions", "five spotted birds")
- Also applies to mixed-category scenes with medium counts ("four rabbits and a sheep")

## Core technique: Regional per-object conditioning

For counts of 4+, text-only prompting collapses. Use regional conditioning with explicit per-object splits:

1. **Split the canvas into N equal regions** using ConditioningSetArea or regional prompt nodes
2. **Apply one object instance per region** with identical prompts
3. **For modified objects** (color/material/pattern), include the modifier in EVERY regional prompt
4. **Use negative prompts** to suppress undercounting: "only one, only two, only three"

## Workflow changes

### Four objects (2×2 grid):
```
For "four purple lions":
- Split 1024×1024 into four 512×512 regions (top-left, top-right, bottom-left, bottom-right)
- Region 1: "a purple lion, single animal" (area: 0,0,512,512)
- Region 2: "a purple lion, single animal" (area: 512,0,512,512)
- Region 3: "a purple lion, single animal" (area: 0,512,512,512)
- Region 4: "a purple lion, single animal" (area: 512,512,512,512)
- Combine all four conditionings with ConditioningCombine
- Negative: "only one lion, only two lions, only three lions, crowd"
```

### Five objects (2×2 grid + center):
```
For "five bears and a donut":
- Four bears in 2×2 grid (each 512×512)
- One bear in center overlap region (256,256,512,512)
- One donut in separate region (768,768,256,256)
- Negative: "only one bear, only two bears, only three bears, only four bears"
```

### Six objects (2×3 or 3×2 grid):
- Divide canvas into 6 equal tiles
- Apply one instance per tile
- Same negative prompt pattern

## Prompt structure per region

**Always include modifiers in each regional prompt:**
- ✅ "a brown monkey, single primate" (repeated 4 times for "four brown monkeys")
- ❌ "a monkey" with global "brown" token (models ignore global color with regional conditioning)

**Add singularity emphasis:**
- "single animal", "one creature", "individual object" in each region
- Prevents the model from packing multiple instances into one region

## When NOT to use

- Counts of 1-3: Use standard prompting or counting-objects skill
- Counts of 7+: Escalate to multi-object-layout or latent tiling
- Abstract arrangements ("a pile of", "many"): This is for exact counts only

## Success criteria

- Exact count matches request (verify with object detection)
- All objects retain specified modifiers (color, material, pattern)
- Objects are distinct and separated (no merging/duplication)
- Mixed-category scenes maintain both counts ("four rabbits AND one sheep" = 5 total)