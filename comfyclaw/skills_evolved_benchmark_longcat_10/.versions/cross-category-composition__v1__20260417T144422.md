---
name: cross-category-composition
description: >-
  Generate scenes mixing distinct object categories (animals with objects, different animal types together) using semantic regional prompts and attention balancing to prevent category collapse
license: MIT
metadata:
  cluster: "multi_object_composition"
  origin: "self-evolve"
---

# Cross-Category Composition

## Problem
Diffusion models struggle when prompts mix different object categories (e.g., 'pig and backpack', 'car and flowers', 'bears and donut'). One category often dominates or objects blend incorrectly.

## Detection Triggers
- Prompt contains multiple distinct object categories from different domains
- Mix of: animals + inanimate objects, different animal species, vehicles + nature, food + animals
- Verifier reports missing object categories or merged objects
- Benchmark prompts like 'a green backpack and a pig'

## Solution Strategy

### 1. Category Identification
Parse prompt and identify distinct semantic categories:
- Animals (mammals, birds, etc.)
- Vehicles (cars, planes, etc.) 
- Objects (backpack, furniture, etc.)
- Food (donut, pizza, etc.)
- Nature (flowers, trees, etc.)

### 2. Semantic Regional Prompts
Use `RegionalConditioningSimple` or `ConditioningSetMask` nodes:
- Allocate spatial regions for each category (e.g., left=animal, right=object)
- Each region gets a focused prompt with ONLY its category
- Example: 'a pig' in left mask, 'a green backpack' in right mask
- Add weak global prompt for scene cohesion: 'outdoor scene, natural lighting'

### 3. Attention Balancing
- Use `ConditioningAverage` with weight 0.6-0.8 for category-specific conditioning
- Apply `ConditioningConcat` to combine regional prompts
- For categories with strong priors (animals), slightly reduce attention weight (0.55)
- For weak categories (small objects), boost weight (0.7-0.75)

### 4. Negative Prompts
- Add category-specific negatives: 'merged objects, blended animals, hybrid creature'
- Prevent semantic leakage: 'animal-shaped backpack, vehicle-shaped animal'

### 5. Layout Guidance
If categories are equal importance:
- Use symmetric layout (side-by-side, diagonal)
- Set equal mask sizes
- Add compositional tokens: 'standing next to', 'beside', 'near'

If one category dominates:
- Foreground/background separation
- Scale emphasis: '(large pig:1.2), (small backpack:0.9)'

## Node Recipe
```
CLIPTextEncode → category_1_prompt → ConditioningSetMask(mask_1)
CLIPTextEncode → category_2_prompt → ConditioningSetMask(mask_2)
ConditioningCombine → [masked_cond_1, masked_cond_2]
KSampler → combined_conditioning
```

## When NOT to Use
- Single category repeated (use counting-objects)
- Spatial relationships only (use spatial skill)
- Unusual attributes on single object (use unusual-attributes)

## Success Metrics
- All categories present in output
- No semantic blending between categories
- Objects retain category-specific features
- Natural spatial arrangement