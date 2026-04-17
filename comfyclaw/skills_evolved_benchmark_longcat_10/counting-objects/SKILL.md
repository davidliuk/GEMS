---
name: counting-objects
description: >-
  Restructure prompts and workflow to generate exact counts of objects, especially when requesting more than 2-3 items of the same type.
license: MIT
metadata:
  cluster: "counting_multiple_objects"
  origin: "self-evolve"
---

# Counting Objects

## When to Use
Trigger when the prompt contains:
- Explicit numeric counts: "five cats", "seven croissants", "ten bottles"
- Number words from four onward ("four rabbits", "eight cars")
- Counting failures reported by verifier: wrong object count, missing instances
- Multiple identical objects in a scene

## Why This is Hard
Diffusion models have no native cardinality mechanism. Attention weights spread across similar objects, causing:
- Undercounting (requesting 7, getting 3-4)
- Overcounting or merging objects into ambiguous blobs
- Degradation beyond 2-3 items

## Solution Strategy

### 1. Decompose Count into Explicit Layout
Replace "seven croissants" with spatial enumeration:
```
a row of seven croissants: one croissant on the far left, one croissant left of center, one croissant at center, one croissant right of center, one croissant on the far right, one croissant in the back left, one croissant in the back right
```

### 2. Use Regional Prompting for Subgroups
For counts ≥6, split into regions (requires regional-control):
- "four rabbits" → region 1: "two rabbits", region 2: "two rabbits"
- "seven croissants" → region 1: "three croissants", region 2: "four croissants"

### 3. Increase CFG for Count Adherence
Boost CFG by +1.5 to +2.5 above baseline to strengthen text conditioning on count tokens.

### 4. Avoid Ambiguous Plural
Replace "rabbits" with "four separate rabbits, each distinct" or "exactly four individual rabbits".

### 5. Use Negative Prompts
Add to negative: "merged objects, blurry count, ambiguous number, too many, too few"

## Example Transform
**Before:** "seven green croissants"
**After:** "exactly seven individual green croissants arranged in a line, each croissant clearly separated, one two three four five six seven croissants, highly detailed count"

Combine with unusual-attributes if color/material is non-standard.
Combine with regional-control for counts ≥5.

## Implementation
- Use prompt_rewriter or modify CLIPTextEncode inputs directly
- Adjust KSampler cfg parameter: add +2.0 for counting precision
- Layer with spatial skill if layout is also specified