---
name: medium-count-generation
description: >-
  Generate accurate counts of 4-5 objects through explicit enumeration, attention reinforcement, and optional latent tiling techniques
license: MIT
metadata:
  cluster: "medium_count_objects"
  origin: "self-evolve"
---

# Medium Count Generation (4-5 Objects)

## When to Use
Trigger when the user requests exactly 4 or 5 objects of the same or different types in a single scene. This skill targets the medium-count regime where simple prompting fails but full regional conditioning may be excessive.

## Core Techniques

### 1. Explicit Enumeration in Prompt
Rewrite the prompt to enumerate each object individually:
- Input: "four brown monkeys"
- Output: "a brown monkey, a second brown monkey, a third brown monkey, a fourth brown monkey, exactly four monkeys in total"
- Input: "five blue birds"
- Output: "a blue bird, another blue bird, a third blue bird, a fourth blue bird, a fifth blue bird, exactly five birds"

### 2. Attention Reinforcement
Use emphasis tokens to strengthen object presence:
- Wrap the enumerated list in (parentheses:1.3) or [brackets:1.2]
- Add the exact count phrase at both start and end: "exactly four X, ..., four X total"
- Example: "(exactly four purple lions:1.3), a purple lion, a second purple lion, a third purple lion, a fourth purple lion, (four lions in total:1.2)"

### 3. Negative Prompts for Count Accuracy
Add to negative prompt:
- "three objects, two objects, one object, six objects, seven objects"
- "missing objects, incomplete count, fewer than [N], more than [N]"

### 4. Mixed-Type Scenes
For combinations like "four rabbits and a sheep":
- Enumerate the larger group: "a rabbit, a second rabbit, a third rabbit, a fourth rabbit, exactly four rabbits"
- Add the singleton clearly: "and one sheep, a single sheep"
- Full example: "(exactly four rabbits and one sheep:1.3), a rabbit, another rabbit, a third rabbit, a fourth rabbit, four rabbits total, and one sheep standing nearby"

### 5. Sampler Configuration
Adjust for better object coherence:
- Increase steps by 20-30% (e.g., 30→40) to allow more attention refinement
- Use cfg_scale 7.5-9.0 for stronger prompt adherence
- Prefer samplers with better composition: dpmpp_2m, euler_ancestral

### 6. Latent Dimensions
Use wider aspect ratios to give objects physical space:
- For 4-5 objects: prefer 1344×768 or 1216×832 over square 1024×1024
- Horizontal arrangements reduce overlap and counting errors

## Node-Level Implementation
No new nodes required — apply these changes in CLIPTextEncode (positive/negative prompts) and KSampler (steps, cfg). If failures persist after 2 attempts, escalate to regional-control or multi-object-layout for full spatial decomposition.

## Success Criteria
Verifier confirms exact object count matches the request, all objects are distinct and visible, no phantom duplicates or missing instances.