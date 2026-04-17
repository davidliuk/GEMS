---
name: learned-errors
description: Read this when building or modifying ComfyUI workflows to avoid common validation and execution errors
---

# ComfyUI Workflow Error Prevention

## Critical: Prompt Must Have Outputs

`Prompt has no outputs` means **no terminal node exists** in the workflow.

**Fix**: Every workflow MUST include at least one output node:
- `SaveImage`
- `PreviewImage`

**Without a terminal node, the workflow is invalid.**

## Required Input Errors - TOP PRIORITY

`Required input is missing` means a node has an unwired mandatory input or missing parameter in `inputs` dict.

### VAEDecode - MOST COMMON ERROR

**ALWAYS verify VAEDecode has BOTH inputs wired:**
- **`samples`** - LATENT from KSampler slot 0 or other LATENT source
  - **NEVER pass string literal, file path, null, or leave unwired**
  - **NEVER pass IMAGE data - only accepts LATENT type**
  - **NEVER use wrong wire reference format - MUST be `["node_id", 0]` array**
  - Common errors:
    - `'str' object has no attribute 'shape'` = wrong data type passed
    - `Required input is missing: samples` = unwired or invalid wire reference
- **`vae`** - VAE from CheckpointLoaderSimple slot 2

**Before submitting, ALWAYS check every VAEDecode node has valid `samples` wire in `["node_id", int_slot]` format.**

### KSampler - ALL Inputs Required

**4 WIRED inputs (all mandatory - NEVER leave unwired):**
- **`model`** - from CheckpointLoaderSimple slot 0
- **`positive`** - from CLIPTextEncode slot 0
- **`negative`** - from CLIPTextEncode slot 0
- **`latent_image`** - from EmptyLatentImage slot 0 or VAEEncode slot 0

**6 PARAMETERS in `inputs` dict (all mandatory - NEVER omit):**
- **`seed`** - **MUST be integer >= 0** (NEVER -1, null, float like `123.0`, or string like `"123"`)
  - Error `Failed to convert an input value to a INT value` means you passed float/string instead of int
- **`steps`** - integer 1-10000
- **`cfg`** - float 0.0-30.0
- **`denoise`** - float 0.0-1.0
- **`sampler_name`** - string
- **`scheduler`** - string

### CLIPTextEncode - Both Required

- **`text`** - **MUST be non-empty string in `inputs` dict** (NEVER null, empty string, or omitted)
  - Error `Required input is missing: text` = you forgot to add `text` key or set it to null/empty
- **`clip`** - wired from CheckpointLoaderSimple slot 1

### ControlNetApply - All Three Required

- **`conditioning`** - wired from CLIPTextEncode slot 0 or previous ControlNetApply slot 0
- **`control_net`** - wired from ControlNetLoader slot 0
- **`image`** - wired from LoadImage slot 0 or preprocessor output

### FluxGuidance - Both Required

- **`guidance`** - float parameter in `inputs` dict (NEVER omit)
- **`conditioning`** - wired CONDITIONING input

### Other Common Missing Inputs

- **SaveImage/PreviewImage**: `images` (from VAEDecode slot 0 - MUST be IMAGE type)
- **VAEEncode**: `pixels` (from LoadImage slot 0), `vae` (from checkpoint slot 2)

## Critical Node Output Slots

**CheckpointLoaderSimple**:
- Slot 0: MODEL
- Slot 1: CLIP
- Slot 2: VAE

**KSampler**: Slot 0: LATENT
**VAEDecode**: Slot 0: IMAGE
**VAEEncode**: Slot 0: LATENT
**CLIPTextEncode**: Slot 0: CONDITIONING
**EmptyLatentImage**: Slot 0: LATENT
**LoadImage**: Slot 0: IMAGE, Slot 1: MASK
**ControlNetLoader**: Slot 0: CONTROL_NET
**ControlNetApply**: Slot 0: CONDITIONING
**FluxGuidance**: Slot 0: CONDITIONING

## Wire Reference Format - CRITICAL

**ALL wire references MUST be `["node_id", slot_index]` where:**
- `node_id` is **string** matching existing node's `id`
- `slot_index` is **integer** (0, 1, or 2)
- Array has exactly 2 non-null elements
- MUST be array format, NEVER object/dict/string/null

**Common errors:**
- Using dict: `{"node": "1", "slot": 0}` ❌ → `["1", 0]` ✓
- String slot: `["1", "0"]` ❌ → `["1", 0]` ✓
- Missing brackets: `"1", 0` ❌ → `["1", 0]` ✓
- Null values: `["1", null]` ❌ → `["1", 0]` ✓
- Wrong node ID: `["99", 0]` where node "99" doesn't exist ❌

**Invalid wire format causes `Required input is missing` even if you think input is wired.**

## Parameter Type and Range Errors

**Integer parameters** (use Python int in `inputs` dict - NEVER float/string):
- **`seed`** - **CRITICAL: MUST be Python int >= 0** 
  - ❌ `123.0` (float), `"123"` (string), `-1`, `null`
  - ✓ `123` (int)
  - Error `Failed to convert an input value to a INT value` = you passed wrong type
- `steps` (1-10000)
- `batch_size`, `width`, `height`

**Float parameters** (use Python float in `inputs` dict):
- **`conditioning_to_strength`** - **MUST be 0.0-1.0, NEVER exceed 1.0**
  - Error `Value 1.2 bigger than max of 1.0` = you exceeded range
- `denoise` (0.0-1.0)
- `cfg` (0.0-30.0)
- `guidance` (for FluxGuidance, typically 3.0-4.0)
- **ALWAYS validate max ranges - exceeding causes `value_bigger_than_max` error**

**String parameters** (non-empty string in `inputs` dict):
- **`text`** (for CLIPTextEncode - REQUIRED, NEVER null/empty/omitted)
- `sampler_name`, `scheduler`

## Runtime Errors

**"'str' object has no attribute 'shape'"** means:
- Wrong data type passed to node input (e.g., string instead of LATENT/IMAGE)
- Most common: passing non-LATENT data to VAEDecode `samples` input

**"string index out of range"** means:
- Invalid wire reference format (not `["string_id", int_slot]`)
- Referencing non-existent node ID
- Accessing wrong slot index
- Malformed JSON structure in workflow

## Pre-Submission Checklist

- [ ] At least one SaveImage/PreviewImage exists
- [ ] **All VAEDecode nodes have `samples` wired in valid `["node_id", 0]` format AND `vae` wired**
- [ ] All KSampler nodes have 4 wired inputs + 6 parameters including **integer** `seed` >= 0 (not float/string)
- [ ] **All CLIPTextEncode nodes have non-empty `text` string in `inputs` dict AND `clip` wired**
- [ ] All ControlNetApply nodes have `conditioning`, `control_net`, AND `image` wired
- [ ] All