---
name: learned-errors
description: Read this when ComfyUI returns validation errors, type mismatches, or workflow execution failures. Contains specific wiring rules, slot indices, and parameter constraints to prevent common connection errors.
---

# ComfyUI Workflow Error Prevention

## Critical Wiring Rules

### Return Type Mismatch Errors

**Problem**: "Return type mismatch between linked nodes" occurs when connecting incompatible output/input types.

**Common Causes**:
- Connecting CLIP output to a node expecting MODEL
- Mixing up VAE, CLIP, and MODEL connections
- Wrong output slot index from multi-output nodes

### Standard Checkpoint Loader Output Slots

**CheckpointLoaderSimple** (and variants) outputs:
- Slot 0: MODEL
- Slot 1: CLIP
- Slot 2: VAE

**Always verify**:
```
CheckpointLoaderSimple → slot 0 → KSampler.model
CheckpointLoaderSimple → slot 1 → CLIPTextEncode.clip
CheckpointLoaderSimple → slot 2 → VAEDecode.vae
```

## Parameter Validation Rules

### Value Range Constraints

**Problem**: "Value X bigger than max of Y" or "Value X smaller than min of Y"

**Common Parameter Limits**:
- **Conditioning strength** (`conditioning_to_strength`, `conditioning_from_strength`): Must be 0.0–1.0
- **Denoise**: Must be 0.0–1.0
- **CFG scale**: Typically 1.0–30.0 (node-specific)
- **Steps**: Must be positive integer, typically 1–10000

**Always check**:
1. Conditioning interpolation parameters stay within [0.0, 1.0]
2. Strength/weight parameters respect documented ranges
3. Integer parameters are whole numbers when required

### Common Nodes with Strict Limits

**ConditioningAverage, ConditioningCombine variants**:
- `conditioning_to_strength`: max 1.0
- Any blend/mix ratio: typically 0.0–1.0

**KSampler**:
- `denoise`: 0.0–1.0
- `steps`: positive integer

## Common Node Connection Patterns

**KSampler inputs**:
- `model`: requires MODEL type (checkpoint slot 0)
- `positive`: requires CONDITIONING (from CLIPTextEncode)
- `negative`: requires CONDITIONING (from CLIPTextEncode)
- `latent_image`: requires LATENT (from EmptyLatentImage or VAEEncode)

**CLIPTextEncode inputs**:
- `clip`: requires CLIP type (checkpoint slot 1)
- `text`: requires STRING

**VAEDecode inputs**:
- `samples`: requires LATENT (from KSampler)
- `vae`: requires VAE type (checkpoint slot 2)

**VAEEncode inputs**:
- `pixels`: requires IMAGE
- `vae`: requires VAE type (checkpoint slot 2)

## Troubleshooting Checklist

When you encounter validation errors:

1. **Check slot indices**: Verify output slot numbers match expected types
2. **Check parameter ranges**: Ensure all numeric inputs are within min/max bounds
3. **Trace MODEL path**: Ensure MODEL flows from loader slot 0 to sampler
4. **Trace CLIP path**: Ensure CLIP flows from loader slot 1 to text encoders
5. **Trace VAE path**: Ensure VAE flows from loader slot 2 to encode/decode nodes
6. **Verify CONDITIONING**: Both positive and negative must come from CLIPTextEncode outputs
7. **Check LATENT flow**: KSampler outputs LATENT, which goes to VAEDecode

## Type Compatibility Quick Reference

| Output Type | Compatible Input Nodes |
|-------------|------------------------|
| MODEL | KSampler, LoRA loaders, model patchers |
| CLIP | CLIPTextEncode, CLIP manipulators |
| VAE | VAEDecode, VAEEncode |
| CONDITIONING | KSampler (positive/negative), conditioning combiners |
| LATENT | VAEDecode, KSampler (latent_image) |
| IMAGE | VAEEncode, PreviewImage, SaveImage |

## Prevention Strategy

- **Always use explicit slot indices** in connections
- **Never assume slot order** - verify against node definitions
- **Validate parameter ranges** before setting values (especially 0.0–1.0 constraints)
- **Double-check multi-output nodes** (loaders, samplers)
- **Match types exactly** - no implicit conversions exist
- **Clamp conditioning/strength values** to [0.0, 1.0] range