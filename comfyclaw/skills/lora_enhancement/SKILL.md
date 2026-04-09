# Skill: LoRA Enhancement

## Description
Add LoRA models to boost specific quality dimensions that the base model struggles with.
Trigger when verifier reports: plasticky texture, flat lighting, poor anatomy, wrong style,
missing fine detail, or explicit fix_strategy "inject_lora_*".

## Instructions

### When to use which LoRA type
| Verifier issue | Recommended LoRA type | Typical filename keywords |
|---|---|---|
| Plasticky / waxy skin or surfaces | Detail / texture LoRA | "add-detail", "detail-tweaker", "texture" |
| Wrong / distorted anatomy (hands, fingers, faces) | Anatomy fix LoRA | "hand", "better-hands", "face-fix", "better-faces" |
| Flat / artificial lighting | Lighting LoRA | "lighting", "illumination", "cinematic-light" |
| Style inconsistency | Style LoRA | match scene: "oil-paint", "photorealism", "anime", etc. |
| Soft / blurry overall quality | Quality booster LoRA | "xl-detail", "quality", "sharpness" |

### How to add a LoRA
1. Always call `query_available_models("loras")` first to see what is installed.
2. Pick the most relevant LoRA for the top issue.
3. Call `add_lora_loader` with:
   - `model_node_id`: the UNETLoader / CheckpointLoaderSimple node ID
   - `clip_node_id`: the CLIPLoader / CheckpointLoaderSimple node ID
   - `strength_model`: start at 0.7–0.85; reduce to 0.5 if style is too strong
   - `strength_clip`: same as strength_model by default
4. You can chain multiple LoRA loaders in sequence (output of one feeds next).

### Strength guidelines
- Detail / texture LoRA: strength 0.6–0.8
- Style LoRA: strength 0.4–0.7 (too high causes style bleed)
- Anatomy fix LoRA: strength 0.5–0.9 (higher when anatomy is seriously wrong)

### Prompt adjustments after adding LoRA
- Trigger keywords: some LoRAs activate on specific words (e.g. `<lora:hand_fix:0.8>`).
  Add the LoRA's trigger token to the positive prompt if known.
- Keep negative prompt updated: add terms that LoRA-enhanced models sometimes produce.
