# Skill: Hires Fix & Refinement

## Description
Add a second-pass high-resolution refinement stage to recover fine detail that low-resolution
base generation loses. Trigger when verifier reports: blurry, low resolution, lack of fine detail,
soft texture at full view, or fix_strategy "add_hires_fix".

## Instructions

### When to use hires fix
- Base latent resolution ≤ 768×768 and output looks soft/blurry
- Fine details (fur, fabric, text, faces) appear mushy in the base output
- You have improved prompt/ControlNet/LoRA but resolution remains a bottleneck
- Score is acceptable but "could be sharper"

### Standard workflow (latent upscale approach)
Call `add_hires_fix` with:
- `base_ksampler_node_id`: the existing KSampler
- `vae_node_id`: the VAELoader
- `scale_by`: 1.5 for moderate upscale; 2.0 for 2× (expensive)
- `upscale_method`: "bicubic" (smooth) or "lanczos" (sharp)
- `hires_steps`: 10–20 (fewer is fine since denoise is low)
- `hires_denoise`: 0.35–0.55 (lower = preserve composition; higher = more new detail)

### Denoise strength guidance
| Goal | hires_denoise |
|---|---|
| Preserve exact composition, just sharpen | 0.25–0.40 |
| Balanced: sharpen + add detail | 0.40–0.55 |
| Heavy refinement (risks layout drift) | 0.55–0.70 |

### Inpaint pass vs. hires fix
- Use **hires fix** when the whole image is low quality
- Use **add_inpaint_pass** when only a specific region (e.g. hands) needs repair
- Combine: do hires fix first, then a targeted inpaint pass if one region remains problematic

### Cost consideration
- 1.5× latent upscale + 15 hires steps ≈ +60% compute
- 2.0× upscale ≈ +200% compute
- Start with 1.5× and denoise 0.45 as a baseline
