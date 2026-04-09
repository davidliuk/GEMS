# Skill: High Quality

## Description
Boost perceptual quality for product shots, portraits, and generic high-fidelity renders.
Trigger when the user asks for "high quality", "detailed", "sharp", or "professional" images.

## Instructions
Apply these workflow adjustments:
1. **Prompt enhancement**: Add quality boosters after the main subject description —
   append ", masterpiece, best quality, highly detailed, 8k uhd, sharp focus, professional photography"
   to the positive CLIPTextEncode text.
2. **Negative prompt**: If there is a negative CLIPTextEncode node, set its text to
   "blurry, low quality, bad anatomy, watermark, signature, jpeg artifacts, noise, overexposed"
3. **Sampler steps**: Set KSampler steps to at least 20. If currently < 20, increase to 25.
4. **CFG scale**: Set KSampler cfg to 7.0 for crisp detail.
5. **Resolution**: Prefer 768×768 or 1024×1024 in EmptySD3LatentImage / EmptyLatentImage.
