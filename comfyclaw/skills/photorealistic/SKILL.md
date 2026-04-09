# Skill: Photorealistic

## Description
Produce photorealistic, camera-like images for scenes described as real photographs.
Trigger when the user uses words like "photo", "photograph", "realistic", "DSLR", or "cinematic".

## Instructions
Apply these workflow adjustments:
1. **Prompt enhancement**: Append to the positive text:
   ", RAW photo, DSLR, 85mm lens, f/1.8, natural lighting, photorealistic, hyper detailed"
2. **Negative prompt**: Add "cartoon, drawing, painting, anime, sketch, illustration, 3d render, cgi"
3. **Steps**: Set to 30 for high fidelity detail.
4. **CFG**: Set to 7.5 for tight adherence to the prompt.
5. **Sampler**: Use "dpmpp_2m" with "karras" scheduler if available for photorealistic quality.
