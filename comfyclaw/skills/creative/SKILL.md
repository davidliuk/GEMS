# Skill: Creative

## Description
Encourage vivid, imaginative outputs for concept art, fantasy scenes, and artistic prompts.
Trigger when the user asks for "creative", "artistic", "fantasy", "concept art", or "surreal" images.

## Instructions
Apply these workflow adjustments:
1. **Prompt enhancement**: Append to the positive text:
   ", concept art, vivid colors, dynamic composition, highly detailed digital painting, trending on artstation"
2. **Negative prompt**: Add "bland, boring, plain, flat colors, low detail"
3. **CFG scale**: Lower to 6.0–6.5 to allow more creative freedom.
4. **Sampler**: Prefer "euler_ancestral" sampler_name if available, for varied outputs.
5. **Seed**: Change to a random odd number (e.g. 12345) to escape local minima.
