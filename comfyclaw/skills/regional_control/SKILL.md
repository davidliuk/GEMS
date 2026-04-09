# Skill: Regional Attention Control

## Description
Use separate text prompts for different regions of the image (foreground subject vs. background,
top vs. bottom, etc.). Trigger when verifier reports: subject and background don't match,
scene elements compete stylistically, background is too plain/generic relative to subject,
or fix_strategy "add_regional_prompt".

## Instructions

### When to use regional attention
- Subject needs a rich, detailed prompt but background needs minimal attention
- Different areas have fundamentally different style requirements
- Subject style is "contaminating" the background or vice versa
- You need to suppress certain elements only in specific areas

### BREAK-token approach (simple, widely compatible)
- Format: `[foreground prompt] BREAK [background prompt]`
- Compatible with most SD models without additional nodes
- The model gives roughly equal attention to each section
- Limitation: no precise spatial masking; useful for loose region guidance

### ConditioningCombine approach (more powerful)
Call `add_regional_attention` with:
- `foreground_prompt`: detailed subject description
- `background_prompt`: background/environment description
- `foreground_weight`: 1.2–1.5 to emphasize subject (default 1.3)

The tool creates:
1. Two separate CLIPTextEncode nodes
2. ConditioningAverage to weight the foreground
3. ConditioningCombine to merge both

### Prompt writing for regional control
- **Foreground prompt**: Subject + quality modifiers + lighting on subject
  Example: "cute tabby cat, detailed fur texture, soft rim lighting, sharp focus"
- **Background prompt**: Environment + mood, avoiding subject-specific terms
  Example: "sunlit wooden windowsill, warm afternoon bokeh, shallow depth of field"
- Keep prompts concise — complex regional prompts can conflict

### What NOT to put in regional prompts
- Don't repeat the same terms in both (causes blending, not separation)
- Don't put negative keywords here (add them to negative conditioning instead)
