---
name: material-rendering
description: >-
  Apply explicit material properties (metal, wood, plastic, stone, glass, ceramic, fabric) to objects through prompt restructuring and emphasis tokens
license: MIT
metadata:
  cluster: "material_texture_application"
  origin: "self-evolve"
---

# Material Rendering Skill

## When to Use
Trigger when the prompt contains explicit material descriptors:
- Metals: "metal", "metallic", "steel", "bronze", "copper", "chrome", "gold", "silver"
- Natural: "wooden", "wood", "stone", "rock", "marble", "granite"
- Synthetic: "plastic", "rubber", "vinyl", "acrylic"
- Transparent: "glass", "crystal", "transparent"
- Special: "sparkling", "glowing", "iridescent", "glossy", "matte"

Also trigger when verifier reports missing material qualities or texture mismatch.

## Prompt Restructuring

### Material Token Placement
1. **Prefix pattern**: Place material BEFORE the object noun
   - Good: "(metal:1.3) toy robot"
   - Bad: "toy robot made of metal"

2. **Emphasis syntax**: Use weight modifiers for materials
   - Standard emphasis: (material:1.2)
   - Strong emphasis for weak materials: (plastic:1.4), (wooden:1.3)
   - Metals need less: (metal:1.1) often sufficient

3. **Descriptor stacking**: Add material-specific qualities
   - Metal: "(metallic:1.2) (shiny:1.1) (reflective:1.1) steel"
   - Wood: "(wooden:1.3) (wood grain texture:1.2) natural wood"
   - Plastic: "(plastic:1.4) (smooth plastic surface:1.2)"
   - Stone: "(stone:1.3) (carved from stone:1.2) (rough texture:1.1)"

### Multi-Object Material Handling
When multiple objects have different materials:
1. Use regional-control skill for spatial separation
2. Apply material emphasis per region:
   - Region 1: "(metal:1.3) zebra, metallic surface, chrome"
   - Region 2: "(wooden:1.3) zebra, wood grain, carved wood"

### Negative Prompts
Add material exclusions to prevent bleed:
- For metal objects: negative="fabric, soft, fuzzy, cloth"
- For wooden objects: negative="metal, plastic, shiny, reflective"
- For plastic objects: negative="natural, organic, metal, wood"
- For stone objects: negative="soft, flexible, fabric, plastic"

## Example Transformations

**Input**: "two metal toys"
**Output**: "two (metallic:1.2) (shiny:1.1) toys, chrome finish, reflective metal surface, steel toys"
**Negative**: "plastic, fabric, wooden, matte"

**Input**: "three metal zebras"
**Output**: "three (metal:1.3) zebras, (metallic sculpture:1.2), chrome zebras, polished steel animal figures"
**Negative**: "fur, organic, plastic, painted"

**Input**: "a wooden flower and a raccoon"
**Output**: "a (wooden:1.3) flower carved from wood with (wood grain:1.2) texture, and a realistic raccoon"
**Negative**: "plastic flower, metal flower, fabric"

**Input**: "a plastic motorcycle and a clock"
**Output**: "a (plastic:1.4) (toy:1.2) motorcycle with (smooth plastic surface:1.2), and a clock"
**Negative**: "metal motorcycle, real motorcycle, die-cast"

## Implementation
1. Detect material keywords in prompt
2. Restructure with prefix pattern + emphasis
3. Add material-specific descriptors
4. Build negative prompt with material exclusions
5. If 3+ objects with different materials, invoke regional-control
6. Update CLIPTextEncode nodes with restructured prompt

## Success Metrics
- Material descriptor appears in generated image
- Surface reflectivity/texture matches material type
- No material confusion between multiple objects