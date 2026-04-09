# Skill: ControlNet Control

## Description
Add ControlNet branches to enforce spatial / structural constraints that text prompts alone cannot.
Trigger when verifier reports: flat background, wrong 3D layout, blurry edges, pose/anatomy issues,
no surface texture, or fix_strategy "add_controlnet_*".

## Instructions

### ControlNet type selection
| Problem | ControlNet type | preprocessor_class | Typical model name keywords |
|---|---|---|---|
| Flat background, no depth separation | Depth | MiDaS-DepthMapPreprocessor | "depth", "depth-midas" |
| Blurry edges, lost structure | Canny | CannyEdgePreprocessor | "canny", "lineart" |
| Wrong human pose / body structure | Pose | DWPreprocessor | "openpose", "dw-pose" |
| Poor surface texture, low detail | Tile | TilePreprocessor | "tile" |
| Segment bleed (subject ↔ background) | Seg | OneFormer-ADE20K | "seg", "ade20k" |
| Wrong surface normals / 3D feel | Normal | NormalMapSimple | "normal", "normal-bae" |

### Workflow for adding ControlNet
1. Call `query_available_models("controlnets")` to see installed models.
2. If no suitable ControlNet exists, skip this upgrade and try a different fix strategy.
3. You need a **control image** (LoadImage node). If none exists in the workflow:
   - For depth/canny: the generated image itself can be a control image via VAEDecode → pass
   - For pose: you likely need an external reference image (skip if unavailable)
4. Call `add_controlnet` with:
   - `controlnet_name`: exact filename from query output
   - `preprocessor_class`: see table above (or "" to pass image directly)
   - `image_node_id`: node providing the image to preprocess
   - `positive_node_id` / `negative_node_id`: existing conditioning nodes
   - `strength`: 0.5–0.8 for subtle; 0.8–1.2 for strong enforcement
   - `start_percent` / `end_percent`: use 0.0–0.7 to avoid over-constraint at final steps

### Strength tuning
- Too weak (< 0.4): ControlNet has little effect
- Good range (0.5–0.8): balanced structure + creative freedom
- Too strong (> 1.0): image becomes stiff and over-constrained
- For detail tile: 0.3–0.6 is usually sufficient

### Combining multiple ControlNets
Chain them: the output conditioning from one `ControlNetApplyAdvanced` feeds the next.
The last one's output feeds the KSampler.
