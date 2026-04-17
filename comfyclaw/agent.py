"""
ClawAgent — LLM agent that evolves ComfyUI workflow topology.

The agent uses LiteLLM's unified completion API (OpenAI-compatible tool-use
format) to call structured workflow operations in a loop until it calls
``finalize_workflow``.  Any provider supported by LiteLLM can be used —
Anthropic, OpenAI, Google Gemini, local Ollama, etc.

Tool categories
---------------
Inspection  : inspect_workflow, query_available_models
Prompt      : set_prompt  (auto-resolves sampler→encoder links; no node ID needed)
Basic edit  : set_param, add_node, connect_nodes, delete_node
LoRA        : add_lora_loader
ControlNet  : add_controlnet
Regional    : add_regional_attention
Refinement  : add_hires_fix, add_inpaint_pass
Control     : report_evolution_strategy, finalize_workflow
"""

from __future__ import annotations

import copy
import json
import os
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable

import litellm

from .skill_manager import SkillManager
from .stage_router import StageRouter
from .workflow import WorkflowManager


def _abbreviate_tool_args(tool_name: str, args: dict) -> dict:
    """Return a small dict suitable for the thinking-log UI.

    Large values (e.g. full skill text, long prompts) are truncated so we
    don't flood the WebSocket channel.
    """
    MAX_VAL = 120
    out: dict = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > MAX_VAL:
            out[k] = v[:MAX_VAL] + "…"
        elif isinstance(v, dict) and len(json.dumps(v)) > MAX_VAL:
            out[k] = "{…}"
        else:
            out[k] = v
    return out

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_BASE = """\
You are ComfyClaw, an expert ComfyUI workflow engineer.  Your job is to BUILD
and GROW ComfyUI workflow topologies — constructing complete pipelines from
scratch when the workflow is empty, and evolving existing ones in response to
the verifier's region-level feedback.

Iteration strategy
------------------
1. Call report_evolution_strategy first: state your plan and the top issue.
2. Call inspect_workflow to see the current topology.
3. **If the workflow is empty** (no nodes):
   a. Call read_skill("workflow-builder") to load architecture recipes.
   b. Call query_available_models("checkpoints") and query_available_models("diffusion_models")
      to discover available models — NEVER guess filenames.
   c. Match the model filename to an architecture (SD 1.5, SDXL, Flux, Qwen, etc.)
      using the patterns in the workflow-builder skill.
   d. Build the full pipeline node-by-node using add_node, following the matching recipe.
   e. Use ONLY exact filenames from query results.
   f. Set detailed prompts on the CLIPTextEncode nodes.
   g. Call validate_workflow to catch wiring errors before submitting.
   h. Call finalize_workflow (it auto-validates and blocks if errors remain).
4. **If the workflow already has nodes**, follow the evolution strategy:
   a. Call set_prompt — craft a detailed, professional positive prompt AND a strong
      negative prompt based on the user's goal (see "Prompt engineering" below).
      Do this EVERY iteration, even if you also plan structural changes.
   b. If a relevant skill is listed in <available_skills>, call read_skill to load
      its full instructions BEFORE applying that upgrade.
   c. Call query_available_models BEFORE adding any LoRA or ControlNet node.
   d. Apply structural upgrades (LoRA / ControlNet / regional / hires / inpaint).
   e. Tune sampler parameters (steps, CFG, seed) as needed.
   f. Call validate_workflow to catch wiring errors.
   g. Call finalize_workflow when done (it auto-validates).

Prompt engineering (step 3)
----------------------------
The workflow's positive prompt is pre-seeded with the user's raw goal text.
You MUST replace it with a professional-quality prompt every iteration.

Positive prompt — structure:
  [subject & scene], [style], [lighting], [camera/lens], [quality boosters]
  • Expand every meaningful concept: vague nouns → vivid adjectives + nouns.
  • Add artistic / photographic style: "cinematic", "concept art", "photorealistic",
    "watercolor painting", "isometric", etc.
  • Add lighting: "golden hour", "dramatic rim lighting", "neon glow", "soft diffuse".
  • Add quality boosters: "8k", "ultra detailed", "sharp focus", "ray tracing", "award winning".
  • If the image has multiple subjects/regions, describe each clearly.

Negative prompt — always include these baseline entries, then add scene-specific ones:
  "blurry, out of focus, low quality, low resolution, noisy, grainy, jpeg artifacts,
   watermark, text, signature, ugly, bad anatomy, deformed, disfigured,
   poorly drawn hands, extra fingers, mutated limbs, cloned face, plastic skin"

Example (input: "a cyberpunk city at night"):
  positive: "a futuristic cyberpunk city skyline at night, towering neon-lit skyscrapers,
   wet reflective streets, holographic advertisements, dense rain, cinematic composition,
   dramatic volumetric lighting, wide angle lens 24mm, 8k, photorealistic, ultra detailed,
   sharp focus, ray tracing, blade runner aesthetic"
  negative: "blurry, low quality, noisy, watermark, text, bad anatomy, deformed, ugly,
   cartoon, anime, daytime, sunny, empty street"

Using skills (progressive disclosure)
--------------------------------------
Skills are specialised instruction sets for specific upgrade patterns.  At startup
you can see each skill's name and description in <available_skills>.  When you decide
to apply a skill, call read_skill("<skill-name>") to load the full instructions.  Only
load skills you actually intend to use — each read consumes context.

Decision heuristics
-------------------
  Workflow is EMPTY (no nodes)          → read_skill("workflow-builder") FIRST.
                                         Then query_available_models to pick arch.
  Workflow contains QwenImageModelLoader → read_skill("qwen-image-2512") FIRST.
                                         Qwen has NO KSampler/ControlNet/LoRA —
                                         all tuning is on RH_QwenImageGenerator.
  Active model contains "longcat"      → read_skill("longcat-image") FIRST.
                                         LongCat uses CFGNorm + FluxGuidance, not
                                         ModelSamplingAuraFlow. No LoRA/ControlNet.
  Active model contains "z_image"      → read_skill("z-image-turbo") FIRST.
                                         Z-Image uses cfg=1, sampler=res_multistep,
                                         ConditioningZeroOut for negative. NEVER change cfg/sampler.
  Active model name contains "lcm"     → read_skill("dreamshaper8-lcm") FIRST, before
                                         any sampler tuning — LCM needs different
                                         steps/cfg/sampler than standard SD models.
  Flat / low-depth background          → read_skill("controlnet-control"), add depth
  Blurry edges / lost structure        → read_skill("controlnet-control"), add canny
  Wrong human pose / body              → read_skill("controlnet-control"), add pose
  Plasticky skin / poor texture        → read_skill("lora-enhancement"), detail LoRA
  Wrong anatomy (hands, fingers)       → read_skill("lora-enhancement"), anatomy LoRA
  Style inconsistency                  → read_skill("lora-enhancement"), style LoRA
  Subject and background bleed         → read_skill("regional-control")
  Low resolution / soft fine detail    → read_skill("hires-fix")
  Localised artifact in one area       → add_inpaint_pass
  User asks for photorealistic image   → read_skill("photorealistic")
  User asks for high quality / sharp   → read_skill("high-quality")
  User asks for creative / artistic    → read_skill("creative")
  Prompt is flat / needs artistic depth→ read_skill("prompt-artist")
  Multiple objects with spatial layout → read_skill("spatial")
  Text / sign / label in the image     → read_skill("text-rendering")

Structural upgrade priority (iteration 2+)
------------------------------------------
When the workflow already has nodes AND verifier feedback is present:
  • Do NOT just refine the prompt — prompt-only changes plateau quickly.
  • PREFER structural upgrades: LoRA, ControlNet, hires-fix, regional, inpaint.
  • If ANY region_issue has fix_strategies containing "inject_lora_*" or
    "add_controlnet_*", you MUST attempt that structural upgrade, not fall
    back to prompt tweaking. Call query_available_models first; if a matching
    model exists, read the corresponding skill and apply it.
  • Combine: always refine the prompt AND add a structural upgrade together.
  • Only fall back to prompt-only when no LoRA / ControlNet / inpaint models
    are installed or the fix strategies are exclusively prompt-related.

Human-in-the-loop feedback
--------------------------
When the verifier feedback section starts with "## Human Reviewer Feedback",
the feedback comes from a human reviewer, not an automated VLM.
Human feedback expresses subjective preferences — style, mood, composition,
color palette, artistic direction.  Prioritize these over structural/technical
changes.  Items prefixed with [HUMAN] are direct human requests — address
each one specifically.  Do not second-guess or override human preferences.

Node parameter constraints (DO NOT violate)
-------------------------------------------
  UNETLoader weight_dtype:  "default" | "fp8_e4m3fn" | "fp8_e4m3fn_fast" | "fp8_e5m2"
                            Never use "fp16" or "fp32" — causes HTTP 400.
  Apple MPS cannot run FP8 models. If you see a Float8_e4m3fn MPS error,
  set weight_dtype to "default" and do not attempt further dtype changes.
  LoRA class is "LoraLoader" (not LoRALoader).
  ControlNet apply class is "ControlNetApplyAdvanced".
"""

_PINNED_MODEL_SECTION = """\

Pinned image model
------------------
The image-generation model for this session is LOCKED to: {model_name}

  • DO NOT change the ckpt_name / unet_name of any CheckpointLoaderSimple,
    CheckpointLoader, or UNETLoader node.
  • DO NOT delete or replace existing loader nodes.
  • You MAY add LoRA loaders on top of the pinned model — LoraLoader wires
    between the primary loader and the sampler, leaving the base model intact.
  • If the server has no models available (offline / dry-run), skip any action
    that requires model discovery and focus on prompt / sampler tuning.
"""


def _build_system_prompt(
    pinned_image_model: str | None,
    available_skills_xml: str = "",
) -> str:
    """Return the full system prompt with optional pinned-model and skills sections."""
    prompt = _SYSTEM_PROMPT_BASE
    if available_skills_xml:
        prompt += f"\n{available_skills_xml}\n"
    if pinned_image_model:
        prompt += _PINNED_MODEL_SECTION.format(model_name=pinned_image_model)
    return prompt


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


def _tool(name: str, description: str, parameters: dict) -> dict:
    """Wrap a tool definition in OpenAI / LiteLLM function-calling format."""
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


_TOOLS: list[dict] = [
    _tool(
        "inspect_workflow",
        "Return a human-readable summary of all nodes, IDs, class types, and key inputs.",
        {"type": "object", "properties": {}, "required": []},
    ),
    _tool(
        "query_available_models",
        (
            "Query the ComfyUI server for available models of a given type. "
            "Call this BEFORE adding LoRA or ControlNet nodes."
        ),
        {
            "type": "object",
            "properties": {
                "model_type": {
                    "type": "string",
                    "description": "loras | controlnets | checkpoints | unets | vae | upscale_models | clip_vision",
                }
            },
            "required": ["model_type"],
        },
    ),
    _tool(
        "set_param",
        "Set a scalar input on a specific node.",
        {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "param_name": {"type": "string"},
                "value": {},
            },
            "required": ["node_id", "param_name", "value"],
        },
    ),
    _tool(
        "add_node",
        "Append a new node to the workflow. Returns the new node ID.",
        {
            "type": "object",
            "properties": {
                "class_type": {"type": "string"},
                "nickname": {"type": "string"},
                "inputs": {"type": "object"},
            },
            "required": ["class_type"],
        },
    ),
    _tool(
        "connect_nodes",
        "Wire src_node output slot to dst_node input name.",
        {
            "type": "object",
            "properties": {
                "src_node_id": {"type": "string"},
                "src_output_index": {"type": "integer"},
                "dst_node_id": {"type": "string"},
                "dst_input_name": {"type": "string"},
            },
            "required": ["src_node_id", "src_output_index", "dst_node_id", "dst_input_name"],
        },
    ),
    _tool(
        "delete_node",
        "Remove a node and clean up its connections.",
        {
            "type": "object",
            "properties": {"node_id": {"type": "string"}},
            "required": ["node_id"],
        },
    ),
    _tool(
        "add_lora_loader",
        (
            "Insert a LoraLoader between the model/clip source and all downstream consumers. "
            "Call query_available_models('loras') first."
        ),
        {
            "type": "object",
            "properties": {
                "lora_name": {"type": "string"},
                "strength_model": {"type": "number"},
                "strength_clip": {"type": "number"},
                "model_node_id": {"type": "string"},
                "clip_node_id": {"type": "string"},
            },
            "required": ["lora_name", "model_node_id", "clip_node_id"],
        },
    ),
    _tool(
        "add_controlnet",
        (
            "Add ControlNetLoader + optional preprocessor + ControlNetApplyAdvanced. "
            "Call query_available_models('controlnets') first."
        ),
        {
            "type": "object",
            "properties": {
                "controlnet_name": {"type": "string"},
                "preprocessor_class": {"type": "string"},
                "image_node_id": {"type": "string"},
                "positive_node_id": {"type": "string"},
                "negative_node_id": {"type": "string"},
                "strength": {"type": "number"},
                "start_percent": {"type": "number"},
                "end_percent": {"type": "number"},
            },
            "required": [
                "controlnet_name",
                "preprocessor_class",
                "positive_node_id",
                "negative_node_id",
            ],
        },
    ),
    _tool(
        "add_regional_attention",
        (
            "Split conditioning into foreground and background regional prompts using "
            "BREAK tokens + ConditioningCombine."
        ),
        {
            "type": "object",
            "properties": {
                "positive_node_id": {"type": "string"},
                "clip_node_id": {"type": "string"},
                "foreground_prompt": {"type": "string"},
                "background_prompt": {"type": "string"},
                "foreground_weight": {"type": "number"},
            },
            "required": [
                "positive_node_id",
                "clip_node_id",
                "foreground_prompt",
                "background_prompt",
            ],
        },
    ),
    _tool(
        "add_hires_fix",
        "Add a hires-fix pass: LatentUpscaleBy + second KSampler + VAEDecode.",
        {
            "type": "object",
            "properties": {
                "upscale_method": {"type": "string"},
                "scale_by": {"type": "number"},
                "hires_steps": {"type": "integer"},
                "hires_denoise": {"type": "number"},
                "base_ksampler_node_id": {"type": "string"},
                "vae_node_id": {"type": "string"},
                "save_image_node_id": {"type": "string"},
            },
            "required": ["base_ksampler_node_id", "vae_node_id"],
        },
    ),
    _tool(
        "add_inpaint_pass",
        "Add a targeted inpaint pass for a specific region.",
        {
            "type": "object",
            "properties": {
                "region_description": {"type": "string"},
                "inpaint_prompt": {"type": "string"},
                "denoise_strength": {"type": "number"},
                "base_ksampler_node_id": {"type": "string"},
                "positive_node_id": {"type": "string"},
                "clip_node_id": {"type": "string"},
                "vae_node_id": {"type": "string"},
                "save_image_node_id": {"type": "string"},
            },
            "required": [
                "region_description",
                "inpaint_prompt",
                "base_ksampler_node_id",
                "positive_node_id",
                "clip_node_id",
                "vae_node_id",
            ],
        },
    ),
    _tool(
        "set_prompt",
        (
            "Set the positive and/or negative prompt text across the whole workflow. "
            "Automatically locates the CLIPTextEncode (or equivalent) nodes connected "
            "to every sampler — no node ID required. "
            "Call this EARLY to replace the raw user goal with a detailed, "
            "professional-quality prompt.  Pass empty string to leave a slot unchanged."
        ),
        {
            "type": "object",
            "properties": {
                "positive_text": {
                    "type": "string",
                    "description": (
                        "Expanded positive prompt.  Structure: "
                        "[subject & scene], [style], [lighting], [camera/lens], [quality boosters].  "
                        "Example: 'a futuristic city skyline at night, neon reflections on wet streets, "
                        "cyberpunk aesthetic, dramatic cinematic lighting, wide angle lens, "
                        "8k, photorealistic, ultra detailed, sharp focus, ray tracing'."
                    ),
                },
                "negative_text": {
                    "type": "string",
                    "description": (
                        "Negative prompt listing artefacts and failure modes to suppress.  "
                        "Always include baseline: 'blurry, low quality, low resolution, noisy, "
                        "watermark, text, bad anatomy, deformed, ugly'.  "
                        "Add model- or scene-specific negatives as needed."
                    ),
                },
            },
            "required": [],
        },
    ),
    _tool(
        "report_evolution_strategy",
        "Declare your evolution plan BEFORE making changes.",
        {
            "type": "object",
            "properties": {
                "strategy": {"type": "string"},
                "top_issue": {"type": "string"},
            },
            "required": ["strategy", "top_issue"],
        },
    ),
    _tool(
        "validate_workflow",
        (
            "Check the workflow graph for common errors BEFORE finalizing. "
            "Returns a list of issues (dangling refs, wrong slot indices, missing outputs). "
            "Call this after building or repairing the workflow to catch mistakes early."
        ),
        {"type": "object", "properties": {}, "required": []},
    ),
    _tool(
        "finalize_workflow",
        "Signal that all modifications are complete. Call validate_workflow first to catch errors.",
        {
            "type": "object",
            "properties": {"rationale": {"type": "string"}},
            "required": ["rationale"],
        },
    ),
    _tool(
        "read_skill",
        (
            "Load the full instructions for a named skill (progressive disclosure). "
            "Call this BEFORE applying a skill's technique. "
            "Available skill names are listed in <available_skills> in the system prompt."
        ),
        {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Exact skill name as shown in <available_skills>.",
                },
            },
            "required": ["skill_name"],
        },
    ),
    _tool(
        "explore_nodes",
        (
            "Explore the ComfyUI server's node ecosystem. Queries /object_info, "
            "classifies every node into pipeline stages, and returns a stage map "
            "showing which nodes and tools are relevant at each workflow phase."
        ),
        {"type": "object", "properties": {}, "required": []},
    ),
    _tool(
        "transition_stage",
        (
            "Advance to the next pipeline stage. The agent progresses through: "
            "planning -> construction -> conditioning -> enhancement -> finalization. "
            "Only tools relevant to the current stage are exposed. "
            "Call this when the current stage's work is complete."
        ),
        {
            "type": "object",
            "properties": {
                "target_stage": {
                    "type": "string",
                    "description": (
                        "Stage to transition to: planning | construction | "
                        "conditioning | enhancement | finalization"
                    ),
                },
                "rationale": {
                    "type": "string",
                    "description": "Why this stage transition is appropriate now.",
                },
            },
            "required": ["target_stage"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ClawAgent:
    """
    Parameters
    ----------
    api_key         : API key for the LLM provider.  For Anthropic this is the
                      ``ANTHROPIC_API_KEY``; for OpenAI ``OPENAI_API_KEY``, etc.
                      When provided it is written into the matching env-var so
                      that LiteLLM can pick it up automatically.  You can also
                      set the env-var directly and leave this empty.
    model           : LiteLLM model string, e.g. ``"anthropic/claude-sonnet-4-5"``,
                      ``"openai/gpt-4o"``, ``"gemini/gemini-2.0-flash"``,
                      ``"ollama/llama3.1"``.
    server_address  : ComfyUI HTTP address (used for model queries).
    skills_dir      : Path to pre-defined skills/ folder; ``None`` uses built-in.
    evolved_skills_dir : Path to evolved/learned skills; ``None`` uses built-in
                      ``skills_evolved/`` directory.
    on_change       : Called with the workflow dict after every mutation.
    max_tool_rounds : Safety cap on tool-call iterations.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "anthropic/claude-sonnet-4-5",
        server_address: str = "127.0.0.1:8188",
        skills_dir: str | None = None,
        evolved_skills_dir: str | None = None,
        on_change: Callable[[dict], None] | None = None,
        max_tool_rounds: int = 40,
        pinned_image_model: str | None = None,
        stage_gated: bool = False,
    ) -> None:
        # Propagate the API key into the environment so LiteLLM picks it up.
        # For Anthropic keys (sk-ant-…) we set ANTHROPIC_API_KEY; for other
        # providers users should set the appropriate env-var themselves.
        if api_key:
            if api_key.startswith("sk-ant-"):
                os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
            else:
                os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
        self.model = model
        self.server_address = server_address
        self.skill_manager = SkillManager(skills_dir, evolved_skills_dir=evolved_skills_dir)
        self.on_change = on_change
        self.on_agent_event: Callable[[str, str, str, dict | None], None] | None = None
        self.max_tool_rounds = max_tool_rounds
        self.pinned_image_model = pinned_image_model
        self.stage_router = StageRouter(enabled=stage_gated)

        # SFT trace: populated after each plan_and_patch() call
        self.last_messages: list[dict] | None = None
        self.last_token_usage: dict | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def plan_and_patch(
        self,
        workflow_manager: WorkflowManager,
        original_prompt: str,
        verifier_feedback: str | None = None,
        memory_summary: str | None = None,
        iteration: int = 1,
    ) -> str:
        """
        Drive the LLM tool-use loop to evolve the workflow.

        Uses LiteLLM's OpenAI-compatible completion API so any provider
        (Anthropic, OpenAI, Gemini, Ollama, …) can be used.

        Returns the rationale string from the ``finalize_workflow`` call.
        """
        user_content = self._build_user_message(
            original_prompt=original_prompt,
            workflow_manager=workflow_manager,
            verifier_feedback=verifier_feedback,
            memory_summary=memory_summary,
            iteration=iteration,
        )

        system_prompt = _build_system_prompt(
            self.pinned_image_model,
            available_skills_xml=self.skill_manager.build_available_skills_xml(),
        )

        # System prompt as the first message (OpenAI / LiteLLM convention).
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        rationale = "(no rationale provided)"
        rounds = 0
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        self.stage_router.reset()
        self._emit_event("info", f"Starting agent loop (model: {self.model})")

        while rounds < self.max_tool_rounds:
            rounds += 1
            active_tools = self.stage_router.filter_tools(_TOOLS)
            resp = litellm.completion(
                model=self.model,
                max_tokens=4096,
                tools=active_tools,
                messages=messages,
            )
            choice = resp.choices[0]
            finish_reason = choice.finish_reason

            if hasattr(resp, "usage") and resp.usage:
                total_usage["prompt_tokens"] += getattr(resp.usage, "prompt_tokens", 0) or 0
                total_usage["completion_tokens"] += getattr(resp.usage, "completion_tokens", 0) or 0
                total_usage["total_tokens"] += getattr(resp.usage, "total_tokens", 0) or 0

            assistant_msg = choice.message
            messages.append(assistant_msg)

            # Emit any assistant text as thinking
            if assistant_msg.content:
                self._emit_event("thinking", assistant_msg.content)

            if finish_reason in ("stop", "end_turn"):
                self._emit_event("info", "Agent finished planning.")
                break
            if finish_reason != "tool_calls":
                print(
                    f"[ClawAgent] Unexpected finish_reason: {finish_reason!r}",
                    file=sys.stderr,
                )
                break

            done = False
            for tc in assistant_msg.tool_calls or []:
                name = tc.function.name
                try:
                    inputs = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    inputs = {}

                # Emit tool_call event with abbreviated args
                abbrev = _abbreviate_tool_args(name, inputs)
                self._emit_event("tool_call", f"Calling {name}", name, abbrev)

                result_text, should_stop = self._dispatch(name, inputs, workflow_manager)

                # Emit tool_result event
                self._emit_event(
                    "tool_result",
                    result_text[:300],
                    name,
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    }
                )
                if should_stop:
                    rationale = inputs.get("rationale", rationale)
                    self._emit_event("info", f"Finalized: {rationale[:200]}")
                    done = True

            if done:
                break

        # Persist full conversation trace for SFT data collection.
        # Messages may contain LiteLLM ModelResponse objects as assistant turns;
        # serialize them to plain dicts for JSON compatibility.
        self.last_messages = self._serialize_messages(messages)
        self.last_token_usage = total_usage

        return rationale

    # ------------------------------------------------------------------
    # Tool dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, name: str, inputs: dict, wm: WorkflowManager) -> tuple[str, bool]:
        """Route a single tool call. Returns ``(result_text, should_stop)``."""
        try:
            match name:
                case "inspect_workflow":
                    return WorkflowManager.summarize(wm.workflow), False

                case "query_available_models":
                    return self._query_models(inputs["model_type"]), False

                case "set_prompt":
                    pos = inputs.get("positive_text") or ""
                    neg = inputs.get("negative_text") or ""
                    pos_nodes, neg_nodes = wm.inject_prompt(
                        positive=pos if pos else None,
                        negative=neg if neg else None,
                    )
                    self._notify(wm)
                    parts = []
                    if pos_nodes:
                        parts.append(f"positive → nodes {pos_nodes}: {pos!r}")
                    if neg_nodes:
                        parts.append(f"negative → nodes {neg_nodes}: {neg!r}")
                    if not parts:
                        return (
                            "⚠️ set_prompt: no CLIPTextEncode nodes found connected to a sampler. Use set_param directly.",
                            False,
                        )
                    return "✅ " + "; ".join(parts), False

                case "set_param":
                    wm.set_param(str(inputs["node_id"]), inputs["param_name"], inputs["value"])
                    self._notify(wm)
                    return (
                        f"✅ Node {inputs['node_id']}: {inputs['param_name']} = {json.dumps(inputs['value'])}",
                        False,
                    )

                case "add_node":
                    nid = wm.add_node(
                        inputs["class_type"],
                        inputs.get("nickname"),
                        **(inputs.get("inputs") or {}),
                    )
                    self._notify(wm)
                    return f"✅ Added node {nid} ({inputs['class_type']})", False

                case "connect_nodes":
                    wm.connect(
                        str(inputs["src_node_id"]),
                        int(inputs.get("src_output_index", 0)),
                        str(inputs["dst_node_id"]),
                        inputs["dst_input_name"],
                    )
                    self._notify(wm)
                    return (
                        f"✅ Connected {inputs['src_node_id']}[{inputs.get('src_output_index', 0)}]"
                        f" → {inputs['dst_node_id']}.{inputs['dst_input_name']}"
                    ), False

                case "delete_node":
                    wm.delete_node(str(inputs["node_id"]))
                    self._notify(wm)
                    return f"✅ Deleted node {inputs['node_id']}", False

                case "add_lora_loader":
                    return self._add_lora(wm, inputs), False

                case "add_controlnet":
                    return self._add_controlnet(wm, inputs), False

                case "add_regional_attention":
                    return self._add_regional_attention(wm, inputs), False

                case "add_hires_fix":
                    return self._add_hires_fix(wm, inputs), False

                case "add_inpaint_pass":
                    return self._add_inpaint_pass(wm, inputs), False

                case "report_evolution_strategy":
                    print(
                        f"[ClawAgent] Strategy: {inputs['strategy']}\n"
                        f"            Top issue: {inputs['top_issue']}"
                    )
                    self._emit_event(
                        "strategy",
                        f"**Strategy:** {inputs['strategy']}\n**Top issue:** {inputs['top_issue']}",
                    )
                    return "Strategy noted.", False

                case "validate_workflow":
                    errs = WorkflowManager.validate_graph(wm.workflow)
                    if errs:
                        msg = "⚠️ Validation found issues:\n" + "\n".join(f"  • {e}" for e in errs)
                        msg += "\n\nFix these before calling finalize_workflow."
                        print(f"[ClawAgent] ⚠️ Validation: {len(errs)} issue(s)")
                        self._emit_event("validation", f"⚠️ {len(errs)} issue(s) found")
                        return msg, False
                    node_count = len(wm.workflow)
                    print(f"[ClawAgent] ✅ Validation passed ({node_count} nodes)")
                    self._emit_event("validation", f"✅ Valid ({node_count} nodes)")
                    return f"✅ Workflow is valid ({node_count} nodes, no issues found).", False

                case "finalize_workflow":
                    errs = WorkflowManager.validate_graph(wm.workflow)
                    if errs:
                        msg = (
                            "⚠️ Auto-validation found issues — fix before finalizing:\n"
                            + "\n".join(f"  • {e}" for e in errs)
                        )
                        print(f"[ClawAgent] ⚠️ Finalize blocked: {len(errs)} validation error(s)")
                        return msg, False
                    print(f"[ClawAgent] 🎯 {inputs.get('rationale', '')}")
                    return "Workflow finalized.", True

                case "read_skill":
                    return self._read_skill(inputs["skill_name"]), False

                case "explore_nodes":
                    return self._explore_nodes(), False

                case "transition_stage":
                    return self._transition_stage(inputs), False

                case _:
                    return f"❌ Unknown tool: {name}", False

        except Exception as exc:
            return f"❌ Tool error ({name}): {exc}", False

    # ------------------------------------------------------------------
    # Node exploration
    # ------------------------------------------------------------------

    def _explore_nodes(self) -> str:
        """Query ComfyUI /object_info and return a stage-classified summary."""
        from .skills.explore.scripts.explore_nodes import explore

        try:
            stage_map = explore(self.server_address)
        except Exception as exc:
            return f"❌ Exploration failed: {exc}"

        lines = [
            f"Discovered {stage_map['total_nodes_discovered']} node classes.",
            "",
        ]
        for name, data in stage_map["stages"].items():
            nc = data["node_count"]
            nodes_preview = ", ".join(data["node_classes"][:8])
            if nc > 8:
                nodes_preview += f", ... (+{nc - 8} more)"
            lines.append(f"**{name}** ({nc} nodes): {nodes_preview}")
            lines.append(f"  Tools: {', '.join(data['agent_tools'])}")

        if stage_map["unclassified_count"] > 0:
            lines.append(f"\nUnclassified: {stage_map['unclassified_count']} nodes")

        self._emit_event("info", f"Explored {stage_map['total_nodes_discovered']} nodes")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stage transitions
    # ------------------------------------------------------------------

    def _transition_stage(self, inputs: dict) -> str:
        """Advance the pipeline stage for stage-gated tool filtering."""
        target = inputs.get("target_stage", "")
        rationale = inputs.get("rationale", "")

        if not self.stage_router.enabled:
            return "Stage gating is disabled — all tools are always available."

        try:
            old_stage = self.stage_router.current_stage
            self.stage_router.transition_to(target)
            self._emit_event(
                "stage_transition",
                f"Stage: {old_stage} → {target}",
            )
            tools = self.stage_router.get_current_tool_names()
            return (
                f"✅ Transitioned from '{old_stage}' to '{target}'.\n"
                f"Available tools: {', '.join(tools)}\n"
                f"Rationale: {rationale}"
            )
        except ValueError as exc:
            return f"❌ Stage transition failed: {exc}"

    # ------------------------------------------------------------------
    # Skill reader (progressive disclosure — stage 2)
    # ------------------------------------------------------------------

    def _read_skill(self, skill_name: str) -> str:
        """
        Load the full body of a named skill on demand.

        This is the **stage-2** of progressive disclosure: the agent calls
        ``read_skill`` from within a tool-use loop after deciding which skill
        to apply.  The body text is returned as the tool result so the agent
        can follow the instructions immediately.
        """
        try:
            body = self.skill_manager.get_body(skill_name)
        except KeyError:
            available = ", ".join(self.skill_manager.skill_names)
            return f"❌ Skill {skill_name!r} not found. Available skills: {available or '(none)'}"
        print(f"[ClawAgent] 📖 read_skill: {skill_name}")
        return f"## Instructions for skill: {skill_name}\n\n{body}"

    # ------------------------------------------------------------------
    # Topology builders
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_clip_slot(wm: WorkflowManager, node_id: str) -> int:
        """Return the output slot index that carries CLIP for *node_id*.

        CheckpointLoaderSimple / CheckpointLoader emit CLIP on slot 1.
        LoraLoader emits CLIP on slot 1.
        Dedicated CLIPLoader / DualCLIPLoader emit on slot 0.
        Falls back to scanning existing "clip" consumers for evidence.
        """
        cls = wm.workflow.get(node_id, {}).get("class_type", "")
        if cls in ("CheckpointLoaderSimple", "CheckpointLoader", "LoraLoader"):
            return 1
        if cls in ("CLIPLoader", "DualCLIPLoader"):
            return 0
        # Heuristic: look at what existing nodes read as "clip" from this node
        for node in wm.workflow.values():
            for inp_name, inp_val in node.get("inputs", {}).items():
                if (
                    inp_name == "clip"
                    and isinstance(inp_val, list)
                    and len(inp_val) == 2
                    and str(inp_val[0]) == node_id
                ):
                    return int(inp_val[1])
        return 0

    def _add_lora(self, wm: WorkflowManager, inputs: dict) -> str:
        lora_name = inputs["lora_name"]
        model_nid = str(inputs["model_node_id"])
        clip_nid = str(inputs["clip_node_id"])
        sm = float(inputs.get("strength_model", 0.8))
        sc = float(inputs.get("strength_clip", 0.8))

        # Determine correct output slots for the source nodes.
        # CheckpointLoaderSimple: slot 0=MODEL, slot 1=CLIP, slot 2=VAE
        # LoraLoader / UNETLoader / CLIPLoader: slot 0 is the primary output
        model_slot = 0
        clip_slot = self._detect_clip_slot(wm, clip_nid)

        lora_nid = wm.add_node(
            "LoraLoader",
            f"LoRA: {lora_name[:30]}",
            model=[model_nid, model_slot],
            clip=[clip_nid, clip_slot],
            lora_name=lora_name,
            strength_model=sm,
            strength_clip=sc,
        )
        self._notify(wm)

        rewired_model, rewired_clip = [], []
        for nid, node in wm.workflow.items():
            if nid == lora_nid:
                continue
            for inp_name, inp_val in list(node.get("inputs", {}).items()):
                if isinstance(inp_val, list) and len(inp_val) == 2:
                    src_id, src_idx = str(inp_val[0]), inp_val[1]
                    if src_id == model_nid and src_idx == model_slot and inp_name == "model":
                        wm.workflow[nid]["inputs"][inp_name] = [lora_nid, 0]
                        rewired_model.append(f"{nid}.{inp_name}")
                    if src_id == clip_nid and src_idx == clip_slot and inp_name == "clip":
                        wm.workflow[nid]["inputs"][inp_name] = [lora_nid, 1]
                        rewired_clip.append(f"{nid}.{inp_name}")

        self._notify(wm)
        return (
            f"✅ LoraLoader {lora_nid} ({lora_name}, sm={sm}, sc={sc})\n"
            f"   Re-wired model: {rewired_model}\n"
            f"   Re-wired clip:  {rewired_clip}"
        )

    def _add_controlnet(self, wm: WorkflowManager, inputs: dict) -> str:
        cn_name = inputs["controlnet_name"]
        preproc_cls = inputs.get("preprocessor_class", "")
        image_nid = str(inputs.get("image_node_id", ""))
        pos_nid = str(inputs["positive_node_id"])
        neg_nid = str(inputs["negative_node_id"])
        strength = float(inputs.get("strength", 0.7))
        start_pct = float(inputs.get("start_percent", 0.0))
        end_pct = float(inputs.get("end_percent", 1.0))

        cn_loader_nid = wm.add_node(
            "ControlNetLoader", f"CN: {cn_name[:25]}", control_net_name=cn_name
        )
        self._notify(wm)

        preproc_output = None
        preproc_nid = None
        if preproc_cls and image_nid:
            preproc_nid = wm.add_node(preproc_cls, preproc_cls, image=[image_nid, 0])
            preproc_output = [preproc_nid, 0]
            self._notify(wm)
        elif image_nid:
            preproc_output = [image_nid, 0]

        apply_inputs: dict = {
            "positive": [pos_nid, 0],
            "negative": [neg_nid, 0],
            "control_net": [cn_loader_nid, 0],
            "strength": strength,
            "start_percent": start_pct,
            "end_percent": end_pct,
        }
        if preproc_output:
            apply_inputs["image"] = preproc_output

        apply_nid = wm.add_node("ControlNetApplyAdvanced", "CN Apply", **apply_inputs)
        self._notify(wm)

        for nid, node in wm.workflow.items():
            if node.get("class_type") == "KSampler":
                for inp_name, inp_val in list(node["inputs"].items()):
                    if isinstance(inp_val, list) and len(inp_val) == 2:
                        src = str(inp_val[0])
                        if src == pos_nid and inp_name == "positive":
                            wm.workflow[nid]["inputs"]["positive"] = [apply_nid, 0]
                        if src == neg_nid and inp_name == "negative":
                            wm.workflow[nid]["inputs"]["negative"] = [apply_nid, 1]

        self._notify(wm)
        parts = [
            "✅ ControlNet branch added:",
            f"   Loader:  {cn_loader_nid} ({cn_name})",
        ]
        if preproc_cls and image_nid:
            parts.append(f"   Preproc: {preproc_nid} ({preproc_cls})")
        parts.append(
            f"   Apply:   {apply_nid} (strength={strength}, {start_pct:.1f}→{end_pct:.1f})"
        )
        return "\n".join(parts)

    def _add_regional_attention(self, wm: WorkflowManager, inputs: dict) -> str:
        pos_nid = str(inputs["positive_node_id"])
        clip_nid = str(inputs["clip_node_id"])
        fg_prompt = inputs["foreground_prompt"]
        bg_prompt = inputs["background_prompt"]
        fg_weight = float(inputs.get("foreground_weight", 1.3))

        if pos_nid in wm.workflow:
            meta = wm.workflow[pos_nid].setdefault("_meta", {})
            meta["title"] = "Regional Prompt (BREAK)"
            wm.workflow[pos_nid]["inputs"]["text"] = f"{fg_prompt} BREAK {bg_prompt}"
            self._notify(wm)

        fg_nid = wm.add_node(
            "CLIPTextEncode", "Foreground Prompt", clip=[clip_nid, 0], text=fg_prompt
        )
        self._notify(wm)
        bg_nid = wm.add_node(
            "CLIPTextEncode", "Background Prompt", clip=[clip_nid, 0], text=bg_prompt
        )
        self._notify(wm)
        avg_nid = wm.add_node(
            "ConditioningAverage",
            "FG Weight",
            conditioning_to=[fg_nid, 0],
            conditioning_from=[pos_nid, 0],
            conditioning_to_strength=fg_weight,
        )
        self._notify(wm)
        combine_nid = wm.add_node(
            "ConditioningCombine",
            "Regional Combine",
            conditioning_1=[avg_nid, 0],
            conditioning_2=[bg_nid, 0],
        )
        self._notify(wm)

        for _nid, node in wm.workflow.items():
            if node.get("class_type") == "KSampler":
                node["inputs"]["positive"] = [combine_nid, 0]

        self._notify(wm)
        return (
            f"✅ Regional attention: FG={fg_nid}, BG={bg_nid}, Avg={avg_nid}, Combine={combine_nid}"
        )

    def _add_hires_fix(self, wm: WorkflowManager, inputs: dict) -> str:
        base_ks_nid = str(inputs["base_ksampler_node_id"])
        vae_nid = str(inputs["vae_node_id"])
        method = inputs.get("upscale_method", "bicubic")
        scale_by = float(inputs.get("scale_by", 1.5))
        hires_steps = int(inputs.get("hires_steps", 15))
        hires_denoise = float(inputs.get("hires_denoise", 0.45))
        save_nid = str(inputs.get("save_image_node_id", ""))

        existing_decode = next(
            (nid for nid, n in wm.workflow.items() if n.get("class_type") == "VAEDecode"),
            None,
        )
        if existing_decode:
            vae_connection = list(wm.workflow[existing_decode]["inputs"].get("vae", [vae_nid, 0]))
        else:
            vae_connection = [vae_nid, 0]

        upscale_nid = wm.add_node(
            "LatentUpscaleBy",
            "Hires Upscale",
            samples=[base_ks_nid, 0],
            upscale_method=method,
            scale_by=scale_by,
        )
        self._notify(wm)

        base_inputs = copy.deepcopy(wm.workflow.get(base_ks_nid, {}).get("inputs", {}))
        base_inputs["latent_image"] = [upscale_nid, 0]
        base_inputs["steps"] = hires_steps
        base_inputs["denoise"] = hires_denoise
        hires_ks_nid = wm.add_node("KSampler", "KSampler (Hires)", **base_inputs)
        self._notify(wm)

        decode_nid = wm.add_node(
            "VAEDecode", "VAEDecode (Hires)", samples=[hires_ks_nid, 0], vae=vae_connection
        )
        self._notify(wm)

        target = (
            save_nid
            if save_nid in wm.workflow
            else next(
                (nid for nid, n in wm.workflow.items() if n.get("class_type") == "SaveImage"), None
            )
        )
        if target:
            wm.workflow[target]["inputs"]["images"] = [decode_nid, 0]

        self._notify(wm)
        return (
            f"✅ Hires fix: Upscale={upscale_nid} (×{scale_by}), "
            f"KSampler={hires_ks_nid} ({hires_steps}st, d={hires_denoise}), Decode={decode_nid}"
        )

    def _add_inpaint_pass(self, wm: WorkflowManager, inputs: dict) -> str:
        region = inputs["region_description"]
        prompt_text = inputs["inpaint_prompt"]
        denoise = float(inputs.get("denoise_strength", 0.55))
        base_ks_nid = str(inputs["base_ksampler_node_id"])
        clip_nid = str(inputs["clip_node_id"])
        vae_nid = str(inputs["vae_node_id"])
        save_nid = str(inputs.get("save_image_node_id", ""))

        ip_pos_nid = wm.add_node(
            "CLIPTextEncode",
            f"Inpaint Prompt ({region[:20]})",
            clip=[clip_nid, 0],
            text=prompt_text,
        )
        self._notify(wm)

        base_inp = copy.deepcopy(wm.workflow.get(base_ks_nid, {}).get("inputs", {}))
        base_inp["positive"] = [ip_pos_nid, 0]
        base_inp["latent_image"] = [base_ks_nid, 0]
        base_inp["denoise"] = denoise

        existing_decode = next(
            (nid for nid, n in wm.workflow.items() if n.get("class_type") == "VAEDecode"),
            None,
        )
        vae_connection = (
            list(wm.workflow[existing_decode]["inputs"].get("vae", [vae_nid, 0]))
            if existing_decode
            else [vae_nid, 0]
        )

        ip_ks_nid = wm.add_node("KSampler", f"KSampler Inpaint ({region[:20]})", **base_inp)
        self._notify(wm)

        ip_decode_nid = wm.add_node(
            "VAEDecode", "VAEDecode Inpaint", samples=[ip_ks_nid, 0], vae=vae_connection
        )
        self._notify(wm)

        target = (
            save_nid
            if save_nid in wm.workflow
            else next(
                (nid for nid, n in wm.workflow.items() if n.get("class_type") == "SaveImage"), None
            )
        )
        if target:
            wm.workflow[target]["inputs"]["images"] = [ip_decode_nid, 0]

        self._notify(wm)
        return (
            f"✅ Inpaint pass [{region}]: Prompt={ip_pos_nid}, "
            f"KSampler={ip_ks_nid} (d={denoise}), Decode={ip_decode_nid}"
        )

    # ------------------------------------------------------------------
    # ComfyUI model queries
    # ------------------------------------------------------------------

    def _query_models(self, model_type: str) -> str:
        type_map = {
            "loras": ("LoraLoader", "lora_name"),
            "controlnets": ("ControlNetLoader", "control_net_name"),
            "checkpoints": ("CheckpointLoaderSimple", "ckpt_name"),
            "unets": ("UNETLoader", "unet_name"),
            "vae": ("VAELoader", "vae_name"),
            "upscale_models": ("UpscaleModelLoader", "model_name"),
            "clip_vision": ("CLIPVisionLoader", "clip_name"),
        }
        entry = type_map.get(model_type.lower())
        if not entry:
            return f"❌ Unknown model_type {model_type!r}. Valid: {list(type_map)}"
        node_class, param = entry
        try:
            url = f"http://{self.server_address}/object_info/{urllib.parse.quote(node_class)}"
            with urllib.request.urlopen(url, timeout=8) as r:
                data = json.loads(r.read())
            models = (
                data.get(node_class, {}).get("input", {}).get("required", {}).get(param, [[]])[0]
            )
            if not models:
                return f"No {model_type} found (ComfyUI returned empty list)."
            return f"Available {model_type} ({len(models)}):\n" + "\n".join(
                f"  - {m}" for m in models
            )
        except Exception as exc:
            return f"❌ Could not query {model_type}: {exc}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_messages(messages: list) -> list[dict]:
        """Convert a messages list to plain JSON-serializable dicts.

        LiteLLM appends ``ModelResponse.choices[0].message`` objects directly,
        which carry tool_calls as pydantic-like objects.  This method converts
        every message to a plain dict so the trace can be written to JSON/JSONL.
        """
        out: list[dict] = []
        for msg in messages:
            if isinstance(msg, dict):
                out.append(msg)
                continue
            # LiteLLM Message / ChatCompletionMessage objects
            d: dict = {"role": getattr(msg, "role", "assistant")}
            if getattr(msg, "content", None):
                d["content"] = msg.content
            tcs = getattr(msg, "tool_calls", None)
            if tcs:
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tcs
                ]
            out.append(d)
        return out

    def _notify(self, wm: WorkflowManager) -> None:
        if self.on_change:
            try:
                self.on_change(wm.to_dict())
            except Exception as exc:
                print(f"[ClawAgent] on_change error: {exc}", file=sys.stderr)

    def _emit_event(
        self,
        event_type: str,
        content: str,
        tool_name: str = "",
        tool_args: dict | None = None,
    ) -> None:
        """Fire an agent-thinking event if a listener is attached."""
        if self.on_agent_event:
            try:
                self.on_agent_event(event_type, content, tool_name, tool_args)
            except Exception:
                pass

    @staticmethod
    def _extract_structural_hints(feedback: str) -> str:
        """Parse verifier feedback for structural upgrade directives.

        Scans for fix_strategy keywords (inject_lora_*, add_controlnet_*,
        add_hires_fix, add_inpaint_pass) and returns a formatted list of
        required structural actions.  Returns empty string when only
        prompt-level fixes are suggested.
        """
        _STRUCTURAL_KEYWORDS = {
            "inject_lora_detail": '→ `read_skill("lora-enhancement")` then `add_lora_loader` (detail LoRA)',
            "inject_lora_style": '→ `read_skill("lora-enhancement")` then `add_lora_loader` (style LoRA)',
            "inject_lora_anatomy": '→ `read_skill("lora-enhancement")` then `add_lora_loader` (anatomy LoRA)',
            "inject_lora_lighting": '→ `read_skill("lora-enhancement")` then `add_lora_loader` (lighting LoRA)',
            "add_controlnet_canny": '→ `read_skill("controlnet-control")` then `add_controlnet` (canny edge)',
            "add_controlnet_depth": '→ `read_skill("controlnet-control")` then `add_controlnet` (depth map)',
            "add_controlnet_normal": '→ `read_skill("controlnet-control")` then `add_controlnet` (normal map)',
            "add_controlnet_pose": '→ `read_skill("controlnet-control")` then `add_controlnet` (pose)',
            "add_controlnet_tile": '→ `read_skill("controlnet-control")` then `add_controlnet` (tile)',
            "add_controlnet_seg": '→ `read_skill("controlnet-control")` then `add_controlnet` (segmentation)',
            "add_hires_fix": '→ `read_skill("hires-fix")` then `add_hires_fix`',
            "add_inpaint_pass": "→ `add_inpaint_pass` for the affected region",
            "add_ip_adapter": "→ consider IP-Adapter if available",
        }
        found = []
        feedback_lower = feedback.lower()
        for kw, action in _STRUCTURAL_KEYWORDS.items():
            if kw in feedback_lower:
                found.append(f"  • `{kw}` {action}")
        return "\n".join(found)

    def _build_user_message(
        self,
        original_prompt: str,
        workflow_manager: WorkflowManager | None,
        verifier_feedback: str | None,
        memory_summary: str | None,
        iteration: int,
    ) -> str:
        parts = [
            f"## Image Goal (user's original request)\n{original_prompt}",
            f"## Iteration\n{iteration}",
        ]

        # Expose the current positive prompt text so the agent can see exactly
        # what the baseline looks like and craft an improved version.
        if workflow_manager:
            current_positive: str | None = None
            for node in workflow_manager.workflow.values():
                if node.get("class_type") in ("CLIPTextEncode", "CLIPTextEncodeSDXL"):
                    text_val = node.get("inputs", {}).get("text") or node.get("inputs", {}).get(
                        "text_g"
                    )
                    if isinstance(text_val, str) and text_val.strip():
                        current_positive = text_val.strip()
                        break
            if current_positive:
                parts.append(
                    f"## Current Positive Prompt (baseline — needs refinement)\n{current_positive}\n\n"
                    "Use `set_prompt` to replace this with a detailed, high-quality version."
                )
            else:
                parts.append(
                    "## Current Positive Prompt\n(none — call `set_prompt` to craft one from the goal above)"
                )

        # Detect the active checkpoint / UNET model from the workflow and surface
        # it prominently so the agent can match it against model-specific skills
        # without having to call inspect_workflow first.
        active_model: str | None = None
        if workflow_manager:
            _loader_classes = ("CheckpointLoaderSimple", "CheckpointLoader", "UNETLoader")
            _model_params = ("ckpt_name", "unet_name")
            for node in workflow_manager.workflow.values():
                if node.get("class_type") in _loader_classes:
                    for param in _model_params:
                        val = node.get("inputs", {}).get(param)
                        if val and isinstance(val, str):
                            active_model = val
                            break
                if active_model:
                    break

        # Detect Qwen-Image pipeline — either via custom pipeline nodes (old plugin)
        # or via native ComfyUI UNETLoader with a qwen model filename.
        _qwen_plugin_classes = {"QwenImageModelLoader", "RH_QwenImageGenerator"}
        _qwen_unet_names = ("qwen_image",)

        def _node_is_qwen_unet(node: dict) -> bool:
            if node.get("class_type") != "UNETLoader":
                return False
            name = str(node.get("inputs", {}).get("unet_name", "")).lower()
            return any(kw in name for kw in _qwen_unet_names)

        is_qwen = workflow_manager is not None and any(
            node.get("class_type") in _qwen_plugin_classes or _node_is_qwen_unet(node)
            for node in workflow_manager.workflow.values()
        )

        if is_qwen:
            parts.append(
                "## Active Model\n`Qwen-Image-2512` (custom pipeline — no KSampler/ControlNet/LoRA)"
            )
        elif active_model:
            parts.append(f"## Active Model\n`{active_model}`")

        # Hint at relevant skills (names only — full instructions loaded via read_skill)
        # Also suggest model-specific skills based on the active checkpoint name.
        relevant = self.skill_manager.detect_relevant_skills(original_prompt)
        if workflow_manager and len(workflow_manager.workflow) == 0:
            if "workflow-builder" not in relevant:
                relevant.insert(0, "workflow-builder")
        if is_qwen:
            if "qwen-image-2512" not in relevant:
                relevant.insert(0, "qwen-image-2512")
        elif active_model:
            for skill_name in self.skill_manager.skill_names:
                # Simple substring match: skill name keywords appear in model filename
                skill_keywords = skill_name.replace("-", " ").split()
                model_lower = active_model.lower()
                if all(kw in model_lower for kw in skill_keywords if len(kw) > 2):
                    if skill_name not in relevant:
                        relevant.append(skill_name)
        # Pre-load model-specific skill(s) directly to avoid a read_skill
        # round-trip that the agent would otherwise make on every invocation.
        # Maps filename substrings → skill names for reliable detection.
        _MODEL_SKILL_MAP: list[tuple[list[str], str]] = [
            (["qwen_image"],                "qwen-image-2512"),
            (["longcat_image"],             "longcat-image"),
            (["longcat-image"],             "longcat-image"),
            (["z_image_turbo"],             "z-image-turbo"),
            (["z_image"],                   "z-image-turbo"),
            (["dreamshaper", "lcm"],        "dreamshaper8-lcm"),
        ]
        preloaded_skill_name: str | None = None

        if is_qwen:
            preloaded_skill_name = "qwen-image-2512"
        elif active_model:
            model_lower = active_model.lower()
            for keywords, skill_name in _MODEL_SKILL_MAP:
                if all(kw in model_lower for kw in keywords):
                    preloaded_skill_name = skill_name
                    break

        if preloaded_skill_name:
            try:
                preloaded_body = self.skill_manager.get_body(preloaded_skill_name)
            except KeyError:
                preloaded_body = None

            if preloaded_body:
                parts.append(
                    f"## Pre-loaded Skill: {preloaded_skill_name}\n"
                    "The following skill instructions are pre-loaded for your active model. "
                    "You do NOT need to call read_skill for this — apply these instructions directly.\n\n"
                    f"{preloaded_body}"
                )
                if preloaded_skill_name in relevant:
                    relevant.remove(preloaded_skill_name)

        if relevant:
            hint = ", ".join(sorted(relevant))
            parts.append(
                f"## Suggested Skills\nThese skills may be relevant: {hint}\n"
                "Call read_skill(<name>) to load full instructions before applying."
            )

        if verifier_feedback:
            if "[HUMAN]" in verifier_feedback or "[Human override]" in verifier_feedback:
                parts.append(
                    f"## Human Reviewer Feedback (previous iteration)\n{verifier_feedback}\n\n"
                    "The human reviewer has given you specific feedback about the image quality.\n"
                    "Prioritize their subjective preferences (style, mood, composition, color) "
                    "over structural or technical changes. Address their feedback directly."
                )
            else:
                structural_hints = self._extract_structural_hints(verifier_feedback)
                directive = (
                    "Use the evolution_suggestions and region_issues above to decide which "
                    "structural upgrade to apply this round."
                )
                if structural_hints:
                    directive = (
                        f"**REQUIRED structural upgrades** (from verifier fix_strategies):\n"
                        f"{structural_hints}\n\n"
                        "You MUST attempt these structural changes — do NOT fall back to "
                        "prompt-only refinement. Call `query_available_models` first to check "
                        "availability, then `read_skill` for the matching skill, then apply."
                    )
                parts.append(
                    f"## Verifier Feedback (previous iteration)\n{verifier_feedback}\n\n"
                    f"{directive}"
                )
        if memory_summary:
            parts.append(f"## Memory / Past Attempts\n{memory_summary}")

        if workflow_manager and len(workflow_manager.workflow) == 0:
            parts.append(
                "## CRITICAL — Empty Workflow — You MUST Build From Scratch\n"
                "The workflow is COMPLETELY EMPTY. You CANNOT finalize without adding nodes.\n\n"
                '**Step 1:** Call `read_skill("workflow-builder")` — it has complete node-by-node\n'
                "recipes for every architecture (SD 1.5, SDXL, Flux, Qwen, SD3, HunyuanDiT).\n\n"
                "**Step 2:** Call `query_available_models('checkpoints')` AND\n"
                "`query_available_models('diffusion_models')` to discover available models.\n\n"
                "**Step 3:** Match the model filename to an architecture using the patterns\n"
                "in the workflow-builder skill, then follow that recipe exactly.\n\n"
                "**Step 4:** Use ONLY exact filenames from query results — NEVER guess names.\n\n"
                "**Step 5:** Add nodes ONE AT A TIME using `add_node`. Set detailed prompts.\n\n"
                "**Step 6:** Call `finalize_workflow` only AFTER all nodes are added and connected."
            )

        parts.append(
            "Begin with report_evolution_strategy, then inspect_workflow, "
            "apply your changes, then finalize_workflow."
        )
        return "\n\n".join(parts)
