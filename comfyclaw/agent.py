"""
ClawAgent — Claude Sonnet agent that evolves ComfyUI workflow topology.

The agent uses Anthropic's tool-use API to call structured workflow
operations in a loop until it calls ``finalize_workflow``.

Tool categories
---------------
Inspection  : inspect_workflow, query_available_models
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
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable

import anthropic

from .skill_manager import SkillManager
from .workflow import WorkflowManager

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_BASE = """\
You are ComfyClaw, an expert ComfyUI workflow engineer.  Your job is to GROW
the workflow topology — not just tweak parameters — in response to the
verifier's region-level feedback.

Iteration strategy
------------------
1. Call report_evolution_strategy first: state your plan and the top issue.
2. Call inspect_workflow to see the current topology.
3. Call query_available_models BEFORE adding any LoRA or ControlNet node.
4. Apply structural upgrades (LoRA / ControlNet / regional / hires / inpaint).
5. Also tune prompt text, steps, CFG, and seed as needed.
6. Call finalize_workflow when done.

Decision heuristics
-------------------
  Flat / low-depth background          → add_controlnet (depth)
  Blurry edges / lost structure        → add_controlnet (canny)
  Wrong human pose / body              → add_controlnet (pose)
  Plasticky skin / poor texture        → add_lora_loader (detail/texture)
  Wrong anatomy (hands, fingers)       → add_lora_loader (anatomy) or inpaint
  Style inconsistency                  → add_lora_loader (style)
  Subject and background bleed         → add_regional_attention
  Low resolution / soft fine detail    → add_hires_fix
  Localised artifact in one area       → add_inpaint_pass

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


def _build_system_prompt(pinned_image_model: str | None) -> str:
    """Return the system prompt, injecting the pinned-model section when set."""
    if pinned_image_model:
        return _SYSTEM_PROMPT_BASE + _PINNED_MODEL_SECTION.format(
            model_name=pinned_image_model
        )
    return _SYSTEM_PROMPT_BASE

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_TOOLS: list[dict] = [
    {
        "name": "inspect_workflow",
        "description": "Return a human-readable summary of all nodes, IDs, class types, and key inputs.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "query_available_models",
        "description": (
            "Query the ComfyUI server for available models of a given type. "
            "Call this BEFORE adding LoRA or ControlNet nodes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_type": {
                    "type": "string",
                    "description": "loras | controlnets | checkpoints | unets | vae | upscale_models | clip_vision",
                }
            },
            "required": ["model_type"],
        },
    },
    {
        "name": "set_param",
        "description": "Set a scalar input on a specific node.",
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "param_name": {"type": "string"},
                "value": {},
            },
            "required": ["node_id", "param_name", "value"],
        },
    },
    {
        "name": "add_node",
        "description": "Append a new node to the workflow. Returns the new node ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "class_type": {"type": "string"},
                "nickname": {"type": "string"},
                "inputs": {"type": "object"},
            },
            "required": ["class_type"],
        },
    },
    {
        "name": "connect_nodes",
        "description": "Wire src_node output slot to dst_node input name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "src_node_id": {"type": "string"},
                "src_output_index": {"type": "integer"},
                "dst_node_id": {"type": "string"},
                "dst_input_name": {"type": "string"},
            },
            "required": ["src_node_id", "src_output_index", "dst_node_id", "dst_input_name"],
        },
    },
    {
        "name": "delete_node",
        "description": "Remove a node and clean up its connections.",
        "input_schema": {
            "type": "object",
            "properties": {"node_id": {"type": "string"}},
            "required": ["node_id"],
        },
    },
    {
        "name": "add_lora_loader",
        "description": (
            "Insert a LoraLoader between the model/clip source and all downstream consumers. "
            "Call query_available_models('loras') first."
        ),
        "input_schema": {
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
    },
    {
        "name": "add_controlnet",
        "description": (
            "Add ControlNetLoader + optional preprocessor + ControlNetApplyAdvanced. "
            "Call query_available_models('controlnets') first."
        ),
        "input_schema": {
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
            "required": ["controlnet_name", "preprocessor_class", "positive_node_id", "negative_node_id"],
        },
    },
    {
        "name": "add_regional_attention",
        "description": (
            "Split conditioning into foreground and background regional prompts using "
            "BREAK tokens + ConditioningCombine."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "positive_node_id": {"type": "string"},
                "clip_node_id": {"type": "string"},
                "foreground_prompt": {"type": "string"},
                "background_prompt": {"type": "string"},
                "foreground_weight": {"type": "number"},
            },
            "required": ["positive_node_id", "clip_node_id", "foreground_prompt", "background_prompt"],
        },
    },
    {
        "name": "add_hires_fix",
        "description": "Add a hires-fix pass: LatentUpscaleBy + second KSampler + VAEDecode.",
        "input_schema": {
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
    },
    {
        "name": "add_inpaint_pass",
        "description": "Add a targeted inpaint pass for a specific region.",
        "input_schema": {
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
                "region_description", "inpaint_prompt",
                "base_ksampler_node_id", "positive_node_id",
                "clip_node_id", "vae_node_id",
            ],
        },
    },
    {
        "name": "report_evolution_strategy",
        "description": "Declare your evolution plan BEFORE making changes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy": {"type": "string"},
                "top_issue": {"type": "string"},
            },
            "required": ["strategy", "top_issue"],
        },
    },
    {
        "name": "finalize_workflow",
        "description": "Signal that all modifications are complete.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rationale": {"type": "string"},
            },
            "required": ["rationale"],
        },
    },
]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ClawAgent:
    """
    Parameters
    ----------
    api_key         : Anthropic API key.
    model           : Claude model string.
    server_address  : ComfyUI HTTP address (used for model queries).
    skills_dir      : Path to skills/ folder; ``None`` uses built-in skills.
    on_change       : Called with the workflow dict after every mutation.
    max_tool_rounds : Safety cap on tool-call iterations.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5",
        server_address: str = "127.0.0.1:8188",
        skills_dir: str | None = None,
        on_change: Callable[[dict], None] | None = None,
        max_tool_rounds: int = 40,
        pinned_image_model: str | None = None,
    ) -> None:
        self.client = anthropic.Anthropic(
            api_key=api_key,
            base_url="https://api.anthropic.com",
        )
        self.model = model
        self.server_address = server_address
        self.skill_manager = SkillManager(skills_dir)
        self.on_change = on_change
        self.max_tool_rounds = max_tool_rounds
        self.pinned_image_model = pinned_image_model

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
        Drive the Claude tool-use loop to evolve the workflow.

        Returns the rationale string from the ``finalize_workflow`` call.
        """
        user_content = self._build_user_message(
            original_prompt=original_prompt,
            verifier_feedback=verifier_feedback,
            memory_summary=memory_summary,
            iteration=iteration,
        )
        messages: list[dict] = [{"role": "user", "content": user_content}]
        rationale = "(no rationale provided)"
        rounds = 0

        system_prompt = _build_system_prompt(self.pinned_image_model)

        while rounds < self.max_tool_rounds:
            rounds += 1
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                tools=_TOOLS,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason == "end_turn":
                break
            if resp.stop_reason != "tool_use":
                print(f"[ClawAgent] Unexpected stop_reason: {resp.stop_reason!r}", file=sys.stderr)
                break

            tool_results = []
            done = False
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                result_text, should_stop = self._dispatch(block.name, block.input, workflow_manager)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })
                if should_stop:
                    rationale = block.input.get("rationale", rationale)
                    done = True

            messages.append({"role": "user", "content": tool_results})
            if done:
                break

        return rationale

    # ------------------------------------------------------------------
    # Tool dispatcher
    # ------------------------------------------------------------------

    def _dispatch(
        self, name: str, inputs: dict, wm: WorkflowManager
    ) -> tuple[str, bool]:
        """Route a single tool call. Returns ``(result_text, should_stop)``."""
        try:
            match name:
                case "inspect_workflow":
                    return WorkflowManager.summarize(wm.workflow), False

                case "query_available_models":
                    return self._query_models(inputs["model_type"]), False

                case "set_param":
                    wm.set_param(str(inputs["node_id"]), inputs["param_name"], inputs["value"])
                    self._notify(wm)
                    return f"✅ Node {inputs['node_id']}: {inputs['param_name']} = {json.dumps(inputs['value'])}", False

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
                        f"✅ Connected {inputs['src_node_id']}[{inputs.get('src_output_index',0)}]"
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
                    return "Strategy noted.", False

                case "finalize_workflow":
                    print(f"[ClawAgent] 🎯 {inputs.get('rationale','')}")
                    return "Workflow finalized.", True

                case _:
                    return f"❌ Unknown tool: {name}", False

        except Exception as exc:
            return f"❌ Tool error ({name}): {exc}", False

    # ------------------------------------------------------------------
    # Topology builders
    # ------------------------------------------------------------------

    def _add_lora(self, wm: WorkflowManager, inputs: dict) -> str:
        lora_name = inputs["lora_name"]
        model_nid = str(inputs["model_node_id"])
        clip_nid = str(inputs["clip_node_id"])
        sm = float(inputs.get("strength_model", 0.8))
        sc = float(inputs.get("strength_clip", 0.8))

        lora_nid = wm.add_node(
            "LoraLoader",
            f"LoRA: {lora_name[:30]}",
            model=[model_nid, 0],
            clip=[clip_nid, 0],
            lora_name=lora_name,
            strength_model=sm,
            strength_clip=sc,
        )

        rewired_model, rewired_clip = [], []
        for nid, node in wm.workflow.items():
            if nid == lora_nid:
                continue
            for inp_name, inp_val in list(node.get("inputs", {}).items()):
                if isinstance(inp_val, list) and len(inp_val) == 2:
                    src_id, src_idx = str(inp_val[0]), inp_val[1]
                    if src_id == model_nid and src_idx == 0 and inp_name == "model":
                        wm.workflow[nid]["inputs"][inp_name] = [lora_nid, 0]
                        rewired_model.append(f"{nid}.{inp_name}")
                    if src_id == clip_nid and src_idx == 0 and inp_name == "clip":
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

        cn_loader_nid = wm.add_node("ControlNetLoader", f"CN: {cn_name[:25]}", control_net_name=cn_name)

        preproc_output = None
        if preproc_cls and image_nid:
            preproc_nid = wm.add_node(preproc_cls, preproc_cls, image=[image_nid, 0])
            preproc_output = [preproc_nid, 0]
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

        # Guard: update existing positive node's meta if it exists
        if pos_nid in wm.workflow:
            meta = wm.workflow[pos_nid].setdefault("_meta", {})
            meta["title"] = "Regional Prompt (BREAK)"
            wm.workflow[pos_nid]["inputs"]["text"] = f"{fg_prompt} BREAK {bg_prompt}"

        fg_nid = wm.add_node("CLIPTextEncode", "Foreground Prompt", clip=[clip_nid, 0], text=fg_prompt)
        bg_nid = wm.add_node("CLIPTextEncode", "Background Prompt", clip=[clip_nid, 0], text=bg_prompt)
        avg_nid = wm.add_node(
            "ConditioningAverage", "FG Weight",
            conditioning_to=[fg_nid, 0],
            conditioning_from=[pos_nid, 0],
            conditioning_to_strength=fg_weight,
        )
        combine_nid = wm.add_node(
            "ConditioningCombine", "Regional Combine",
            conditioning_1=[avg_nid, 0],
            conditioning_2=[bg_nid, 0],
        )
        for _nid, node in wm.workflow.items():
            if node.get("class_type") == "KSampler":
                node["inputs"]["positive"] = [combine_nid, 0]

        self._notify(wm)
        return (
            f"✅ Regional attention: FG={fg_nid}, BG={bg_nid}, "
            f"Avg={avg_nid}, Combine={combine_nid}"
        )

    def _add_hires_fix(self, wm: WorkflowManager, inputs: dict) -> str:
        base_ks_nid = str(inputs["base_ksampler_node_id"])
        vae_nid = str(inputs["vae_node_id"])
        method = inputs.get("upscale_method", "bicubic")
        scale_by = float(inputs.get("scale_by", 1.5))
        hires_steps = int(inputs.get("hires_steps", 15))
        hires_denoise = float(inputs.get("hires_denoise", 0.45))
        save_nid = str(inputs.get("save_image_node_id", ""))

        upscale_nid = wm.add_node(
            "LatentUpscaleBy", "Hires Upscale",
            samples=[base_ks_nid, 0],
            upscale_method=method,
            scale_by=scale_by,
        )
        base_inputs = copy.deepcopy(wm.workflow.get(base_ks_nid, {}).get("inputs", {}))
        base_inputs["latent_image"] = [upscale_nid, 0]
        base_inputs["steps"] = hires_steps
        base_inputs["denoise"] = hires_denoise
        hires_ks_nid = wm.add_node("KSampler", "KSampler (Hires)", **base_inputs)
        decode_nid = wm.add_node("VAEDecode", "VAEDecode (Hires)", samples=[hires_ks_nid, 0], vae=[vae_nid, 0])

        target = save_nid if save_nid in wm.workflow else next(
            (nid for nid, n in wm.workflow.items() if n.get("class_type") == "SaveImage"), None
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
        str(inputs["positive_node_id"])
        clip_nid = str(inputs["clip_node_id"])
        vae_nid = str(inputs["vae_node_id"])
        save_nid = str(inputs.get("save_image_node_id", ""))

        ip_pos_nid = wm.add_node(
            "CLIPTextEncode", f"Inpaint Prompt ({region[:20]})",
            clip=[clip_nid, 0], text=prompt_text,
        )
        base_inp = copy.deepcopy(wm.workflow.get(base_ks_nid, {}).get("inputs", {}))
        base_inp["positive"] = [ip_pos_nid, 0]
        base_inp["latent_image"] = [base_ks_nid, 0]
        base_inp["denoise"] = denoise
        ip_ks_nid = wm.add_node("KSampler", f"KSampler Inpaint ({region[:20]})", **base_inp)
        ip_decode_nid = wm.add_node("VAEDecode", "VAEDecode Inpaint", samples=[ip_ks_nid, 0], vae=[vae_nid, 0])

        target = save_nid if save_nid in wm.workflow else next(
            (nid for nid, n in wm.workflow.items() if n.get("class_type") == "SaveImage"), None
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
            "loras":          ("LoraLoader",             "lora_name"),
            "controlnets":    ("ControlNetLoader",       "control_net_name"),
            "checkpoints":    ("CheckpointLoaderSimple", "ckpt_name"),
            "unets":          ("UNETLoader",             "unet_name"),
            "vae":            ("VAELoader",              "vae_name"),
            "upscale_models": ("UpscaleModelLoader",     "model_name"),
            "clip_vision":    ("CLIPVisionLoader",       "clip_name"),
        }
        entry = type_map.get(model_type.lower())
        if not entry:
            return f"❌ Unknown model_type {model_type!r}. Valid: {list(type_map)}"
        node_class, param = entry
        try:
            url = (
                f"http://{self.server_address}/object_info/"
                f"{urllib.parse.quote(node_class)}"
            )
            with urllib.request.urlopen(url, timeout=8) as r:
                data = json.loads(r.read())
            models = (
                data.get(node_class, {})
                    .get("input", {})
                    .get("required", {})
                    .get(param, [[]])[0]
            )
            if not models:
                return f"No {model_type} found (ComfyUI returned empty list)."
            return f"Available {model_type} ({len(models)}):\n" + "\n".join(f"  - {m}" for m in models)
        except Exception as exc:
            return f"❌ Could not query {model_type}: {exc}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _notify(self, wm: WorkflowManager) -> None:
        if self.on_change:
            try:
                self.on_change(wm.to_dict())
            except Exception as exc:
                print(f"[ClawAgent] on_change error: {exc}", file=sys.stderr)

    def _build_user_message(
        self,
        original_prompt: str,
        verifier_feedback: str | None,
        memory_summary: str | None,
        iteration: int,
    ) -> str:
        parts = [f"## Image Prompt\n{original_prompt}", f"## Iteration\n{iteration}"]

        # Inject skill manifest + relevant instructions
        manifest = self.skill_manager.get_manifest()
        if manifest != "(no skills loaded)":
            parts.append(f"## Available Skills\n{manifest}")

        relevant_instructions = self.skill_manager.get_relevant_instructions(original_prompt)
        if relevant_instructions:
            parts.append(f"## Skill Instructions (relevant to this prompt)\n{relevant_instructions}")

        if verifier_feedback:
            parts.append(
                f"## Verifier Feedback (previous iteration)\n{verifier_feedback}\n\n"
                "Use the evolution_suggestions and region_issues above to decide which "
                "structural upgrade to apply this round."
            )
        if memory_summary:
            parts.append(f"## Memory / Past Attempts\n{memory_summary}")

        parts.append(
            "Begin with report_evolution_strategy, then inspect_workflow, "
            "apply your changes, then finalize_workflow."
        )
        return "\n\n".join(parts)
