#!/usr/bin/env python3
"""
Unified ComfyClaw benchmark runner.

Supports multiple image generation models and benchmark datasets via CLI flags.
All shared logic (harness setup, parallel execution, skill evolution, SFT trace
collection, resume, and summary reporting) lives here.

Usage:
    python experiments/run_benchmark.py --model longcat --benchmark geneval2 \\
        --n-prompts 800 --max-iterations 4 --evolve-batch-size 5 --parallel 2

    python experiments/run_benchmark.py --model qwen --benchmark dpg-bench

    python experiments/run_benchmark.py --model longcat --benchmark wise \\
        --prompt "Einstein's favorite musical instrument"
"""
import argparse
import copy
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(line_buffering=True)

try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]
    _env_path = REPO_ROOT / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("claw_bench")

from experiments.models import MODELS
from experiments.benchmarks import BENCHMARKS

# ── Config (env overridable) ──────────────────────────────────────────────
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4-5")
LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
LLM_API_BASE = os.environ.get("LLM_API_BASE", "")
SKILLS_DIR = str(REPO_ROOT / "comfyclaw" / "skills")

if LLM_API_BASE:
    os.environ.setdefault("OPENAI_API_BASE", LLM_API_BASE)

# ── LiteLLM retry/timeout config ─────────────────────────────────────────
import litellm
litellm.num_retries = int(os.environ.get("LLM_NUM_RETRIES", "3"))
litellm.request_timeout = int(os.environ.get("LLM_REQUEST_TIMEOUT", "300"))
litellm.retry_after = 5  # seconds between retries (with jitter)
litellm.drop_params = True  # silently drop unsupported params per provider


def _agent_slug(model: str) -> str:
    """Derive a short slug from the LLM model name for folder naming."""
    name = model.rsplit("/", 1)[-1]
    name = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return name


def _parse_comfyui_addrs() -> list[str]:
    """Parse ComfyUI server addresses from env vars.

    Checks COMFYUI_ADDRS (comma-separated) first, then falls back to
    the single-address COMFYUI_ADDR for backwards compatibility.
    """
    multi = os.environ.get("COMFYUI_ADDRS", "")
    if multi:
        return [a.strip() for a in multi.split(",") if a.strip()]
    return [os.environ.get("COMFYUI_ADDR", "127.0.0.1:8188")]


COMFYUI_ADDRS: list[str] = _parse_comfyui_addrs()


def _prepare_baseline_skills_dir(model_skill: str) -> str:
    """Create a minimal skills directory containing only the model-specific skill.

    Copies the single skill folder so the agent has access to model-specific
    configuration (e.g. sampler params, architecture constraints) but nothing else.
    Returns the path to the temporary skills directory.
    """
    import shutil

    src_skills = REPO_ROOT / "comfyclaw" / "skills"
    baseline_dir = REPO_ROOT / "comfyclaw" / "_baseline_skills"
    if baseline_dir.exists():
        shutil.rmtree(baseline_dir)
    baseline_dir.mkdir(parents=True)

    src_skill = src_skills / model_skill
    if src_skill.is_dir():
        shutil.copytree(src_skill, baseline_dir / model_skill)
        log.info("Baseline skills: copied %s → %s", model_skill, baseline_dir / model_skill)
    else:
        log.warning("Model skill %r not found at %s — baseline will have no skills", model_skill, src_skill)

    return str(baseline_dir)


def _slug(text: str, max_len: int = 50) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:max_len]


# ── Paths derived from model + benchmark ──────────────────────────────────

_DEFAULT_EXPERIMENTS_ROOT = str(REPO_ROOT.parent / "comfy_agent_experiments_output")
_DEFAULT_BASELINE_ROOT = str(REPO_ROOT.parent / "comfy_agent_baseline_experiments_output")
EXPERIMENTS_ROOT: Path = Path(os.environ.get("EXPERIMENTS_ROOT", _DEFAULT_EXPERIMENTS_ROOT))


def _build_paths(model_short: str, bench_short: str, agent_name: str = "") -> dict:
    # Single source of truth for the ``<model>_<bench>[/<agent>]`` convention
    # lives in ``comfyclaw.skill_manager.evolved_dir_for``.  Keep both this
    # helper and ``SkillManager(model=..., benchmark=...)`` in sync by
    # routing through it.
    from comfyclaw.skill_manager import evolved_dir_for

    if agent_name:
        experiment_dir = EXPERIMENTS_ROOT / f"{model_short}_{bench_short}" / agent_name
    else:
        experiment_dir = EXPERIMENTS_ROOT / f"{model_short}_{bench_short}"
    output_dir = os.environ.get("OUTPUT_DIR", str(experiment_dir / "results"))
    detailed_dir = os.environ.get("DETAILED_DIR", str(experiment_dir / "detailed"))
    evolved_dir = str(evolved_dir_for(model_short, bench_short, agent_name or None))
    return {
        "output_dir": output_dir,
        "detailed_dir": detailed_dir,
        "evolved_skills_dir": evolved_dir,
        "learned_skills_dir": os.path.join(evolved_dir, "learned-errors"),
    }


# ── Helpers ───────────────────────────────────────────────────────────────

def load_results(results_path: str) -> list[dict]:
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return []


def save_results(results: list[dict], results_path: str):
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def _write_evolution_metadata(evolved_dir: str, **kwargs: object) -> None:
    meta_path = os.path.join(evolved_dir, "evolution_metadata.json")
    os.makedirs(evolved_dir, exist_ok=True)
    meta: dict = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    meta.update(kwargs)
    meta["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def _collect_error_data(harness) -> list[dict]:
    errors = []
    for entry in harness.evolution_log.entries:
        for repair in entry.repair_history:
            if repair["error"]:
                errors.append({
                    "iteration": entry.iteration,
                    "phase": repair["phase"],
                    "error": repair["error"][:500],
                    "outcome": repair["outcome"],
                    "rationale": entry.rationale[:300] if entry.rationale else "",
                })
    for attempt in harness.memory.attempts:
        if attempt.verifier_score == 0.0:
            for f in attempt.failed:
                if f.startswith("Execution error:"):
                    errors.append({
                        "iteration": attempt.iteration,
                        "phase": "execution",
                        "error": f[:500],
                        "outcome": "failed",
                        "rationale": attempt.experience[:300],
                    })
    return errors


def synthesize_learned_skill(new_errors: list[dict], learned_dir: str) -> bool:
    import litellm

    skill_dir = Path(learned_dir)
    skill_path = skill_dir / "SKILL.md"

    existing_body = ""
    if skill_path.exists():
        existing_body = skill_path.read_text(encoding="utf-8")

    error_summary = "\n".join(
        f"- [{e['phase']}] iter {e['iteration']}, outcome={e['outcome']}: {e['error'][:300]}"
        for e in new_errors
    )

    prompt_text = (
        "You are maintaining a ComfyUI workflow troubleshooting skill for an AI agent.\n"
        "The agent builds and modifies ComfyUI node-graph workflows. It frequently makes "
        "wiring and configuration errors that ComfyUI rejects. Your job is to write a concise "
        "SKILL.md that teaches the agent to avoid these errors.\n\n"
    )

    if existing_body:
        prompt_text += (
            "Here is the EXISTING skill content:\n"
            f"```\n{existing_body}\n```\n\n"
            "NEW errors encountered since last update:\n"
            f"{error_summary}\n\n"
            "Update the skill to incorporate lessons from the new errors. "
            "Merge with existing content — don't lose previously learned lessons. "
            "Remove duplicates. Keep it concise and actionable.\n\n"
        )
    else:
        prompt_text += (
            "Errors encountered during workflow generation:\n"
            f"{error_summary}\n\n"
            "Create a NEW skill from these errors.\n\n"
        )

    from comfyclaw.evolve import DESCRIPTION_WRITING_GUIDE

    prompt_text += (
        "Output ONLY the complete SKILL.md content. It MUST follow this exact format:\n"
        "1. Start with YAML frontmatter between --- delimiters\n"
        "2. The 'name' field MUST be exactly 'learned-errors'\n"
        "3. The 'description' field is CRITICAL — follow the guide below.\n"
        "4. Include 'tags: [agent]' in the frontmatter (required for agent discovery)\n"
        "5. After the frontmatter, write clear Markdown instructions\n\n"
        f"{DESCRIPTION_WRITING_GUIDE}\n"
        "Cite at least one of the concrete error strings above in the WHEN clause\n"
        "so the agent trivially recognises when to consult this skill.\n\n"
        "Focus on:\n"
        "- ComfyUI node output slot indices (which slot is MODEL vs CLIP vs VAE)\n"
        "- Common wiring mistakes and how to avoid them\n"
        "- Parameter type constraints\n"
        "- Patterns that cause 'string index out of range' or similar runtime errors\n\n"
        "Keep it under 150 lines. Be specific with slot numbers and node class names."
    )

    try:
        resp = litellm.completion(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            **({"api_base": LLM_API_BASE} if LLM_API_BASE else {}),
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=2000,
        )
        skill_content = resp.choices[0].message.content.strip()

        if skill_content.startswith("```"):
            skill_content = re.sub(r"^```\w*\n?", "", skill_content)
            skill_content = re.sub(r"\n?```$", "", skill_content)

        if not skill_content.startswith("---"):
            log.warning("  Synthesized skill missing frontmatter, skipping")
            return False

        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(skill_content, encoding="utf-8")
        log.info("  Wrote learned skill to %s (%d lines)",
                 skill_path, skill_content.count("\n") + 1)
        return True

    except Exception as exc:
        log.warning("  Failed to synthesize learned skill: %s", exc)
        return False


_TRIVIAL_TOOLS = frozenset({
    "set_prompt", "set_param", "finalize_workflow",
    "report_evolution_strategy", "inspect_workflow",
    "validate_workflow",
})


def _collect_success_data(result: dict, prompt_dir: str) -> dict | None:
    """Extract tool-call sequences and strategy data from a high-scoring prompt.

    Returns None if the result used only trivial tools (set_prompt/set_param).
    Captures full arguments for non-trivial tools (LoRA, etc.)
    so the success synthesiser can learn specific configurations.
    """
    details_path = os.path.join(prompt_dir, "details.json")
    sft_path = os.path.join(prompt_dir, "sft_traces.jsonl")

    tool_sequence: list[str] = []
    tool_details: list[dict] = []
    best_rationale = ""

    if os.path.exists(sft_path):
        best_score = -1.0
        best_tools: list[str] = []
        best_details: list[dict] = []
        with open(sft_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    trace = json.loads(line)
                except json.JSONDecodeError:
                    continue
                vs = trace.get("verifier_score", 0) or 0
                tools_in_trace: list[str] = []
                details_in_trace: list[dict] = []
                for msg in trace.get("messages", []):
                    if msg.get("role") == "assistant" and msg.get("tool_calls"):
                        for tc in msg["tool_calls"]:
                            fn = tc.get("function", {}).get("name", "")
                            if not fn:
                                continue
                            tools_in_trace.append(fn)
                            if fn not in _TRIVIAL_TOOLS:
                                raw_args = tc["function"].get("arguments", "")
                                try:
                                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                                except (json.JSONDecodeError, TypeError):
                                    args = {"_raw": str(raw_args)[:200]}
                                details_in_trace.append({
                                    "tool": fn,
                                    "args": {
                                        k: (v if len(str(v)) < 200 else str(v)[:200] + "...")
                                        for k, v in (args if isinstance(args, dict) else {}).items()
                                    },
                                })
                if vs > best_score:
                    best_score = vs
                    best_tools = tools_in_trace
                    best_details = details_in_trace
            tool_sequence = best_tools
            tool_details = best_details

    if os.path.exists(details_path):
        try:
            with open(details_path) as f:
                details = json.load(f)
            best_iter = details.get("best_iteration")
            if best_iter is not None:
                for it in details.get("iterations", []):
                    if it.get("iteration") == best_iter:
                        best_rationale = it.get("agent_rationale") or ""
                        break
        except (json.JSONDecodeError, KeyError):
            pass

    non_trivial = [t for t in tool_sequence if t not in _TRIVIAL_TOOLS]
    if not non_trivial:
        return None

    return {
        "prompt": result.get("prompt", ""),
        "best_score": result.get("best_score", 0),
        "passed": result.get("passed", []),
        "tool_sequence": tool_sequence,
        "non_trivial_tools": non_trivial,
        "tool_details": tool_details,
        "rationale": best_rationale,
        "node_count": result.get("node_count", 0),
    }


def synthesize_success_patterns(
    success_entries: list[dict], learned_dir: str
) -> bool:
    """Synthesize a 'learned-successes' skill from high-scoring prompt data.

    Parallel to ``synthesize_learned_skill`` but learns from wins, not errors.
    """
    import litellm

    skill_dir = Path(learned_dir).parent / "learned-successes"
    skill_path = skill_dir / "SKILL.md"

    existing_body = ""
    if skill_path.exists():
        existing_body = skill_path.read_text(encoding="utf-8")

    parts: list[str] = []
    for e in success_entries:
        line = (
            f"- Prompt: {e['prompt'][:80]} | Score: {e['best_score']:.2f} | "
            f"Key tools: {', '.join(e['non_trivial_tools'][:5])}"
        )
        if e.get("rationale"):
            line += f" | Rationale: {e['rationale'][:150]}"
        td = e.get("tool_details", [])
        if td:
            detail_strs = []
            for d in td[:5]:
                args_snippet = json.dumps(d["args"], default=str)
                if len(args_snippet) > 150:
                    args_snippet = args_snippet[:150] + "..."
                detail_strs.append(f"  {d['tool']}({args_snippet})")
            line += "\n" + "\n".join(detail_strs)
        parts.append(line)
    success_summary = "\n".join(parts)

    prompt_text = (
        "You are maintaining a ComfyUI workflow best-practices skill for an AI agent.\n"
        "The agent builds and modifies ComfyUI node-graph workflows. Your job is to write a "
        "concise SKILL.md that teaches the agent to replicate successful strategies.\n\n"
    )

    if existing_body:
        prompt_text += (
            "Here is the EXISTING skill content:\n"
            f"```\n{existing_body}\n```\n\n"
            "NEW successful patterns observed since last update:\n"
            f"{success_summary}\n\n"
            "Update the skill to incorporate these new successful patterns. "
            "Merge with existing content — don't lose previously learned patterns. "
            "Remove duplicates. Keep it concise and actionable.\n\n"
        )
    else:
        prompt_text += (
            "Successful strategies observed during workflow generation:\n"
            f"{success_summary}\n\n"
            "Create a NEW skill from these successful patterns.\n\n"
        )

    from comfyclaw.evolve import DESCRIPTION_WRITING_GUIDE

    prompt_text += (
        "Output ONLY the complete SKILL.md content. It MUST follow this exact format:\n"
        "1. Start with YAML frontmatter between --- delimiters\n"
        "2. The 'name' field MUST be exactly 'learned-successes'\n"
        "3. The 'description' field is CRITICAL — follow the guide below.\n"
        "4. Include 'tags: [agent]' in the frontmatter (required for agent discovery)\n"
        "5. After the frontmatter, write clear Markdown instructions\n\n"
        f"{DESCRIPTION_WRITING_GUIDE}\n"
        "Cite at least 2 concrete trigger signals from the successful prompts above\n"
        "(number words, object-type combinations, prompt phrasings, etc.) so the\n"
        "agent recognises a matching prompt at plan time and calls read_skill.\n\n"
        "Focus on:\n"
        "- Which tools and node configurations produced the best results\n"
        "- Effective parameter values (guidance, steps, sampler settings)\n"
        "- LoRA usage: which LoRA checkpoints worked, at what strength, for which prompt types\n"
        "- Successful prompt engineering patterns\n"
        "- Workflow topology patterns that consistently improve scores\n"
        "- When to use specific techniques (e.g. regional attention for multi-object scenes)\n\n"
        "Keep it under 150 lines. Be specific with tool names, node classes, parameter values, "
        "and LoRA checkpoint names."
    )

    try:
        resp = litellm.completion(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            **({"api_base": LLM_API_BASE} if LLM_API_BASE else {}),
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=2000,
        )
        skill_content = resp.choices[0].message.content.strip()

        if skill_content.startswith("```"):
            skill_content = re.sub(r"^```\w*\n?", "", skill_content)
            skill_content = re.sub(r"\n?```$", "", skill_content)

        if not skill_content.startswith("---"):
            log.warning("  Synthesized success skill missing frontmatter, skipping")
            return False

        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(skill_content, encoding="utf-8")
        log.info("  Wrote learned-successes skill to %s (%d lines)",
                 skill_path, skill_content.count("\n") + 1)
        return True

    except Exception as exc:
        log.warning("  Failed to synthesize success skill: %s", exc)
        return False


# ── Core: run a single prompt ─────────────────────────────────────────────

def run_one(
    prompt: str,
    idx: int,
    base_workflow: dict,
    model_config: dict,
    paths: dict,
    max_iterations: int,
    warm_start: bool,
    server_address: str = "127.0.0.1:8188",
    is_baseline: bool = False,
    skills_dir_override: str | None = None,
    benchmark_short: str | None = None,
    agent_name: str | None = None,
) -> dict:
    from comfyclaw.harness import ClawHarness, HarnessConfig

    t0 = time.time()
    # Pass image_model_short / benchmark / agent_name so HarnessConfig can
    # keep the (explicit path) and (auto-derived path) in sync via
    # ``evolved_dir_for`` — and so the agent gets the right "model:<short>"
    # include-tags for skill filtering.  For baseline runs we pass ""
    # (empty string) to explicitly DISABLE evolved-skill loading.
    cfg = HarnessConfig(
        api_key=LLM_API_KEY,
        server_address=server_address,
        model=LLM_MODEL,
        max_iterations=max_iterations,
        success_threshold=0.95,
        sync_port=0,
        image_model=None,
        stage_gated=not is_baseline,
        skills_dir=skills_dir_override or SKILLS_DIR,
        evolved_skills_dir="" if is_baseline else paths["evolved_skills_dir"],
        image_model_short=model_config["short_name"],
        benchmark=benchmark_short,
        agent_name=agent_name,
        max_nodes=20,
        baseline_first=warm_start and not is_baseline,
        max_images=max_iterations + 2,
        verifier_mode="none" if is_baseline else "vlm",
    )

    init_wf = copy.deepcopy(base_workflow) if warm_start else {}
    harness = ClawHarness.from_workflow_dict(init_wf, cfg)
    with harness:
        image_bytes = harness.run(prompt)

    best_score = 0.0
    baseline_score = 0.0
    for entry in harness.evolution_log.entries:
        if entry.verifier_score is not None:
            best_score = max(best_score, entry.verifier_score)
    for attempt in harness.memory.attempts:
        if attempt.verifier_score > best_score:
            best_score = attempt.verifier_score
        if attempt.iteration == 0:
            baseline_score = attempt.verifier_score

    passed_all, failed_all = [], []
    for attempt in harness.memory.attempts:
        if attempt.verifier_score == best_score:
            passed_all = attempt.passed
            failed_all = attempt.failed
            break

    img_path = None
    if image_bytes:
        img_path = os.path.join(paths["output_dir"], "images",
                                f"{model_config['short_name']}_{idx:03d}.png")
        with open(img_path, "wb") as f:
            f.write(image_bytes)

    elapsed = time.time() - t0
    node_count = (
        harness.evolution_log.entries[-1].node_count_after
        if harness.evolution_log.entries else model_config["default_node_count"]
    )

    # ── Per-prompt detailed results ───────────────────────────────────
    prompt_dir = os.path.join(paths["detailed_dir"], f"prompt_{idx:03d}_{_slug(prompt)}")
    os.makedirs(prompt_dir, exist_ok=True)

    evo_entries_by_iter = {e.iteration: e for e in harness.evolution_log.entries}

    iterations_detail = []
    for attempt in harness.memory.attempts:
        label = "baseline" if attempt.iteration == 0 else f"iter_{attempt.iteration}"
        img_filename = f"{label}.png"

        if attempt.image_bytes:
            with open(os.path.join(prompt_dir, img_filename), "wb") as f:
                f.write(attempt.image_bytes)

        evo = evo_entries_by_iter.get(attempt.iteration)
        is_error = (
            attempt.verifier_score == 0.0
            and any(f.startswith("Execution error:") for f in attempt.failed)
        )
        iterations_detail.append({
            "iteration": attempt.iteration,
            "label": label,
            "status": "error" if is_error else "success",
            "image_file": img_filename if attempt.image_bytes else None,
            "verifier_score": attempt.verifier_score,
            "passed": attempt.passed,
            "failed": attempt.failed,
            "experience": attempt.experience,
            "workflow_snapshot": attempt.workflow_snapshot,
            "agent_rationale": evo.rationale if evo else None,
            "node_count_before": evo.node_count_before if evo else None,
            "node_count_after": evo.node_count_after if evo else None,
            "nodes_added": evo.node_ids_added if evo else None,
            "repair_history": evo.repair_history if evo else [],
            "timing": {
                "agent_s": evo.agent_time_s,
                "generation_s": evo.generation_time_s,
                "verify_s": evo.verify_time_s,
                "repair_s": evo.repair_time_s,
            } if evo else None,
        })

    evo_entries = harness.evolution_log.entries
    timing_summary = {
        "total_s": round(elapsed, 1),
        "agent_s": round(sum(e.agent_time_s for e in evo_entries), 2),
        "generation_s": round(sum(e.generation_time_s for e in evo_entries), 2),
        "verify_s": round(sum(e.verify_time_s for e in evo_entries), 2),
        "repair_s": round(sum(e.repair_time_s for e in evo_entries), 2),
    }

    detail_json = {
        "idx": idx,
        "prompt": prompt,
        "baseline_score": baseline_score,
        "best_score": best_score,
        "best_iteration": max(
            harness.memory.attempts,
            key=lambda a: a.verifier_score,
        ).iteration if harness.memory.attempts else None,
        "total_iterations": len(evo_entries),
        "elapsed_s": round(elapsed, 1),
        "timing": timing_summary,
        "final_node_count": node_count,
        "config": {
            "model": model_config["name"],
            "llm_model": LLM_MODEL,
            "max_iterations": max_iterations,
            "warm_start": warm_start,
            "stage_gated": True,
        },
        "iterations": iterations_detail,
    }
    with open(os.path.join(prompt_dir, "details.json"), "w") as f:
        json.dump(detail_json, f, indent=2, default=str)

    log.info("  Detailed results saved to %s", prompt_dir)

    error_data = _collect_error_data(harness)

    # ── SFT traces ────────────────────────────────────────────────────
    sft_traces = harness.sft_traces
    if sft_traces:
        sft_path = os.path.join(prompt_dir, "sft_traces.jsonl")
        with open(sft_path, "w", encoding="utf-8") as f:
            for trace in sft_traces:
                f.write(json.dumps(trace, default=str, ensure_ascii=False) + "\n")
        log.info("  Saved %d SFT trace(s) to %s", len(sft_traces), sft_path)

    return {
        "idx": idx,
        "prompt": prompt,
        "baseline_score": baseline_score,
        "best_score": best_score,
        "passed": passed_all,
        "failed": failed_all,
        "elapsed_s": round(elapsed, 1),
        "node_count": node_count,
        "iterations": len(evo_entries),
        "image_path": img_path,
        "error_data": error_data,
        "sft_trace_count": len(sft_traces),
        "skills_read": list(harness.skills_read),
    }


# ── Batch skill evolution ─────────────────────────────────────────────────

def run_batch_evolution(
    results: list[dict],
    cycle: int,
    evolved_dir: str,
    model_short: str,
    benchmark_short: str,
    agent_name: str | None = None,
) -> None:
    from comfyclaw.evolve import SkillEvolver

    auto_tags = [f"model:{model_short}", f"bench:{benchmark_short}"]
    if agent_name:
        auto_tags.append(f"agent:{agent_name}")

    evolver = SkillEvolver(
        evolved_skills_dir=evolved_dir,
        llm_model=LLM_MODEL,
        api_key=LLM_API_KEY,
        min_improvement=0.02,
        max_mutations_per_cycle=3,
        auto_tags=auto_tags,
    )
    report = evolver.run_cycle(results, cycle=cycle)
    log.info("  Evolution cycle %d: %s", cycle, report.summary())
    if report.mutations_accepted > 0:
        log.info("  %d mutation(s) accepted — skills updated for next batch",
                 report.mutations_accepted)
    else:
        log.info("  No mutations accepted this cycle")

    report_path = os.path.join(evolved_dir, "evolution_reports.jsonl")
    try:
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "cycle": report.cycle,
                "pre_mean_score": report.pre_mean_score,
                "post_mean_score": report.post_mean_score,
                "mutations_proposed": report.mutations_proposed,
                "mutations_accepted": report.mutations_accepted,
                "mutations_rejected": report.mutations_rejected,
                "n_failure_clusters": len(report.failure_clusters),
                "n_success_clusters": len(report.success_clusters),
                "reinforce_mutations": report.reinforce_mutations,
                "duration_s": report.duration_s,
                "accepted_skills": [
                    {"type": m.mutation_type, "targets": m.target_skills}
                    for m in report.mutations if m.accepted
                ],
            }) + "\n")
    except Exception as exc:
        log.warning("Failed to persist evolution report: %s", exc)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified ComfyClaw benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Models:     " + ", ".join(MODELS.keys()) + "\n"
            "Benchmarks: " + ", ".join(BENCHMARKS.keys())
        ),
    )
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()),
                        help="Image generation model")
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARKS.keys()),
                        help="Benchmark dataset")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Run a single custom prompt instead of the benchmark set")
    parser.add_argument("--n-prompts", type=int, default=None,
                        help="Number of prompts (default: benchmark's full set)")
    parser.add_argument("--max-iterations", type=int, default=4,
                        help="Max agent iterations per prompt (default: 4)")
    parser.add_argument("--evolve-batch-size", type=int, default=0,
                        help="Evolve skills every N prompts (0 = disabled)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Run N prompts concurrently (default: 1)")
    parser.add_argument("--no-warm-start", action="store_true",
                        help="Start from empty workflow instead of model's base workflow")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Override path to benchmark data file/directory")
    parser.add_argument("--comfyui-addrs", type=str, default=None,
                        help="Comma-separated ComfyUI addresses (e.g. 127.0.0.1:8188,127.0.0.1:8189). "
                             "Also settable via COMFYUI_ADDRS env var. Defaults to single instance on :8188")
    parser.add_argument("--agent-name", type=str, default=None,
                        help="Agent/LLM name for organizing output folders (e.g. 'gpt-5.4'). "
                             "Auto-derived from LLM_MODEL if not set. Results go to "
                             "{model}_{benchmark}/{agent_name}/")
    parser.add_argument("--baseline", action="store_true",
                        help="Baseline mode: 1 iteration, no verifier, only model-specific "
                             "skill. The agent builds/tunes the workflow once with no feedback loop.")
    args = parser.parse_args()

    model_config = MODELS[args.model]
    bench_config = BENCHMARKS[args.benchmark]
    is_baseline = args.baseline
    if is_baseline and "EXPERIMENTS_ROOT" not in os.environ:
        global EXPERIMENTS_ROOT
        EXPERIMENTS_ROOT = Path(_DEFAULT_BASELINE_ROOT)
    agent_name = args.agent_name or ("baseline" if is_baseline else _agent_slug(LLM_MODEL))
    paths = _build_paths(model_config["short_name"], bench_config["short_name"], agent_name)

    n_prompts = args.n_prompts or int(os.environ.get("N_PROMPTS", bench_config["default_n_prompts"]))
    max_iterations = 1 if is_baseline else args.max_iterations
    evolve_batch_size = 0 if is_baseline else args.evolve_batch_size
    parallel_size = args.parallel
    warm_start = not args.no_warm_start

    if is_baseline:
        model_skill = model_config.get("model_skill", "")
        if model_skill:
            SKILLS_DIR_OVERRIDE = _prepare_baseline_skills_dir(model_skill)
        else:
            log.warning("No model_skill defined for %s — baseline will use no skills",
                        model_config["short_name"])
            SKILLS_DIR_OVERRIDE = str(REPO_ROOT / "comfyclaw" / "_baseline_skills")
            os.makedirs(SKILLS_DIR_OVERRIDE, exist_ok=True)
    else:
        SKILLS_DIR_OVERRIDE = None

    if args.comfyui_addrs:
        comfyui_addrs = [a.strip() for a in args.comfyui_addrs.split(",") if a.strip()]
    else:
        comfyui_addrs = COMFYUI_ADDRS

    os.makedirs(os.path.join(paths["output_dir"], "images"), exist_ok=True)
    os.makedirs(paths["detailed_dir"], exist_ok=True)

    results_path = os.path.join(paths["output_dir"], "results.json")

    # ── Load prompts ──────────────────────────────────────────────────
    if args.prompt:
        n_prompts = 1
        prompts = [{"prompt": args.prompt, "idx": 0, "meta": {}}]
        log.info("=" * 70)
        log.info("ComfyClaw + %s — custom prompt", model_config["name"])
        log.info("Prompt: %s", args.prompt)
    else:
        prompts = bench_config["load_prompts"](n_prompts, args.data_path)
        n_prompts = len(prompts)
        log.info("=" * 70)
        log.info("ComfyClaw + %s on %s — %d prompts",
                 model_config["name"], bench_config["name"], n_prompts)

    log.info("LLM: %s  Agent: %s  Diffusion: %s", LLM_MODEL, agent_name, model_config["name"])
    if is_baseline:
        log.info("MODE: BASELINE (1 iteration, no verifier, model skill only: %s)",
                 model_config.get("model_skill", "none"))
    if LLM_API_BASE:
        log.info("LLM API base: %s", LLM_API_BASE)
    log.info("ComfyUI instances: %d  [%s]", len(comfyui_addrs), ", ".join(comfyui_addrs))
    log.info("Max iterations: %d  Warm-start: %s  Evolve batch: %s  Parallel: %d",
             max_iterations, warm_start,
             str(evolve_batch_size) if evolve_batch_size > 0 else "disabled",
             parallel_size)
    log.info("Output: %s", paths["output_dir"])
    log.info("Evolved skills: %s", paths["evolved_skills_dir"])
    log.info("=" * 70)

    _write_evolution_metadata(
        paths["evolved_skills_dir"],
        benchmark=bench_config["short_name"],
        image_model=model_config["name"],
        agent_model=LLM_MODEL,
        n_prompts=n_prompts,
        max_iterations=max_iterations,
        evolve_batch_size=evolve_batch_size,
    )

    # ── Resume ────────────────────────────────────────────────────────
    if args.prompt:
        existing = []
        completed_idx = set()
    else:
        existing = load_results(results_path)
        completed_idx = {r["idx"] for r in existing if r.get("best_score", -1) >= 0}

    log.info("Prompts: %d total, %d already completed", n_prompts, len(completed_idx))
    results = list(existing)

    # ── Thread-safe state ─────────────────────────────────────────────
    results_lock = threading.Lock()
    pending_errors: list[dict] = []
    errors_lock = threading.Lock()
    pending_successes: list[dict] = []
    successes_lock = threading.Lock()

    def _run_and_record(i: int, prompt_text: str, server_address: str) -> None:
        tag = f"[idx={i:03d}]"
        log.info("%s RUNNING on %s  %s", tag, server_address, prompt_text)
        try:
            r = run_one(
                prompt_text, i,
                base_workflow=model_config["workflow"],
                model_config=model_config,
                paths=paths,
                max_iterations=max_iterations,
                warm_start=warm_start,
                server_address=server_address,
                is_baseline=is_baseline,
                skills_dir_override=SKILLS_DIR_OVERRIDE,
                benchmark_short=bench_config["short_name"],
                agent_name=agent_name,
            )
            with results_lock:
                results.append(r)
                save_results(results, results_path)
            log.info("%s DONE  base=%.3f best=%.3f  time=%ds  nodes=%d",
                     tag, r["baseline_score"], r["best_score"],
                     r["elapsed_s"], r["node_count"])

            if r.get("error_data"):
                with errors_lock:
                    pending_errors.extend(r["error_data"])
                    log.info("%s %d error(s) encountered — synthesizing learned skill",
                             tag, len(r["error_data"]))
                    synthesize_learned_skill(pending_errors, paths["learned_skills_dir"])
                    pending_errors.clear()

            if r["best_score"] >= 0.9 and r.get("sft_trace_count", 0) > 0:
                prompt_dir = os.path.join(
                    paths["detailed_dir"],
                    f"prompt_{i:03d}_{_slug(prompt_text)}"
                )
                sdata = _collect_success_data(r, prompt_dir)
                if sdata:
                    with successes_lock:
                        pending_successes.append(sdata)
                        if len(pending_successes) >= 3:
                            log.info("%s Synthesizing success patterns from %d entries",
                                     tag, len(pending_successes))
                            synthesize_success_patterns(
                                pending_successes, paths["learned_skills_dir"]
                            )
                            pending_successes.clear()

        except Exception as exc:
            log.error("%s FAILED: %s", tag, exc, exc_info=True)
            with errors_lock:
                pending_errors.append({
                    "iteration": 0, "phase": "crash",
                    "error": str(exc)[:500], "outcome": "failed", "rationale": "",
                })
                synthesize_learned_skill(pending_errors, paths["learned_skills_dir"])
                pending_errors.clear()
            with results_lock:
                results.append({
                    "idx": i, "prompt": prompt_text,
                    "baseline_score": 0.0, "best_score": 0.0,
                    "passed": [], "failed": [str(exc)], "error": str(exc),
                    "elapsed_s": 0.0, "node_count": 0, "iterations": 0,
                })
                save_results(results, results_path)

    # ── Build pending list ────────────────────────────────────────────
    pending: list[tuple[int, str]] = []
    for item in prompts:
        i = item["idx"]
        if i in completed_idx:
            r = next(r for r in results if r["idx"] == i)
            log.info("[%3d/%d] CACHED  base=%.3f best=%.3f  %s",
                     i + 1, n_prompts, r.get("baseline_score", 0),
                     r["best_score"], item["prompt"][:60])
        else:
            pending.append((i, item["prompt"]))

    # ── Run in parallel batches ───────────────────────────────────────
    evolve_cycle = 0
    total_since_evolve = 0

    for batch_start in range(0, len(pending), parallel_size):
        batch = pending[batch_start : batch_start + parallel_size]
        if not batch:
            break

        log.info("")
        log.info("── Parallel batch: %d prompt(s) [%s] ──",
                 len(batch), ", ".join(str(idx) for idx, _ in batch))

        if parallel_size == 1:
            _run_and_record(batch[0][0], batch[0][1], comfyui_addrs[0])
        else:
            with ThreadPoolExecutor(max_workers=parallel_size) as pool:
                futures = {
                    pool.submit(_run_and_record, idx, prompt_text,
                                comfyui_addrs[j % len(comfyui_addrs)]): idx
                    for j, (idx, prompt_text) in enumerate(batch)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        log.error("[idx=%03d] Unhandled thread error: %s", idx, exc)

        total_since_evolve += len(batch)

        if evolve_batch_size > 0 and total_since_evolve >= evolve_batch_size:
            evolve_cycle += 1
            log.info("")
            log.info("=" * 60)
            log.info("EVOLVING SKILLS — %d prompts since last evolution (cycle %d)",
                     total_since_evolve, evolve_cycle)
            log.info("=" * 60)
            try:
                run_batch_evolution(
                    results,
                    cycle=evolve_cycle,
                    evolved_dir=paths["evolved_skills_dir"],
                    model_short=model_config["short_name"],
                    benchmark_short=bench_config["short_name"],
                    agent_name=agent_name,
                )
            except Exception as exc:
                log.error("Skill evolution failed: %s", exc, exc_info=True)
            total_since_evolve = 0

    # ── Summary ───────────────────────────────────────────────────────
    results_sorted = sorted(results, key=lambda r: r["idx"])

    log.info("")
    log.info("=" * 90)
    log.info("RESULTS — %s on %s", model_config["name"], bench_config["name"])
    log.info("=" * 90)
    log.info("%-4s  %-40s  %8s  %8s  %7s  %6s", "#", "Prompt", "Baseline", "Best", "Delta", "Time")
    log.info("-" * 90)
    for r in results_sorted:
        b = r.get("baseline_score", 0)
        s = r["best_score"]
        d = s - b
        tag = "UP" if d > 0.01 else ("ERR" if r.get("error") else "")
        log.info("%-4d  %-40s  %8.3f  %8.3f  %+7.3f  %5.0fs  %s",
                 r["idx"], r["prompt"][:38], b, s, d, r.get("elapsed_s", 0), tag)
    log.info("-" * 90)

    valid = [r for r in results_sorted if not r.get("error")]
    if valid:
        mb = sum(r.get("baseline_score", 0) for r in valid) / len(valid)
        ms = sum(r["best_score"] for r in valid) / len(valid)
        mean_time = sum(r.get("elapsed_s", 0) for r in valid) / len(valid)
        improved = sum(1 for r in valid if r["best_score"] > r.get("baseline_score", 0) + 0.01)
        log.info("%-4s  %-40s  %8.3f  %8.3f  %5.0fs", "", "MEAN", mb, ms, mean_time)
        log.info("")
        log.info("Agent improvement: %d/%d prompts improved over baseline (+%.3f avg)",
                 improved, len(valid), ms - mb)

    log.info("Results saved to %s", results_path)
    log.info("Detailed results in %s", paths["detailed_dir"])

    # ── Aggregate SFT traces ──────────────────────────────────────────
    sft_all_path = os.path.join(paths["output_dir"], "sft_traces_all.jsonl")
    total_traces = 0
    with open(sft_all_path, "w", encoding="utf-8") as out_f:
        for prompt_subdir in sorted(Path(paths["detailed_dir"]).iterdir()):
            trace_file = prompt_subdir / "sft_traces.jsonl"
            if trace_file.is_file():
                for line in trace_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        out_f.write(line + "\n")
                        total_traces += 1
    if total_traces > 0:
        log.info("Aggregated %d SFT traces into %s", total_traces, sft_all_path)
    else:
        log.info("No SFT traces to aggregate")


if __name__ == "__main__":
    main()
