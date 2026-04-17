#!/usr/bin/env python3
"""
ComfyClaw benchmark with LongCat Image as base model.
Uses the LongCat Image pipeline: UNETLoader + CLIPLoader + VAELoader +
ModelSamplingAuraFlow + EmptySD3LatentImage.

Saves ALL intermediate images and detailed per-iteration JSON for each prompt
into per-prompt subfolders under DETAILED_DIR.
"""
import argparse, copy, json, logging, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("longcat_bench")

# ── Config ────────────────────────────────────────────────────────────────
GENEVAL2_PATH = os.environ.get("GENEVAL2_DATA", str(REPO_ROOT.parent / "GenEval2" / "geneval2_data.jsonl"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(REPO_ROOT.parent / "benchmark_longcat_10"))
DETAILED_DIR = os.environ.get("DETAILED_DIR", str(REPO_ROOT.parent / "benchmark_longcat_detailed"))
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
COMFYUI_ADDR = "127.0.0.1:8188"
LLM_MODEL = "anthropic/claude-sonnet-4-5"
SKILLS_DIR = str(REPO_ROOT / "comfyclaw" / "skills")

N_PROMPTS = int(os.environ.get("N_PROMPTS", 10))
MAX_ITERATIONS = 2
WARM_START = True
EVOLVE_BATCH_SIZE = int(os.environ.get("EVOLVE_BATCH_SIZE", 0))  # 0 = disabled
PARALLEL_SIZE = int(os.environ.get("PARALLEL_SIZE", 1))  # 1 = sequential

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(DETAILED_DIR, exist_ok=True)


def _slug(text: str, max_len: int = 50) -> str:
    """Turn a prompt string into a filesystem-safe folder name."""
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:max_len]

# ── LongCat Image base workflow (API format) ──────────────────────────────
#
# Pipeline: UNETLoader → CFGNorm → KSampler
#           CLIPLoader → CLIPTextEncode (pos) → FluxGuidance → KSampler
#           CLIPLoader → CLIPTextEncode (neg) → FluxGuidance → KSampler
#           VAELoader → VAEDecode
#           EmptySD3LatentImage → KSampler
#
LONGCAT_BASE_WORKFLOW = {
    "1": {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": "longcat_image_bf16.safetensors",
            "weight_dtype": "default",
        },
        "_meta": {"title": "UNET Loader"},
    },
    "2": {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": "qwen_2.5_vl_7b.safetensors",
            "type": "longcat_image",
            "device": "default",
        },
        "_meta": {"title": "CLIP Loader"},
    },
    "3": {
        "class_type": "VAELoader",
        "inputs": {
            "vae_name": "ae.safetensors",
        },
        "_meta": {"title": "VAE Loader"},
    },
    "4": {
        "class_type": "CFGNorm",
        "inputs": {
            "model": ["1", 0],
            "strength": 1.0,
        },
        "_meta": {"title": "CFG Norm"},
    },
    "5": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["2", 0],
            "text": "",
        },
        "_meta": {"title": "Positive Prompt"},
    },
    "6": {
        "class_type": "FluxGuidance",
        "inputs": {
            "conditioning": ["5", 0],
            "guidance": 4.0,
        },
        "_meta": {"title": "Positive FluxGuidance"},
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["2", 0],
            "text": "blurry, low resolution, oversaturated, harsh lighting, messy composition, distorted face, extra fingers, bad anatomy, watermark",
        },
        "_meta": {"title": "Negative Prompt"},
    },
    "8": {
        "class_type": "FluxGuidance",
        "inputs": {
            "conditioning": ["7", 0],
            "guidance": 4.0,
        },
        "_meta": {"title": "Negative FluxGuidance"},
    },
    "9": {
        "class_type": "EmptySD3LatentImage",
        "inputs": {
            "width": 1024,
            "height": 1024,
            "batch_size": 1,
        },
        "_meta": {"title": "Empty Latent (1024x1024)"},
    },
    "10": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["8", 0],
            "latent_image": ["9", 0],
            "seed": 42,
            "steps": 20,
            "cfg": 4.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
        },
        "_meta": {"title": "KSampler"},
    },
    "11": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["10", 0],
            "vae": ["3", 0],
        },
        "_meta": {"title": "VAE Decode"},
    },
    "12": {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["11", 0],
            "filename_prefix": "LongCatClaw",
        },
        "_meta": {"title": "Save Image"},
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────

RESULTS_PATH = os.path.join(OUTPUT_DIR, "results.json")

def load_results() -> list[dict]:
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []

def save_results(results: list[dict]):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

def load_prompts(n: int) -> list[dict]:
    items = []
    with open(GENEVAL2_PATH) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            items.append(json.loads(line))
    return items


_output_basename = os.path.basename(OUTPUT_DIR.rstrip("/"))
EVOLVED_SKILLS_DIR = str(REPO_ROOT / "comfyclaw" / f"skills_evolved_{_output_basename}")
LEARNED_SKILLS_DIR = os.path.join(EVOLVED_SKILLS_DIR, "learned-errors")


def _write_evolution_metadata(evolved_dir: str, **kwargs: object) -> None:
    """Write or update a metadata JSON in the evolved skills directory."""
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
    """Extract error/repair events from a completed harness run."""
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


def synthesize_learned_skill(new_errors: list[dict]) -> bool:
    """Call the LLM to synthesize or update a learned-errors skill from error data.

    Returns True if a skill was written, False if skipped.
    """
    import litellm

    skill_dir = Path(LEARNED_SKILLS_DIR)
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

    prompt_text += (
        "Output ONLY the complete SKILL.md content. It MUST follow this exact format:\n"
        "1. Start with YAML frontmatter between --- delimiters\n"
        "2. The 'name' field MUST be exactly 'learned-errors'\n"
        "3. The 'description' field should explain when the agent should read this skill\n"
        "4. After the frontmatter, write clear Markdown instructions\n\n"
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
            api_key=API_KEY,
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


def run_one(prompt: str, idx: int) -> dict:
    from comfyclaw.harness import ClawHarness, HarnessConfig

    t0 = time.time()
    cfg = HarnessConfig(
        api_key=API_KEY,
        server_address=COMFYUI_ADDR,
        model=LLM_MODEL,
        max_iterations=MAX_ITERATIONS,
        success_threshold=0.95,
        sync_port=0,
        image_model=None,
        stage_gated=True,
        skills_dir=SKILLS_DIR,
        evolved_skills_dir=EVOLVED_SKILLS_DIR,
        max_nodes=20,
        baseline_first=WARM_START,
        max_images=MAX_ITERATIONS + 2,
    )

    init_wf = copy.deepcopy(LONGCAT_BASE_WORKFLOW) if WARM_START else {}
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
        img_path = os.path.join(OUTPUT_DIR, "images", f"longcat_{idx:02d}.png")
        with open(img_path, "wb") as f:
            f.write(image_bytes)

    elapsed = time.time() - t0
    node_count = (
        harness.evolution_log.entries[-1].node_count_after
        if harness.evolution_log.entries else 10
    )

    # ── Save detailed per-prompt results ──────────────────────────────
    prompt_dir = os.path.join(DETAILED_DIR, f"prompt_{idx:02d}_{_slug(prompt)}")
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

    # Aggregate timing across all iterations
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
        "total_iterations": len(harness.evolution_log.entries),
        "elapsed_s": round(elapsed, 1),
        "timing": timing_summary,
        "final_node_count": node_count,
        "config": {
            "model": "LongCat Image (BF16)",
            "llm_model": LLM_MODEL,
            "max_iterations": MAX_ITERATIONS,
            "warm_start": WARM_START,
            "stage_gated": True,
        },
        "iterations": iterations_detail,
    }
    with open(os.path.join(prompt_dir, "details.json"), "w") as f:
        json.dump(detail_json, f, indent=2, default=str)

    log.info("  Detailed results saved to %s", prompt_dir)

    error_data = _collect_error_data(harness)

    # ── Save SFT conversation traces ─────────────────────────────────
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
        "iterations": len(harness.evolution_log.entries),
        "image_path": img_path,
        "error_data": error_data,
        "sft_trace_count": len(sft_traces),
    }


# ── Batch skill evolution ──────────────────────────────────────────────────

def run_batch_evolution(results: list[dict], cycle: int) -> None:
    """Run one SkillEvolver cycle on the accumulated results."""
    from comfyclaw.evolve import SkillEvolver

    evolver = SkillEvolver(
        evolved_skills_dir=EVOLVED_SKILLS_DIR,
        llm_model=LLM_MODEL,
        api_key=API_KEY,
        min_improvement=0.02,
        max_mutations_per_cycle=3,
    )
    report = evolver.run_cycle(results, cycle=cycle)
    log.info("  Evolution cycle %d: %s", cycle, report.summary())
    if report.mutations_accepted > 0:
        log.info("  %d mutation(s) accepted — skills updated for next batch",
                 report.mutations_accepted)
    else:
        log.info("  No mutations accepted this cycle")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    global N_PROMPTS, MAX_ITERATIONS, EVOLVE_BATCH_SIZE, PARALLEL_SIZE

    parser = argparse.ArgumentParser(description="ComfyClaw LongCat benchmark / single-prompt runner")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Run a single custom prompt instead of the GenEval2 benchmark set")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help=f"Override max agent iterations (default: {MAX_ITERATIONS})")
    parser.add_argument("--evolve-batch-size", type=int, default=None,
                        help="Evolve skills every N prompts (0 = disabled, default: EVOLVE_BATCH_SIZE env or 0)")
    parser.add_argument("--parallel", type=int, default=None,
                        help="Run N prompts concurrently (default: PARALLEL_SIZE env or 1)")
    args = parser.parse_args()

    if args.max_iterations is not None:
        MAX_ITERATIONS = args.max_iterations
    if args.evolve_batch_size is not None:
        EVOLVE_BATCH_SIZE = args.evolve_batch_size
    if args.parallel is not None:
        PARALLEL_SIZE = args.parallel

    if args.prompt:
        N_PROMPTS = 1
        prompts = [{"prompt": args.prompt}]
        log.info("=" * 70)
        log.info("ComfyClaw + LongCat Image — custom prompt")
        log.info("Prompt: %s", args.prompt)
    else:
        prompts = load_prompts(N_PROMPTS)
        log.info("=" * 70)
        log.info("ComfyClaw + LongCat Image Benchmark — %d prompts", N_PROMPTS)

    log.info("LLM: %s  Diffusion: LongCat Image (BF16)", LLM_MODEL)
    log.info("Max iterations: %d  Warm-start: %s  Evolve batch: %s  Parallel: %d",
             MAX_ITERATIONS, WARM_START,
             str(EVOLVE_BATCH_SIZE) if EVOLVE_BATCH_SIZE > 0 else "disabled",
             PARALLEL_SIZE)
    log.info("=" * 70)

    _write_evolution_metadata(
        EVOLVED_SKILLS_DIR,
        benchmark="claw_longcat_benchmark",
        image_model="LongCat Image (BF16)",
        agent_model=LLM_MODEL,
        n_prompts=N_PROMPTS,
        max_iterations=MAX_ITERATIONS,
        evolve_batch_size=EVOLVE_BATCH_SIZE,
    )

    if args.prompt:
        existing = []
        completed_idx = set()
    else:
        existing = load_results()
        completed_idx = {r["idx"] for r in existing if r.get("best_score", -1) >= 0}

    log.info("Prompts: %d total, %d already completed", len(prompts), len(completed_idx))
    results = list(existing)

    # Thread-safe state for parallel execution
    results_lock = threading.Lock()
    pending_errors: list[dict] = []
    errors_lock = threading.Lock()

    def _run_and_record(i: int, prompt_text: str) -> None:
        """Run a single prompt and record results (thread-safe)."""
        tag = f"[idx={i:02d}]"
        log.info("%s RUNNING  %s", tag, prompt_text)
        try:
            r = run_one(prompt_text, i)
            with results_lock:
                results.append(r)
                save_results(results)
            log.info("%s DONE  base=%.3f best=%.3f  time=%ds  nodes=%d",
                     tag, r["baseline_score"], r["best_score"],
                     r["elapsed_s"], r["node_count"])

            if r.get("error_data"):
                with errors_lock:
                    pending_errors.extend(r["error_data"])
                    log.info("%s %d error(s) encountered — synthesizing learned skill",
                             tag, len(r["error_data"]))
                    synthesize_learned_skill(pending_errors)
                    pending_errors.clear()

        except Exception as exc:
            log.error("%s FAILED: %s", tag, exc, exc_info=True)
            with errors_lock:
                pending_errors.append({
                    "iteration": 0,
                    "phase": "crash",
                    "error": str(exc)[:500],
                    "outcome": "failed",
                    "rationale": "",
                })
                synthesize_learned_skill(pending_errors)
                pending_errors.clear()
            with results_lock:
                results.append({
                    "idx": i, "prompt": prompt_text, "baseline_score": 0.0, "best_score": 0.0,
                    "passed": [], "failed": [str(exc)], "error": str(exc),
                    "elapsed_s": 0.0, "node_count": 0, "iterations": 0,
                })
                save_results(results)

    # Build list of pending prompts (skip cached)
    pending: list[tuple[int, str]] = []
    for i, item in enumerate(prompts):
        if i in completed_idx:
            r = next(r for r in results if r["idx"] == i)
            log.info("[%2d/%d] CACHED  base=%.3f best=%.3f  %s",
                     i + 1, N_PROMPTS, r.get("baseline_score", 0), r["best_score"],
                     item["prompt"])
        else:
            pending.append((i, item["prompt"]))

    # Process pending prompts in parallel batches
    evolve_cycle = 0
    total_since_evolve = 0

    for batch_start in range(0, len(pending), PARALLEL_SIZE):
        batch = pending[batch_start : batch_start + PARALLEL_SIZE]
        if not batch:
            break

        log.info("")
        log.info("── Parallel batch: %d prompt(s) [%s] ──",
                 len(batch), ", ".join(str(idx) for idx, _ in batch))

        if PARALLEL_SIZE == 1:
            _run_and_record(batch[0][0], batch[0][1])
        else:
            with ThreadPoolExecutor(max_workers=PARALLEL_SIZE) as pool:
                futures = {
                    pool.submit(_run_and_record, idx, prompt_text): idx
                    for idx, prompt_text in batch
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        log.error("[idx=%02d] Unhandled thread error: %s", idx, exc)

        total_since_evolve += len(batch)

        # Batch skill evolution — only between parallel batches, never mid-flight
        if EVOLVE_BATCH_SIZE > 0 and total_since_evolve >= EVOLVE_BATCH_SIZE:
            evolve_cycle += 1
            log.info("")
            log.info("=" * 60)
            log.info("EVOLVING SKILLS — %d prompts since last evolution (cycle %d)",
                     total_since_evolve, evolve_cycle)
            log.info("=" * 60)
            try:
                run_batch_evolution(results, cycle=evolve_cycle)
            except Exception as exc:
                log.error("Skill evolution failed: %s", exc, exc_info=True)
            total_since_evolve = 0

    # ── Summary ───────────────────────────────────────────────────────
    results_sorted = sorted(results, key=lambda r: r["idx"])

    log.info("")
    log.info("=" * 90)
    log.info("RESULTS — LongCat Image")
    log.info("=" * 90)
    log.info("%-3s  %-40s  %8s  %8s  %7s  %6s", "#", "Prompt", "Baseline", "Best", "Delta", "Time")
    log.info("-" * 90)
    for r in results_sorted:
        b = r.get("baseline_score", 0)
        s = r["best_score"]
        d = s - b
        tag = "UP" if d > 0.01 else ("ERR" if r.get("error") else "")
        log.info("%-3d  %-40s  %8.3f  %8.3f  %+7.3f  %5.0fs  %s",
                 r["idx"], r["prompt"][:38], b, s, d, r.get("elapsed_s", 0), tag)
    log.info("-" * 90)

    valid = [r for r in results_sorted if not r.get("error")]
    if valid:
        mb = sum(r.get("baseline_score", 0) for r in valid) / len(valid)
        ms = sum(r["best_score"] for r in valid) / len(valid)
        mean_time = sum(r.get("elapsed_s", 0) for r in valid) / len(valid)
        improved = sum(1 for r in valid if r["best_score"] > r.get("baseline_score", 0) + 0.01)
        log.info("%-3s  %-40s  %8.3f  %8.3f  %5.0fs", "", "MEAN", mb, ms, mean_time)
        log.info("")
        log.info("Agent improvement: %d/%d prompts improved over baseline (+%.3f avg)",
                 improved, len(valid), ms - mb)

    log.info("Results saved to %s", RESULTS_PATH)
    log.info("Detailed results in %s", DETAILED_DIR)

    # ── Aggregate SFT traces into one training file ───────────────────
    sft_all_path = os.path.join(OUTPUT_DIR, "sft_traces_all.jsonl")
    total_traces = 0
    with open(sft_all_path, "w", encoding="utf-8") as out_f:
        for prompt_subdir in sorted(Path(DETAILED_DIR).iterdir()):
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
