#!/usr/bin/env python3
"""
ComfyClaw-only benchmark — 20 GenEval2 prompts with warm-start + baseline_first.
Supports resume: completed prompts are cached to disk and skipped on re-run.

Saves ALL intermediate images and detailed per-iteration JSON for each prompt
into per-prompt subfolders under DETAILED_DIR.
"""
import argparse, copy, json, logging, os, re, sys, time

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("claw_bench")

# ── Config ────────────────────────────────────────────────────────────────
GENEVAL2_PATH = os.environ.get("GENEVAL2_DATA", str(REPO_ROOT.parent / "GenEval2" / "geneval2_data.jsonl"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(REPO_ROOT.parent / "benchmark_claw_20"))
DETAILED_DIR = os.environ.get("DETAILED_DIR", str(REPO_ROOT.parent / "benchmark_claw_detailed"))
CHECKPOINT = "DreamShaper_8_pruned.safetensors"
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
COMFYUI_ADDR = "127.0.0.1:8188"
LLM_MODEL = "anthropic/claude-sonnet-4-5"
SKILLS_DIR = str(REPO_ROOT / "comfyclaw" / "skills")

N_PROMPTS = int(os.environ.get("N_PROMPTS", 20))
MAX_ITERATIONS = 2
WARM_START = True

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(DETAILED_DIR, exist_ok=True)


def _slug(text: str, max_len: int = 50) -> str:
    """Turn a prompt string into a filesystem-safe folder name."""
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:max_len]

BASE_WORKFLOW = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": CHECKPOINT},
        "_meta": {"title": "Load Checkpoint"},
    },
    "2": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["1", 1], "text": ""},
        "_meta": {"title": "Positive Prompt"},
    },
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["1", 1], "text": "blurry, ugly, deformed, low quality"},
        "_meta": {"title": "Negative Prompt"},
    },
    "4": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 512, "height": 512, "batch_size": 1},
        "_meta": {"title": "Empty Latent"},
    },
    "5": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["4", 0],
            "seed": 42,
            "steps": 25,
            "cfg": 7.0,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": 1.0,
        },
        "_meta": {"title": "KSampler"},
    },
    "6": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        "_meta": {"title": "VAE Decode"},
    },
    "7": {
        "class_type": "SaveImage",
        "inputs": {"images": ["6", 0], "filename_prefix": "ComfyClaw"},
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


EVOLVED_SKILLS_DIR = str(REPO_ROOT / "comfyclaw" / "skills_evolved")
LEARNED_SKILLS_DIR = os.path.join(EVOLVED_SKILLS_DIR, "learned-errors")

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
        image_model=CHECKPOINT,
        stage_gated=True,
        skills_dir=SKILLS_DIR,
        evolved_skills_dir=EVOLVED_SKILLS_DIR,
        max_nodes=20,
        baseline_first=WARM_START,
        max_images=MAX_ITERATIONS + 2,
    )

    init_wf = copy.deepcopy(BASE_WORKFLOW) if WARM_START else {}
    harness = ClawHarness.from_workflow_dict(init_wf, cfg)
    with harness:
        image_bytes = harness.run(prompt)

    # Extract best score from both evolution log and memory (baseline is in memory)
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

    # Save best image to the original location
    img_path = None
    if image_bytes:
        img_path = os.path.join(OUTPUT_DIR, "images", f"claw_{idx:02d}.png")
        with open(img_path, "wb") as f:
            f.write(image_bytes)

    elapsed = time.time() - t0
    node_count = (
        harness.evolution_log.entries[-1].node_count_after
        if harness.evolution_log.entries else 7
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
        })

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
        "final_node_count": node_count,
        "config": {
            "checkpoint": CHECKPOINT,
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
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    global N_PROMPTS, MAX_ITERATIONS

    parser = argparse.ArgumentParser(description="ComfyClaw benchmark / single-prompt runner")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Run a single custom prompt instead of the GenEval2 benchmark set")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help=f"Override max agent iterations (default: {MAX_ITERATIONS})")
    args = parser.parse_args()

    if args.max_iterations is not None:
        MAX_ITERATIONS = args.max_iterations

    if args.prompt:
        N_PROMPTS = 1
        prompts = [{"prompt": args.prompt}]
        log.info("=" * 70)
        log.info("ComfyClaw — custom prompt")
        log.info("Prompt: %s", args.prompt)
    else:
        prompts = load_prompts(N_PROMPTS)
        log.info("=" * 70)
        log.info("ComfyClaw Benchmark — %d GenEval2 prompts", N_PROMPTS)

    log.info("Model: %s  Checkpoint: %s", LLM_MODEL, CHECKPOINT)
    log.info("Max iterations: %d  Warm-start: %s", MAX_ITERATIONS, WARM_START)
    log.info("=" * 70)

    if args.prompt:
        existing = []
        completed_idx = set()
    else:
        existing = load_results()
        completed_idx = {r["idx"] for r in existing if r.get("best_score", -1) >= 0}

    log.info("Prompts: %d total, %d already completed", len(prompts), len(completed_idx))
    for i, p in enumerate(prompts):
        tag = "✓" if i in completed_idx else " "
        log.info("  [%s] %2d: %s", tag, i, p["prompt"])

    results = list(existing)
    pending_errors: list[dict] = []

    for i, item in enumerate(prompts):
        prompt = item["prompt"]
        if i in completed_idx:
            r = next(r for r in results if r["idx"] == i)
            log.info("[%2d/%d] CACHED  base=%.3f best=%.3f  %s",
                     i + 1, N_PROMPTS, r.get("baseline_score", 0), r["best_score"], prompt)
            continue

        log.info("")
        log.info("[%2d/%d] RUNNING  %s", i + 1, N_PROMPTS, prompt)
        try:
            r = run_one(prompt, i)
            results.append(r)
            save_results(results)
            log.info("[%2d/%d] DONE  base=%.3f best=%.3f  time=%ds  nodes=%d",
                     i + 1, N_PROMPTS, r["baseline_score"], r["best_score"],
                     r["elapsed_s"], r["node_count"])

            if r.get("error_data"):
                pending_errors.extend(r["error_data"])
                log.info("  %d error(s) encountered — synthesizing learned skill",
                         len(r["error_data"]))
                synthesize_learned_skill(pending_errors)
                pending_errors.clear()

        except Exception as exc:
            log.error("[%2d/%d] FAILED: %s", i + 1, N_PROMPTS, exc, exc_info=True)
            pending_errors.append({
                "iteration": 0,
                "phase": "crash",
                "error": str(exc)[:500],
                "outcome": "failed",
                "rationale": "",
            })
            synthesize_learned_skill(pending_errors)
            pending_errors.clear()
            results.append({
                "idx": i,
                "prompt": prompt,
                "baseline_score": 0.0,
                "best_score": 0.0,
                "passed": [],
                "failed": [str(exc)],
                "error": str(exc),
                "elapsed_s": 0.0,
                "node_count": 0,
                "iterations": 0,
            })
            save_results(results)

    # ── Summary ───────────────────────────────────────────────────────
    results_sorted = sorted(results, key=lambda r: r["idx"])
    baseline_scores = [r.get("baseline_score", 0) for r in results_sorted]
    best_scores = [r["best_score"] for r in results_sorted]
    times = [r.get("elapsed_s", 0) for r in results_sorted]

    log.info("")
    log.info("=" * 90)
    log.info("RESULTS SUMMARY")
    log.info("=" * 90)
    log.info("%-3s  %-40s  %8s  %8s  %6s  %5s", "#", "Prompt", "Baseline", "Best", "Time", "Nodes")
    log.info("-" * 90)
    for r in results_sorted:
        improved = "↑" if r["best_score"] > r.get("baseline_score", 0) + 0.01 else " "
        log.info("%-3d  %-40s  %8.3f  %8.3f  %5.0fs  %5d  %s",
                 r["idx"], r["prompt"][:38], r.get("baseline_score", 0),
                 r["best_score"], r.get("elapsed_s", 0), r.get("node_count", 0), improved)
    log.info("-" * 90)

    mean_base = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0
    mean_best = sum(best_scores) / len(best_scores) if best_scores else 0
    mean_time = sum(times) / len(times) if times else 0
    improved_count = sum(1 for b, s in zip(baseline_scores, best_scores) if s > b + 0.01)

    log.info("%-3s  %-40s  %8.3f  %8.3f  %5.0fs", "", "MEAN", mean_base, mean_best, mean_time)
    log.info("")
    log.info("Agent improvement: %d/%d prompts improved over baseline (+%.3f avg)",
             improved_count, len(results_sorted), mean_best - mean_base)
    log.info("Results saved to %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
