#!/usr/bin/env python3
"""
ComfyClaw-only benchmark — 20 GenEval2 prompts with warm-start + baseline_first.
Supports resume: completed prompts are cached to disk and skipped on re-run.

Saves ALL intermediate images and detailed per-iteration JSON for each prompt
into per-prompt subfolders under DETAILED_DIR.
"""
import copy, json, logging, os, re, sys, time

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

N_PROMPTS = 20
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
        max_nodes=20,
        baseline_first=WARM_START,
        max_images=MAX_ITERATIONS + 2,  # retain all iteration images in memory
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
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("ComfyClaw Benchmark — %d GenEval2 prompts", N_PROMPTS)
    log.info("Model: %s  Checkpoint: %s", LLM_MODEL, CHECKPOINT)
    log.info("Max iterations: %d  Warm-start: %s", MAX_ITERATIONS, WARM_START)
    log.info("=" * 70)

    prompts = load_prompts(N_PROMPTS)
    existing = load_results()
    completed_idx = {r["idx"] for r in existing if r.get("best_score", -1) >= 0}

    log.info("Prompts: %d total, %d already completed", len(prompts), len(completed_idx))
    for i, p in enumerate(prompts):
        tag = "✓" if i in completed_idx else " "
        log.info("  [%s] %2d: %s", tag, i, p["prompt"])

    results = list(existing)

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
            save_results(results)  # save after each prompt for resume
            log.info("[%2d/%d] DONE  base=%.3f best=%.3f  time=%ds  nodes=%d",
                     i + 1, N_PROMPTS, r["baseline_score"], r["best_score"],
                     r["elapsed_s"], r["node_count"])
        except Exception as exc:
            log.error("[%2d/%d] FAILED: %s", i + 1, N_PROMPTS, exc, exc_info=True)
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
