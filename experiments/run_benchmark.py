#!/usr/bin/env python3
"""
Unified ComfyClaw benchmark runner.

Supports multiple image generation models and benchmark datasets via CLI flags.
All shared logic (harness setup, parallel execution, skill evolution, SFT trace
collection, resume, and summary reporting) lives here.

Usage:
    python experiments/run_benchmark.py --model longcat --benchmark geneval2 \\
        --n-prompts 800 --max-iterations 5 --evolve-batch-size 5 --parallel 2

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("claw_bench")

from experiments.models import MODELS
from experiments.benchmarks import BENCHMARKS

# ── Config (env overridable) ──────────────────────────────────────────────
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
COMFYUI_ADDR = os.environ.get("COMFYUI_ADDR", "127.0.0.1:8188")
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4-5")
SKILLS_DIR = str(REPO_ROOT / "comfyclaw" / "skills")


def _slug(text: str, max_len: int = 50) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:max_len]


# ── Paths derived from model + benchmark ──────────────────────────────────

EXPERIMENTS_ROOT = Path(os.environ.get("EXPERIMENTS_ROOT", str(REPO_ROOT.parent / "comfy_agent_experiments_output")))


def _build_paths(model_short: str, bench_short: str) -> dict:
    experiment_dir = EXPERIMENTS_ROOT / f"{model_short}_{bench_short}"
    output_dir = os.environ.get("OUTPUT_DIR", str(experiment_dir / "results"))
    detailed_dir = os.environ.get("DETAILED_DIR", str(experiment_dir / "detailed"))
    evolved_root = REPO_ROOT / "comfyclaw" / "evolved_skills"
    evolved_dir = str(evolved_root / f"{model_short}_{bench_short}")
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


# ── Core: run a single prompt ─────────────────────────────────────────────

def run_one(
    prompt: str,
    idx: int,
    base_workflow: dict,
    model_config: dict,
    paths: dict,
    max_iterations: int,
    warm_start: bool,
) -> dict:
    from comfyclaw.harness import ClawHarness, HarnessConfig

    t0 = time.time()
    cfg = HarnessConfig(
        api_key=API_KEY,
        server_address=COMFYUI_ADDR,
        model=LLM_MODEL,
        max_iterations=max_iterations,
        success_threshold=0.95,
        sync_port=0,
        image_model=None,
        stage_gated=True,
        skills_dir=SKILLS_DIR,
        evolved_skills_dir=paths["evolved_skills_dir"],
        max_nodes=20,
        baseline_first=warm_start,
        max_images=max_iterations + 2,
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
    }


# ── Batch skill evolution ─────────────────────────────────────────────────

def run_batch_evolution(results: list[dict], cycle: int, evolved_dir: str) -> None:
    from comfyclaw.evolve import SkillEvolver

    evolver = SkillEvolver(
        evolved_skills_dir=evolved_dir,
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
    parser.add_argument("--max-iterations", type=int, default=5,
                        help="Max agent iterations per prompt (default: 5)")
    parser.add_argument("--evolve-batch-size", type=int, default=0,
                        help="Evolve skills every N prompts (0 = disabled)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Run N prompts concurrently (default: 1)")
    parser.add_argument("--no-warm-start", action="store_true",
                        help="Start from empty workflow instead of model's base workflow")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Override path to benchmark data file/directory")
    args = parser.parse_args()

    model_config = MODELS[args.model]
    bench_config = BENCHMARKS[args.benchmark]
    paths = _build_paths(model_config["short_name"], bench_config["short_name"])

    n_prompts = args.n_prompts or int(os.environ.get("N_PROMPTS", bench_config["default_n_prompts"]))
    max_iterations = args.max_iterations
    evolve_batch_size = args.evolve_batch_size
    parallel_size = args.parallel
    warm_start = not args.no_warm_start

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

    log.info("LLM: %s  Diffusion: %s", LLM_MODEL, model_config["name"])
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

    def _run_and_record(i: int, prompt_text: str) -> None:
        tag = f"[idx={i:03d}]"
        log.info("%s RUNNING  %s", tag, prompt_text)
        try:
            r = run_one(
                prompt_text, i,
                base_workflow=model_config["workflow"],
                model_config=model_config,
                paths=paths,
                max_iterations=max_iterations,
                warm_start=warm_start,
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
            _run_and_record(batch[0][0], batch[0][1])
        else:
            with ThreadPoolExecutor(max_workers=parallel_size) as pool:
                futures = {
                    pool.submit(_run_and_record, idx, prompt_text): idx
                    for idx, prompt_text in batch
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
                run_batch_evolution(results, cycle=evolve_cycle,
                                    evolved_dir=paths["evolved_skills_dir"])
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
