"""
NeurIPS experiment runner — orchestrates the full evaluation pipeline.

Runs these experiments in sequence:
1. Baseline: hand-crafted skills, no stage gating
2. Stage-gated: hand-crafted skills + stage-gated tool disclosure
3. Self-evolved (N rounds): run evolution cycles, then re-benchmark

Usage::

    python experiments/run_neurips_experiments.py \\
        --geneval2-data path/to/geneval2_data.jsonl \\
        --crea-data path/to/CREA/data.jsonl \\
        --server 127.0.0.1:8188 \\
        --max-prompts 50 \\
        --evolution-cycles 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from comfyclaw.benchmark import BenchmarkConfig, BenchmarkRunner
from comfyclaw.evolve import SkillEvolver

log = logging.getLogger(__name__)

SKILLS_DIR = str(Path(__file__).resolve().parent.parent / "comfyclaw" / "skills")


def run_benchmark(
    suite: str,
    name: str,
    data_path: str,
    output_dir: str,
    max_iterations: int = 3,
    max_prompts: int | None = None,
    server: str = "127.0.0.1:8188",
    model: str = "anthropic/claude-sonnet-4-5",
    image_model: str | None = None,
    stage_gated: bool = False,
    workflow: str | None = None,
) -> dict:
    """Run a single benchmark and return the summary."""
    config = BenchmarkConfig(
        suite=suite,
        name=name,
        data_path=data_path,
        output_dir=output_dir,
        max_iterations=max_iterations,
        server_address=server,
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        model=model,
        image_model=image_model,
        workflow_path=workflow,
        stage_gated=stage_gated,
        max_prompts=max_prompts,
    )

    runner = BenchmarkRunner(config)
    result = runner.run()

    return {
        "suite": result.suite,
        "name": result.name,
        "mean_score": result.mean_score,
        "completed": result.completed,
        "failed": result.failed,
        "total_prompts": result.total_prompts,
        "mean_latency_s": result.mean_latency_s,
        "metrics": result.metrics,
    }


def run_evolution(
    results_path: str,
    skills_dir: str,
    model: str,
    max_cycles: int = 5,
) -> list[dict]:
    """Run self-evolution cycles and return reports."""
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    evolved_dir = str(Path(skills_dir).parent / "skills_evolved")

    evolver = SkillEvolver(
        evolved_skills_dir=evolved_dir,
        llm_model=model,
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        max_mutations_per_cycle=3,
    )

    reports = evolver.run_multi_cycle(
        results=results,
        max_cycles=max_cycles,
    )

    return [
        {
            "cycle": r.cycle,
            "pre_score": r.pre_mean_score,
            "post_score": r.post_mean_score,
            "mutations_proposed": r.mutations_proposed,
            "mutations_accepted": r.mutations_accepted,
            "duration_s": r.duration_s,
        }
        for r in reports
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="ComfyClaw NeurIPS experiment runner")
    parser.add_argument("--geneval2-data", help="Path to GenEval2 JSONL")
    parser.add_argument("--crea-data", help="Path to CREA JSONL")
    parser.add_argument("--oneig-data", help="Path to OneIG-EN JSONL")
    parser.add_argument("--server", default="127.0.0.1:8188")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4-5")
    parser.add_argument("--image-model", default=None)
    parser.add_argument("--workflow", default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--evolution-cycles", type=int, default=5)
    parser.add_argument("--output-dir", default="experiment_results")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    os.makedirs(args.output_dir, exist_ok=True)
    all_results: dict[str, dict] = {}
    t_start = time.time()

    suites = []
    if args.geneval2_data:
        suites.append(("geneval2", args.geneval2_data))
    if args.crea_data:
        suites.append(("crea", args.crea_data))
    if args.oneig_data:
        suites.append(("oneig", args.oneig_data))

    if not suites:
        print("Error: provide at least one of --geneval2-data, --crea-data, --oneig-data")
        sys.exit(1)

    # ── Experiment 1: Baseline (hand-crafted skills, no stage gating) ────
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Baseline (hand-crafted skills)")
    print("=" * 60)

    for suite, data_path in suites:
        name = f"baseline_{suite}"
        log.info("Running %s baseline...", suite)
        result = run_benchmark(
            suite=suite,
            name=name,
            data_path=data_path,
            output_dir=args.output_dir,
            max_iterations=args.max_iterations,
            max_prompts=args.max_prompts,
            server=args.server,
            model=args.model,
            image_model=args.image_model,
            workflow=args.workflow,
            stage_gated=False,
        )
        all_results[name] = result
        print(f"  {suite}: score={result['mean_score']:.4f}, "
              f"completed={result['completed']}/{result['total_prompts']}")

    # ── Experiment 2: Stage-gated tool disclosure ────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Stage-gated tool disclosure")
    print("=" * 60)

    for suite, data_path in suites:
        name = f"stage_gated_{suite}"
        log.info("Running %s with stage gating...", suite)
        result = run_benchmark(
            suite=suite,
            name=name,
            data_path=data_path,
            output_dir=args.output_dir,
            max_iterations=args.max_iterations,
            max_prompts=args.max_prompts,
            server=args.server,
            model=args.model,
            image_model=args.image_model,
            workflow=args.workflow,
            stage_gated=True,
        )
        all_results[name] = result
        print(f"  {suite}: score={result['mean_score']:.4f}, "
              f"completed={result['completed']}/{result['total_prompts']}")

    # ── Experiment 3: Self-evolution ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Self-evolution cycles")
    print("=" * 60)

    # Use baseline results for evolution input
    for suite, data_path in suites:
        baseline_results_path = os.path.join(
            args.output_dir, f"baseline_{suite}", "results.json"
        )
        if not os.path.exists(baseline_results_path):
            log.warning("Skipping evolution for %s: no baseline results", suite)
            continue

        log.info("Running %d evolution cycles on %s...", args.evolution_cycles, suite)
        evo_reports = run_evolution(
            results_path=baseline_results_path,
            skills_dir=SKILLS_DIR,
            model=args.model,
            max_cycles=args.evolution_cycles,
        )
        all_results[f"evolution_{suite}"] = {"reports": evo_reports}

        for r in evo_reports:
            delta = r["post_score"] - r["pre_score"]
            print(f"  Cycle {r['cycle']}: {r['pre_score']:.4f} -> "
                  f"{r['post_score']:.4f} ({'+' if delta >= 0 else ''}{delta:.4f}), "
                  f"mutations: {r['mutations_accepted']}/{r['mutations_proposed']}")

    # ── Experiment 4: Post-evolution benchmark ───────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Post-evolution benchmark")
    print("=" * 60)

    for suite, data_path in suites:
        name = f"post_evolve_{suite}"
        log.info("Running %s post-evolution...", suite)
        result = run_benchmark(
            suite=suite,
            name=name,
            data_path=data_path,
            output_dir=args.output_dir,
            max_iterations=args.max_iterations,
            max_prompts=args.max_prompts,
            server=args.server,
            model=args.model,
            image_model=args.image_model,
            workflow=args.workflow,
            stage_gated=True,
        )
        all_results[name] = result
        print(f"  {suite}: score={result['mean_score']:.4f}, "
              f"completed={result['completed']}/{result['total_prompts']}")

    # ── Final summary ────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    for suite, _ in suites:
        baseline = all_results.get(f"baseline_{suite}", {})
        staged = all_results.get(f"stage_gated_{suite}", {})
        evolved = all_results.get(f"post_evolve_{suite}", {})

        print(f"\n{suite.upper()}:")
        if baseline:
            print(f"  Baseline:       {baseline.get('mean_score', 0):.4f}")
        if staged:
            print(f"  Stage-gated:    {staged.get('mean_score', 0):.4f}")
        if evolved:
            print(f"  Post-evolution: {evolved.get('mean_score', 0):.4f}")
        if baseline and evolved:
            delta = evolved.get("mean_score", 0) - baseline.get("mean_score", 0)
            print(f"  Improvement:    {'+' if delta >= 0 else ''}{delta:.4f}")

    print(f"\nTotal experiment time: {total_time:.0f}s ({total_time/60:.1f}min)")

    # Save comprehensive results
    summary_path = os.path.join(args.output_dir, "experiment_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full results saved to: {summary_path}")


if __name__ == "__main__":
    main()
