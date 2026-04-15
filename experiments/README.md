# Experiments

Benchmark and experiment scripts for evaluating ComfyClaw against baselines.

## Prerequisites

- ComfyUI running and accessible (default: `127.0.0.1:8188`)
- In ComfyUI, do python main.py --listen 127.0.0.1 --port 8188
- `ANTHROPIC_API_KEY` environment variable set
- GenEval2 data (`geneval2_data.jsonl`) — override path with `GENEVAL2_DATA` env var
- Model checkpoints installed in ComfyUI's `models/` directories
- Run experiment `python experiments/claw_qwen_benchmark.py --evolve-batch-size 5 --max_iterations 5 --parallel 3`

## Scripts

| Script | Description |
|---|---|
| `gems_vs_comfyclaw.py` | Head-to-head comparison: GEMS (prompt-only refinement) vs ComfyClaw (topology evolution) on 10 GenEval2 prompts |
| `claw_benchmark.py` | ComfyClaw-only benchmark — 20 GenEval2 prompts with DreamShaper, warm-start + baseline_first |
| `claw_qwen_benchmark.py` | ComfyClaw benchmark with Qwen Image 2512 as base model (10 prompts) |
| `full_experiment.py` | Full NeurIPS experiment pipeline: baseline → stage-gated → self-evolution → post-evolution |
| `summarize_results.py` | Summarize and compare benchmark result JSON files |
| `run_neurips_experiments.py` | Programmatic runner for the full NeurIPS ablation matrix |

## Configuration

All scripts support these environment variables:

| Variable | Default | Description |
|---|---|---|
| `GENEVAL2_DATA` | `../GenEval2/geneval2_data.jsonl` | Path to GenEval2 prompt data |
| `OUTPUT_DIR` | varies per script | Output directory for results |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `COMFYUI_ADDR` | `127.0.0.1:8188` | ComfyUI server address |

## Running

```bash
# GEMS vs ComfyClaw comparison (10 prompts)
python experiments/gems_vs_comfyclaw.py

# ComfyClaw-only with DreamShaper (20 prompts)
python experiments/claw_benchmark.py

# ComfyClaw with Qwen Image 2512 (10 prompts)
python experiments/claw_qwen_benchmark.py

# Full NeurIPS experiment
python experiments/full_experiment.py
```

All benchmark scripts support **resume**: completed prompts are cached to disk and skipped on re-run.

## Data

- `stage_map.json` — Auto-discovered node-to-stage mapping from ComfyUI (736 node classes across 7 pipeline stages)
