# Experiments

Benchmark and experiment scripts for evaluating ComfyClaw against baselines.

## Prerequisites

- ComfyUI running and accessible (default: `127.0.0.1:8188`)
- In ComfyUI, do `python main.py --listen 127.0.0.1 --port 8188`
- `ANTHROPIC_API_KEY` environment variable set
- GenEval2 data (`geneval2_data.jsonl`) — override path with `GENEVAL2_DATA` env var
- Model checkpoints installed in ComfyUI's `models/` directories
- Run experiment `N_PROMPTS=800 python experiments/claw_qwen_benchmark.py --max-iterations 5 --evolve-batch-size 5 --parallel 2`

## Scripts

| Script | Description |
|---|---|
| `gems_vs_comfyclaw.py` | Head-to-head comparison: GEMS (prompt-only refinement) vs ComfyClaw (topology evolution) on 10 GenEval2 prompts |
| `claw_benchmark.py` | ComfyClaw-only benchmark — 20 GenEval2 prompts with DreamShaper, warm-start + baseline_first |
| `claw_qwen_benchmark.py` | ComfyClaw benchmark with Qwen Image 2512 as base model (10 prompts) |
| `prepare_sft_data.py` | Post-process raw SFT traces into training-ready JSONL with inlined skills |
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
| `DETAILED_DIR` | `../benchmark_qwen_detailed` | Per-prompt detailed results (images, workflows, SFT traces) |
| `EVOLVE_BATCH_SIZE` | `0` | Evolve skills every N prompts (0 = disabled) |
| `PARALLEL_SIZE` | `1` | Run N prompts concurrently (1 = sequential) |

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

## SFT Data Collection

The Qwen benchmark (`claw_qwen_benchmark.py`) automatically captures full agent conversation traces for supervised fine-tuning. Each trace records the complete multi-turn tool-use conversation (system prompt, user message, every tool call with full arguments, every tool result untruncated) in OpenAI chat format.

**Raw traces** are saved automatically during benchmark runs:

- Per-prompt: `<DETAILED_DIR>/prompt_XX_<slug>/sft_traces.jsonl`
- Aggregated: `<OUTPUT_DIR>/sft_traces_all.jsonl`

**Post-processing** with `prepare_sft_data.py` converts raw traces into training-ready data by inlining skill content into the system prompt (so the SFT model learns the knowledge directly rather than learning to call `read_skill`):

```bash
# Basic — process all traces, inline skills
python experiments/prepare_sft_data.py \
    --input  ../benchmark_qwen_10/sft_traces_all.jsonl \
    --output ../benchmark_qwen_10/sft_training.jsonl

# Only keep traces that scored well (quality filter)
python experiments/prepare_sft_data.py \
    --input  ../benchmark_qwen_10/sft_traces_all.jsonl \
    --output ../benchmark_qwen_10/sft_training.jsonl \
    --min-score 0.5

# Include repair traces too (for teaching error recovery)
python experiments/prepare_sft_data.py \
    --input  ../benchmark_qwen_10/sft_traces_all.jsonl \
    --output ../benchmark_qwen_10/sft_training.jsonl \
    --include-repairs

# Process from per-prompt directories instead
python experiments/prepare_sft_data.py \
    --input-dir ../benchmark_qwen_detailed \
    --output    ../benchmark_qwen_10/sft_training.jsonl
```

Each output line is a JSON object with `messages` (OpenAI chat format) and `metadata` (prompt, score, iteration, etc.). Compatible with `trl`, `axolotl`, or OpenAI fine-tuning API.

## Data

- `stage_map.json` — Auto-discovered node-to-stage mapping from ComfyUI (736 node classes across 7 pipeline stages)
