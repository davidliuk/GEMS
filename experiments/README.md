# Experiments

Benchmark and experiment scripts for evaluating ComfyClaw against baselines.

## File Structure

```
experiments/
├── run_benchmark.py              # Unified entry point (recommended)
├── models/
│   ├── __init__.py               # Auto-discovers and loads YAML configs
│   ├── longcat.yaml              # LongCat Image (BF16) — Flux-based workflow
│   ├── qwen.yaml                 # Qwen Image 2512 (BF16) — AuraFlow-based workflow
│   └── dreamshaper.yaml          # DreamShaper 8 — SD 1.5 single-checkpoint workflow
├── benchmarks/
│   ├── __init__.py               # Auto-discovers and loads YAML configs
│   ├── geneval2.yaml             # GenEval2 (800 prompts, JSONL)
│   ├── dpg-bench.yaml            # DPG-Bench (1,065 prompts, JSONL or .txt dir)
│   ├── oneig-en.yaml             # OneIG-Bench EN (1,120 prompts, JSON)
│   ├── oneig-zh.yaml             # OneIG-Bench ZH (1,320 prompts, JSON)
│   └── wise.yaml                 # WISE (1,000 prompts, JSON or JSONL)
├── gems_vs_comfyclaw.py          # GEMS vs ComfyClaw comparison
├── prepare_sft_data.py           # Post-process SFT traces for fine-tuning
├── full_experiment.py            # Full NeurIPS experiment pipeline
├── summarize_results.py          # Summarize/compare result JSON files
├── run_neurips_experiments.py    # NeurIPS ablation matrix runner
└── README.md
```

To add a new model, drop a `.yaml` file in `models/` — it's auto-discovered by `short_name`.
To add a new benchmark, drop a `.yaml` file in `benchmarks/` — it's auto-discovered by `short_name`.

## Unified Benchmark Runner

The recommended way to run benchmarks is via **`run_benchmark.py`**, which supports
any combination of model and benchmark through CLI flags.

### Quick start

```bash
source .venv/bin/activate

# LongCat on GenEval2, 800 prompts, evolve every 5, 2 parallel
python experiments/run_benchmark.py \
    --model longcat --benchmark geneval2 \
    --n-prompts 800 --max-iterations 5 --evolve-batch-size 5 --parallel 2

# Qwen on DPG-Bench (full 1,065 prompts)
python experiments/run_benchmark.py --model qwen --benchmark dpg-bench

# DreamShaper on WISE
python experiments/run_benchmark.py --model dreamshaper --benchmark wise

# LongCat on OneIG-EN (first 200 prompts)
python experiments/run_benchmark.py --model longcat --benchmark oneig-en --n-prompts 200

# Single custom prompt (no benchmark dataset needed)
python experiments/run_benchmark.py --model longcat --benchmark geneval2 \
    --prompt "a red sports car parked next to a blue bicycle"
```

### Supported models

| Flag | Model | Architecture |
|---|---|---|
| `--model longcat` | LongCat Image (BF16) | Flux-based (UNET + CFGNorm + FluxGuidance) |
| `--model qwen` | Qwen Image 2512 (BF16) | AuraFlow-based (UNET + ModelSamplingAuraFlow) |
| `--model dreamshaper` | DreamShaper 8 | Stable Diffusion 1.5 (single checkpoint) |

### Supported benchmarks

| Flag | Benchmark | Prompts | Source |
|---|---|---|---|
| `--benchmark geneval2` | GenEval2 | 800 | [facebookresearch/GenEval2](https://github.com/facebookresearch/GenEval2) |
| `--benchmark dpg-bench` | DPG-Bench | 1,065 | [Jialuo21/DPG-Bench](https://huggingface.co/datasets/Jialuo21/DPG-Bench) |
| `--benchmark oneig-en` | OneIG-Bench (English) | 1,120 | [OneIG-Bench/OneIG-Bench](https://huggingface.co/datasets/OneIG-Bench/OneIG-Bench) |
| `--benchmark oneig-zh` | OneIG-Bench (Chinese) | 1,320 | [OneIG-Bench/OneIG-Bench](https://huggingface.co/datasets/OneIG-Bench/OneIG-Bench) |
| `--benchmark wise` | WISE | 1,000 | [PKU-YuanGroup/WISE](https://github.com/PKU-YuanGroup/WISE) |

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--model` | *required* | Image generation model (`longcat`, `qwen`, `dreamshaper`) |
| `--benchmark` | *required* | Benchmark dataset (`geneval2`, `dpg-bench`, `oneig-en`, `oneig-zh`, `wise`) |
| `--prompt` | — | Run a single custom prompt instead of the benchmark set |
| `--n-prompts` | benchmark default | Number of prompts to run |
| `--max-iterations` | `5` | Max agent iterations per prompt |
| `--evolve-batch-size` | `0` (disabled) | Evolve skills every N prompts |
| `--parallel` | `1` | Run N prompts concurrently |
| `--no-warm-start` | `false` | Start from empty workflow instead of model's base workflow |
| `--data-path` | auto-detected | Override path to benchmark data file/directory |

### Output directories

The runner automatically creates isolated directories per model+benchmark combination:

```
../benchmark_{model}_{benchmark}/          # results.json + images/
../benchmark_{model}_{benchmark}_detailed/  # per-prompt details, SFT traces
comfyclaw/skills_evolved_benchmark_{model}_{benchmark}/  # evolved skills (isolated per run)
```

For example, `--model longcat --benchmark dpg-bench` produces:
- `../benchmark_longcat_dpg-bench/`
- `../benchmark_longcat_dpg-bench_detailed/`
- `comfyclaw/skills_evolved_benchmark_longcat_dpg-bench/`

Override with `OUTPUT_DIR` / `DETAILED_DIR` environment variables if needed.

### Usage examples

```bash
source .venv/bin/activate

# ── Cross-model × cross-benchmark ─────────────────────────────────────

# Run every model on DPG-Bench
for model in longcat qwen dreamshaper; do
    python experiments/run_benchmark.py --model $model --benchmark dpg-bench --parallel 2
done

# Run LongCat on every benchmark
for bench in geneval2 dpg-bench oneig-en oneig-zh wise; do
    python experiments/run_benchmark.py --model longcat --benchmark $bench --parallel 2
done

# ── Quick smoke test with a single prompt ──────────────────────────────

python experiments/run_benchmark.py --model longcat --benchmark geneval2 \
    --prompt "a cat wearing a top hat sitting on a stack of books"

# ── Run a subset of prompts ────────────────────────────────────────────

python experiments/run_benchmark.py --model qwen --benchmark wise --n-prompts 50

# ── Use a custom data path ─────────────────────────────────────────────

python experiments/run_benchmark.py --model longcat --benchmark dpg-bench \
    --data-path /data/my-dpg-prompts/
```

## Prerequisites

- ComfyUI running and accessible (default: `127.0.0.1:8188`)
- In ComfyUI, do `python main.py --listen 127.0.0.1 --port 8188`
- `ANTHROPIC_API_KEY` environment variable set

### Benchmark data

Clone or download the benchmark datasets you plan to use as sibling directories:

```bash
# GenEval2 (required for --benchmark geneval2)
git clone https://github.com/facebookresearch/GenEval2.git ../GenEval2

# DPG-Bench (required for --benchmark dpg-bench)
# Download from https://huggingface.co/datasets/Jialuo21/DPG-Bench
# or clone the ELLA repo: git clone https://github.com/TencentQQGYLab/ELLA.git ../DPG-Bench

# OneIG-Bench (required for --benchmark oneig-en / oneig-zh)
# Download from https://huggingface.co/datasets/OneIG-Bench/OneIG-Bench

# WISE (required for --benchmark wise)
git clone https://github.com/PKU-YuanGroup/WISE.git ../WISE
```

Override data paths with env vars (`GENEVAL2_DATA`, `DPG_BENCH_DATA`, `ONEIG_EN_DATA`, `ONEIG_ZH_DATA`, `WISE_DATA`) or `--data-path`.

### Model checkpoints

#### Qwen Image (BF16)

```bash
COMFYUI_DIR=/path/to/ComfyUI

hf download Comfy-Org/Qwen-Image_ComfyUI \
  split_files/diffusion_models/qwen_image_bf16.safetensors \
  --local-dir /tmp/qwen-image

hf download Comfy-Org/Qwen-Image_ComfyUI \
  split_files/text_encoders/qwen_2.5_vl_7b.safetensors \
  --local-dir /tmp/qwen-image

hf download Comfy-Org/Qwen-Image_ComfyUI \
  split_files/vae/qwen_image_vae.safetensors \
  --local-dir /tmp/qwen-image

mv /tmp/qwen-image/split_files/diffusion_models/* "$COMFYUI_DIR/models/diffusion_models/"
mv /tmp/qwen-image/split_files/text_encoders/*    "$COMFYUI_DIR/models/text_encoders/"
mv /tmp/qwen-image/split_files/vae/*               "$COMFYUI_DIR/models/vae/"
rm -r /tmp/qwen-image
```

#### LongCat Image (BF16)

```bash
COMFYUI_DIR=/path/to/ComfyUI

# Diffusion model (~12.5 GB)
hf download Comfy-Org/LongCat-Image \
  split_files/diffusion_models/longcat_image_bf16.safetensors \
  --local-dir /tmp/longcat-image

# Text encoder (~16.6 GB) — same Qwen2.5-VL-7B, shared with Qwen Image
hf download Comfy-Org/Qwen-Image_ComfyUI \
  split_files/text_encoders/qwen_2.5_vl_7b.safetensors \
  --local-dir /tmp/longcat-image

# VAE — Flux ae.safetensors
# (download from an ungated source like f5aiteam/ComfyUI or your existing Flux VAE)

mv /tmp/longcat-image/split_files/diffusion_models/* "$COMFYUI_DIR/models/diffusion_models/"
mv /tmp/longcat-image/split_files/text_encoders/*    "$COMFYUI_DIR/models/text_encoders/"
# Place ae.safetensors into $COMFYUI_DIR/models/vae/
rm -r /tmp/longcat-image
```

#### DreamShaper 8

Download `DreamShaper_8_pruned.safetensors` from [Lykon/DreamShaper](https://huggingface.co/Lykon/DreamShaper) into `$COMFYUI_DIR/models/checkpoints/`.

Then restart ComfyUI so it picks up the new models.

## Other Scripts

| Script | Description |
|---|---|
| `gems_vs_comfyclaw.py` | GEMS vs ComfyClaw comparison on 10 GenEval2 prompts |
| `prepare_sft_data.py` | Post-process raw SFT traces into training-ready JSONL |
| `full_experiment.py` | Full NeurIPS experiment pipeline |
| `summarize_results.py` | Summarize and compare benchmark result JSON files |
| `run_neurips_experiments.py` | Programmatic runner for the NeurIPS ablation matrix |

## Configuration

All scripts support these environment variables:

| Variable | Default | Description |
|---|---|---|
| `N_PROMPTS` | benchmark default | Number of prompts |
| `OUTPUT_DIR` | `../benchmark_{model}_{benchmark}` | Output directory for results |
| `DETAILED_DIR` | `../benchmark_{model}_{benchmark}_detailed` | Per-prompt detailed results |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `COMFYUI_ADDR` | `127.0.0.1:8188` | ComfyUI server address |
| `LLM_MODEL` | `anthropic/claude-sonnet-4-5` | LLM for the agent and verifier |
| `GENEVAL2_DATA` | `../GenEval2/geneval2_data.jsonl` | Path to GenEval2 prompt data |
| `DPG_BENCH_DATA` | `../DPG-Bench/prompts.jsonl` | Path to DPG-Bench prompt data |
| `ONEIG_EN_DATA` | `../OneIG-Bench/OneIG-Bench.json` | Path to OneIG-Bench (EN) data |
| `ONEIG_ZH_DATA` | `../OneIG-Bench/OneIG-Bench-ZH.json` | Path to OneIG-Bench (ZH) data |
| `WISE_DATA` | `../WISE/prompts.json` | Path to WISE prompt data |

## Running

```bash
source .venv/bin/activate
python experiments/run_benchmark.py --model longcat --benchmark geneval2
```

All benchmark scripts support **resume**: completed prompts are cached to disk and skipped on re-run.

## SFT Data Collection

The benchmark runner automatically captures full agent conversation traces for supervised fine-tuning. Each trace records the complete multi-turn tool-use conversation (system prompt, user message, every tool call with full arguments, every tool result untruncated) in OpenAI chat format.

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
