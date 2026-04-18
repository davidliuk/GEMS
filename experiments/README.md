# Experiments

Benchmark and experiment scripts for evaluating ComfyClaw against baselines.

## File Structure

```
experiments/
├── run_benchmark.py              # Unified entry point (recommended)
├── launch_comfyui.sh             # Launch multiple ComfyUI instances for GPU parallelism
├── setup.py                      # Auto-download models + benchmark datasets
├── models/
│   ├── __init__.py               # Auto-discovers and loads YAML configs
│   ├── longcat.yaml              # LongCat Image (BF16) — Flux-based workflow
│   ├── qwen.yaml                 # Qwen Image 2512 (BF16) — AuraFlow-based workflow
│   ├── dreamshaper.yaml          # DreamShaper 8 — SD 1.5 single-checkpoint workflow
│   ├── z-image-turbo.yaml        # Z-Image Turbo (BF16) — AuraFlow + zero-out negative
│   └── flux-klein-9b.yaml        # FLUX.2 Klein 9B (BF16) — CFGGuider + Flux2Scheduler
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

COMFYUI_DIR=/path/to/ComfyUI

### Launch comfyUI
bash ./experiments/launch_comfyui.sh 3

python experiments/run_benchmark.py \
    --model longcat --benchmark geneval2 \
    --max-iterations 4 --evolve-batch-size 5 --parallel 5 \
    --comfyui-addrs 127.0.0.1:8189,127.0.0.1:8190

python experiments/run_benchmark.py --model longcat --benchmark geneval2 --n-prompts 800 --max-iterations 4 --evolve-batch-size 5 --parallel 5 --comfyui-addrs 127.0.0.1:8188,127.0.0.1:8189,127.0.0.1:8190

or

python experiments/run_benchmark.py --model longcat --benchmark geneval2 --n-prompts 800 --max-iterations 4 --evolve-batch-size 5 --parallel 5 --comfyui-addrs 127.0.0.1:8188
```

### Run GPT-5.4

```
LLM_MODEL="openai/gpt-5.4" \
LLM_API_KEY="sk-..." \
LLM_API_BASE="https://your-endpoint.openai.azure.com/v1" \

python experiments/run_benchmark.py --model longcat --benchmark geneval2 \
    --max-iterations 4 --evolve-batch-size 5 --parallel 5 \
    --comfyui-addrs 127.0.0.1:8190 \
    --agent-name gpt-5.4
```


### Run baseline model
```
# Claude
python experiments/run_benchmark.py --model longcat --benchmark geneval2 --baseline --parallel 5

# GPT-5.4
LLM_MODEL="openai/gpt-5.4" \
LLM_API_KEY="sk-..." \
LLM_API_BASE="https://your-endpoint.openai.azure.com/v1" \
python experiments/run_benchmark.py --model longcat --benchmark geneval2 --baseline --parallel 5 \
    --agent-name gpt-5.4
```

### Inspect outputs

```
# Follow all 3 logs at once
tail -f /tmp/comfyui_8188.log /tmp/comfyui_8189.log /tmp/comfyui_8190.log
```


### Evaluation Example

```
cd /fs/nexus-scratch/zli12321/self-evolving-UI/GEMS

PYTHONPATH=/fs/nexus-scratch/zli12321/self-evolving-UI/GEMS python eval/GenEval2_data/evaluation.py \
    --benchmark_data eval/GenEval2_data/geneval2_data.jsonl \
    --image_filepath_data /fs/nexus-scratch/zli12321/self-evolving-UI/comfy_agent_baseline_experiments_output/longcat_geneval2/baseline_claude/results/image_paths.json \
    --method soft_tifa_gm \
    --output_file /fs/nexus-scratch/zli12321/self-evolving-UI/comfy_agent_baseline_experiments_output/longcat_geneval2/baseline_claude/results/scores_soft_tifa_gm.json
```


### All model × benchmark commands

| Model | Benchmark | Command |
|---|---|---|
| LongCat | GenEval2 | `python experiments/run_benchmark.py --model longcat --benchmark geneval2` |
| LongCat | DPG-Bench | `python experiments/run_benchmark.py --model longcat --benchmark dpg-bench` |
| LongCat | OneIG-EN | `python experiments/run_benchmark.py --model longcat --benchmark oneig-en` |
| LongCat | OneIG-ZH | `python experiments/run_benchmark.py --model longcat --benchmark oneig-zh` |
| LongCat | WISE | `python experiments/run_benchmark.py --model longcat --benchmark wise` |
| Qwen | GenEval2 | `python experiments/run_benchmark.py --model qwen --benchmark geneval2` |
| Qwen | DPG-Bench | `python experiments/run_benchmark.py --model qwen --benchmark dpg-bench` |
| Qwen | OneIG-EN | `python experiments/run_benchmark.py --model qwen --benchmark oneig-en` |
| Qwen | OneIG-ZH | `python experiments/run_benchmark.py --model qwen --benchmark oneig-zh` |
| Qwen | WISE | `python experiments/run_benchmark.py --model qwen --benchmark wise` |
| Z-Image-Turbo | GenEval2 | `python experiments/run_benchmark.py --model z-image-turbo --benchmark geneval2` |
| Z-Image-Turbo | DPG-Bench | `python experiments/run_benchmark.py --model z-image-turbo --benchmark dpg-bench` |
| Z-Image-Turbo | OneIG-EN | `python experiments/run_benchmark.py --model z-image-turbo --benchmark oneig-en` |
| Z-Image-Turbo | OneIG-ZH | `python experiments/run_benchmark.py --model z-image-turbo --benchmark oneig-zh` |
| Z-Image-Turbo | WISE | `python experiments/run_benchmark.py --model z-image-turbo --benchmark wise` |
| FLUX.2 Klein 9B | GenEval2 | `python experiments/run_benchmark.py --model flux-klein-9b --benchmark geneval2` |
| FLUX.2 Klein 9B | DPG-Bench | `python experiments/run_benchmark.py --model flux-klein-9b --benchmark dpg-bench` |
| FLUX.2 Klein 9B | OneIG-EN | `python experiments/run_benchmark.py --model flux-klein-9b --benchmark oneig-en` |
| FLUX.2 Klein 9B | OneIG-ZH | `python experiments/run_benchmark.py --model flux-klein-9b --benchmark oneig-zh` |
| FLUX.2 Klein 9B | WISE | `python experiments/run_benchmark.py --model flux-klein-9b --benchmark wise` |
| DreamShaper | GenEval2 | `python experiments/run_benchmark.py --model dreamshaper --benchmark geneval2` |
| DreamShaper | DPG-Bench | `python experiments/run_benchmark.py --model dreamshaper --benchmark dpg-bench` |
| DreamShaper | OneIG-EN | `python experiments/run_benchmark.py --model dreamshaper --benchmark oneig-en` |
| DreamShaper | OneIG-ZH | `python experiments/run_benchmark.py --model dreamshaper --benchmark oneig-zh` |
| DreamShaper | WISE | `python experiments/run_benchmark.py --model dreamshaper --benchmark wise` |

Append common flags as needed: `--n-prompts 800 --max-iterations 4 --evolve-batch-size 5 --parallel 5`

Single custom prompt (no benchmark dataset needed):

```bash
python experiments/run_benchmark.py --model longcat --benchmark geneval2 \
    --prompt "a red sports car parked next to a blue bicycle"
```

### Supported models

| Flag | Model | Architecture |
|---|---|---|
| `--model longcat` | LongCat Image (BF16) | Flux-based (UNET + CFGNorm + FluxGuidance) |
| `--model qwen` | Qwen Image 2512 (BF16) | AuraFlow-based (UNET + ModelSamplingAuraFlow) |
| `--model z-image-turbo` | Z-Image Turbo (BF16) | S3-DiT (UNET + AuraFlow), 8-step turbo, zero-out negative (cfg=1, res_multistep) |
| `--model flux-klein-9b` | FLUX.2 Klein 9B (BF16) | Flux2 sampler stack (CFGGuider + Flux2Scheduler + SamplerCustomAdvanced), 4-step turbo |
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
| `--model` | *required* | Image generation model (`longcat`, `qwen`, `z-image-turbo`, `flux-klein-9b`, `dreamshaper`) |
| `--benchmark` | *required* | Benchmark dataset (`geneval2`, `dpg-bench`, `oneig-en`, `oneig-zh`, `wise`) |
| `--prompt` | — | Run a single custom prompt instead of the benchmark set |
| `--n-prompts` | benchmark default | Number of prompts to run |
| `--max-iterations` | `4` | Max agent iterations per prompt |
| `--evolve-batch-size` | `0` (disabled) | Evolve skills every N prompts |
| `--parallel` | `1` | Run N prompts concurrently |
| `--no-warm-start` | `false` | Start from empty workflow instead of model's base workflow |
| `--data-path` | auto-detected | Override path to benchmark data file/directory |
| `--comfyui-addrs` | `127.0.0.1:8188` | Comma-separated ComfyUI addresses for multi-instance parallelism |
| `--agent-name` | auto from `LLM_MODEL` | Agent/LLM name for organizing output folders |

### Output directories

The runner automatically creates isolated directories per model+benchmark+agent combination:

```
experiments_output/                                    # top-level output root
├── {model}_{benchmark}/                               # one folder per experiment
│   ├── {agent_name}/                                  # sub-folder per LLM agent
│   │   ├── results/                                   # results.json + images/
│   │   └── detailed/                                  # per-prompt details, SFT traces
│   └── ...
└── ...

comfyclaw/evolved_skills/                              # evolved skills (inside repo)
├── {model}_{benchmark}/                               # isolated per experiment
│   ├── {agent_name}/                                  # sub-folder per LLM agent
│   │   └── learned-errors/                            # auto-synthesized error skills
│   └── ...
└── ...
```

The `{agent_name}` is auto-derived from `LLM_MODEL` (e.g. `claude-sonnet-4-5`, `gpt-5-4`)
or set explicitly with `--agent-name`.

For example, `--model longcat --benchmark dpg-bench` with `LLM_MODEL=anthropic/claude-sonnet-4-5`:
- `experiments_output/longcat_dpg-bench/claude-sonnet-4-5/results/`
- `experiments_output/longcat_dpg-bench/claude-sonnet-4-5/detailed/`
- `comfyclaw/evolved_skills/longcat_dpg-bench/claude-sonnet-4-5/`

Override the output root with `EXPERIMENTS_ROOT` env var, or individual dirs with `OUTPUT_DIR` / `DETAILED_DIR`.

### Usage examples

```bash
source .venv/bin/activate

# ── Cross-model × cross-benchmark ─────────────────────────────────────

# Run every model on DPG-Bench
for model in z-image-turbo longcat flux-klein-9b qwen dreamshaper; do
    python experiments/run_benchmark.py --model $model --benchmark dpg-bench --parallel 5
done

# Run LongCat on every benchmark
for bench in geneval2 dpg-bench oneig-en oneig-zh wise; do
    python experiments/run_benchmark.py --model longcat --benchmark $bench --parallel 5
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

- ComfyUI running and accessible (default: `127.0.0.1:8188`). For multi-instance
  GPU parallelism, see [Multi-Instance GPU Parallelism](#multi-instance-gpu-parallelism) below.
- In ComfyUI, do `python main.py --listen 127.0.0.1 --port 8188`
- LLM API key set (`LLM_API_KEY` or `ANTHROPIC_API_KEY`)

### Using alternative LLM providers

The benchmark runner routes every LLM call through [litellm](https://docs.litellm.ai/),
so it works with any OpenAI-compatible endpoint — OpenAI, Anthropic, NVIDIA NIM,
Azure OpenAI, Gemini, Perplexity, vLLM servers, etc. Three env vars control it:

| Variable | Purpose | Example |
|---|---|---|
| `LLM_MODEL` | litellm model string | `openai/azure/openai/gpt-5.4` |
| `LLM_API_KEY` | provider API key (falls back to `ANTHROPIC_API_KEY`) | `nvapi-...` |
| `LLM_API_BASE` | base URL for non-default / OpenAI-compatible gateways | `https://inference-api.nvidia.com` |

**Model-string convention.** The prefix before the first `/` tells litellm which
HTTP protocol to speak; everything after it is sent verbatim as the endpoint's
`model` field. So `openai/azure/openai/gpt-5.4` speaks the OpenAI chat protocol
to `LLM_API_BASE` with `model="azure/openai/gpt-5.4"` in the request body.

**Loading.** `run_benchmark.py` auto-loads the repo-root `.env`, so you can put
`LLM_*` there instead of exporting every time. CLI / shell `export`s still win
(the loader uses `override=False`).

```bash
# ── Anthropic (default; no LLM_* vars needed) ──────────────────────────
export ANTHROPIC_API_KEY="sk-ant-..."
python experiments/run_benchmark.py --model longcat --benchmark geneval2

# ── OpenAI directly ────────────────────────────────────────────────────
export LLM_API_KEY="sk-..."
export LLM_MODEL="openai/gpt-4o"
python experiments/run_benchmark.py --model longcat --benchmark geneval2

# ── NVIDIA Inference API (hosts GPT, Claude, Gemini, Llama, …) ─────────
export LLM_API_KEY="nvapi-..."
export LLM_API_BASE="https://inference-api.nvidia.com"
export LLM_MODEL="openai/azure/openai/gpt-5.4"
python experiments/run_benchmark.py --model longcat --benchmark geneval2 \
    --agent-name gpt-5.4

# ── Azure OpenAI (your own deployment) ─────────────────────────────────
export LLM_API_KEY="<azure-key>"
export LLM_API_BASE="https://<resource>.openai.azure.com"
export LLM_MODEL="azure/<deployment-name>"
export AZURE_API_VERSION="2024-08-01-preview"
python experiments/run_benchmark.py --model longcat --benchmark geneval2
```

**NVIDIA model catalog.** The same NVIDIA key usually gives access to multiple
providers. Useful `LLM_MODEL` values:

| Provider | `LLM_MODEL` value | Notes |
|---|---|---|
| OpenAI GPT | `openai/azure/openai/gpt-5.4` | Flagship, recommended for agent |
| OpenAI GPT | `openai/azure/openai/gpt-4o` | Faster / cheaper |
| OpenAI GPT | `openai/azure/openai/gpt-4.1` | |
| OpenAI reasoning | `openai/azure/openai/o3` | Slow but strong |
| Anthropic | `openai/azure/anthropic/claude-sonnet-4-5` | Claude via NVIDIA |
| Open weights | `openai/nvcf/meta/llama-3.3-70b-instruct` | Free tier |
| Open weights | `openai/nvcf/openai/gpt-oss-120b` | Free tier |

List the full catalog (and check which ones your key is actually allowed to use):

```bash
# Enumerate every model the endpoint knows about
curl -sS https://inference-api.nvidia.com/v1/models \
  -H "Authorization: Bearer $LLM_API_KEY" | jq -r '.data[].id'

# Probe one specific model (success = 200, blocked = 401 key_model_access_denied)
curl -sS -X POST https://inference-api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $LLM_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"azure/openai/gpt-5.4","messages":[{"role":"user","content":"ping"}],"max_tokens":8}'
```

If you see `401 key_model_access_denied: key can only access models=['default-models']`,
your key is on the free/default tier — use one of the `nvcf/…` models above, or
request access to the gated ones.

**Output organisation.** The `--agent-name` flag (or the auto-slug derived from
`LLM_MODEL`) determines the output subfolder, so runs with different agents stay
separate:

```
experiments_output/longcat_geneval2/
├── claude-sonnet-4-5/    # default agent
├── gpt-5.4/              # --agent-name gpt-5.4
└── gpt-4o/               # auto-derived from openai/gpt-4o
```

**Retry / timeout tuning.** Useful when a hosted endpoint is flaky:

```bash
export LLM_NUM_RETRIES=5           # default 3
export LLM_REQUEST_TIMEOUT=600     # seconds, default 300
```

### Automated setup (recommended)

Use `setup.py` to download model checkpoints and benchmark datasets automatically:

```bash
source .venv/bin/activate

# Download everything (all models + all benchmarks)
python experiments/setup.py --all -comfyui-dir /path/to/your/ComfyUI

# Download only specific models and benchmarks
python experiments/setup.py --model longcat --model qwen --benchmark geneval2

# Just models (no benchmarks)
python experiments/setup.py --models-only

# Just benchmarks (no models)
python experiments/setup.py --benchmarks-only

# Preview what would be downloaded (no actual downloads)
python experiments/setup.py --dry-run --all

# Custom ComfyUI path
python experiments/setup.py --all --comfyui-dir /path/to/ComfyUI
```

The script skips files that already exist, so it's safe to re-run. Shared files (e.g. `qwen_2.5_vl_7b.safetensors`, `ae.safetensors`) are downloaded once.

After downloading models, restart ComfyUI so it picks up the new files.

### Manual setup (alternative)

<details>
<summary>Click to expand manual download instructions</summary>

#### Benchmark data

```bash
git clone https://github.com/facebookresearch/GenEval2.git ../GenEval2
git clone https://github.com/PKU-YuanGroup/WISE.git ../WISE
# DPG-Bench: download from https://huggingface.co/datasets/Jialuo21/DPG-Bench
# OneIG-Bench: download from https://huggingface.co/datasets/OneIG-Bench/OneIG-Bench
```

Override data paths with env vars (`GENEVAL2_DATA`, `DPG_BENCH_DATA`, `ONEIG_EN_DATA`, `ONEIG_ZH_DATA`, `WISE_DATA`) or `--data-path`.

#### Model checkpoints

```bash
COMFYUI_DIR=/path/to/ComfyUI

# Qwen Image
hf download Comfy-Org/Qwen-Image_ComfyUI split_files/diffusion_models/qwen_image_bf16.safetensors --local-dir /tmp/qwen && mv /tmp/qwen/split_files/diffusion_models/* "$COMFYUI_DIR/models/diffusion_models/"
hf download Comfy-Org/Qwen-Image_ComfyUI split_files/text_encoders/qwen_2.5_vl_7b.safetensors --local-dir /tmp/qwen && mv /tmp/qwen/split_files/text_encoders/* "$COMFYUI_DIR/models/text_encoders/"
hf download Comfy-Org/Qwen-Image_ComfyUI split_files/vae/qwen_image_vae.safetensors --local-dir /tmp/qwen && mv /tmp/qwen/split_files/vae/* "$COMFYUI_DIR/models/vae/"

# LongCat Image
hf download Comfy-Org/LongCat-Image split_files/diffusion_models/longcat_image_bf16.safetensors --local-dir /tmp/longcat && mv /tmp/longcat/split_files/diffusion_models/* "$COMFYUI_DIR/models/diffusion_models/"

# Z-Image-Turbo
hf download Comfy-Org/z_image_turbo split_files/diffusion_models/z_image_turbo_bf16.safetensors --local-dir /tmp/zimage && mv /tmp/zimage/split_files/diffusion_models/* "$COMFYUI_DIR/models/diffusion_models/"
hf download Comfy-Org/z_image_turbo split_files/text_encoders/qwen_3_4b.safetensors --local-dir /tmp/zimage && mv /tmp/zimage/split_files/text_encoders/* "$COMFYUI_DIR/models/text_encoders/"

# FLUX.2 Klein 9B
hf download Comfy-Org/flux2-klein_ComfyUI split_files/diffusion_models/flux-2-klein-9b.safetensors --local-dir /tmp/flux-klein && mv /tmp/flux-klein/split_files/diffusion_models/* "$COMFYUI_DIR/models/diffusion_models/"
hf download Comfy-Org/flux2-klein_ComfyUI split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors --local-dir /tmp/flux-klein && mv /tmp/flux-klein/split_files/text_encoders/* "$COMFYUI_DIR/models/text_encoders/"
hf download Comfy-Org/flux2-klein_ComfyUI split_files/vae/flux2-vae.safetensors --local-dir /tmp/flux-klein && mv /tmp/flux-klein/split_files/vae/* "$COMFYUI_DIR/models/vae/"

# DreamShaper 8
hf download Lykon/DreamShaper DreamShaper_8_pruned.safetensors --local-dir "$COMFYUI_DIR/models/checkpoints/"

# Shared VAE (ae.safetensors) — used by LongCat and Z-Image-Turbo
# Download from f5aiteam/ComfyUI or any ungated Flux VAE source
```

</details>

Then restart ComfyUI so it picks up the new models.

## Multi-Instance GPU Parallelism

By default, all parallel threads share one ComfyUI instance, so image generation
is serialized on the GPU even when `--parallel` is set. If your GPU has enough VRAM,
you can launch multiple ComfyUI instances to generate images in true parallel.

### Memory budget

Each model instance uses roughly:

| Model | VRAM per instance |
|---|---|
| LongCat Image (BF16) | ~27 GB |
| Qwen Image (BF16) | ~25 GB |
| FLUX.2 Klein 9B (BF16) | ~28 GB |
| Z-Image-Turbo (BF16) | ~22 GB |
| DreamShaper 8 (SD 1.5) | ~4 GB |

For example, on a 96 GB GPU you can run 3 LongCat instances (3 × 27 GB = 81 GB).

### Launching multiple instances

Use the provided launch script:

```bash
# Launch 3 ComfyUI instances on ports 8188, 8189, 8190
./experiments/launch_comfyui.sh 3

# Custom base port
./experiments/launch_comfyui.sh 3 8200

# Custom ComfyUI path
COMFYUI_DIR=/path/to/ComfyUI ./experiments/launch_comfyui.sh 3
```

The script waits for all instances to be ready and prints the `COMFYUI_ADDRS` line to use.

### Running benchmarks with multiple instances

Pass the addresses via `--comfyui-addrs` or the `COMFYUI_ADDRS` env var. Work is
distributed round-robin across instances.

```bash
# Via CLI flag
python experiments/run_benchmark.py --model longcat --benchmark dpg-bench \
    --n-prompts 800 --max-iterations 4 --parallel 3 \
    --comfyui-addrs 127.0.0.1:8188,127.0.0.1:8189,127.0.0.1:8190

# Via env var
export COMFYUI_ADDRS=127.0.0.1:8188,127.0.0.1:8189,127.0.0.1:8190
python experiments/run_benchmark.py --model longcat --benchmark dpg-bench --parallel 3
```

### Running multiple benchmarks concurrently

When running two benchmarks at the same time, dedicate separate instances to each
run so they don't contend for the same GPU queue:

```bash
# Terminal 1 — dpg-bench on instance :8188
python experiments/run_benchmark.py --model longcat --benchmark dpg-bench \
    --parallel 1 --comfyui-addrs 127.0.0.1:8188

# Terminal 2 — geneval2 on instances :8189 and :8190
python experiments/run_benchmark.py --model longcat --benchmark geneval2 \
    --parallel 2 --comfyui-addrs 127.0.0.1:8189,127.0.0.1:8190
```

The legacy `COMFYUI_ADDR` env var (single address) is still supported for backwards
compatibility.

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
| `LLM_API_KEY` | — | API key for the LLM provider (falls back to `ANTHROPIC_API_KEY`) |
| `LLM_API_BASE` | — | Base URL for OpenAI-compatible LLM endpoints (e.g. `https://inference-api.nvidia.com`) |
| `LLM_MODEL` | `anthropic/claude-sonnet-4-5` | LLM model name (litellm format, e.g. `openai/gpt-4o`, `azure/openai/gpt-5.4`) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (legacy, prefer `LLM_API_KEY`) |
| `COMFYUI_ADDR` | `127.0.0.1:8188` | ComfyUI server address (single instance, legacy) |
| `COMFYUI_ADDRS` | — | Comma-separated ComfyUI addresses for multi-instance parallelism |
| `LLM_NUM_RETRIES` | `3` | Number of retries on rate-limit or timeout errors |
| `LLM_REQUEST_TIMEOUT` | `300` | Request timeout in seconds per LLM call |
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
