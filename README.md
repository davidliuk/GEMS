# ComfyClaw

[![CI](https://github.com/davidliuk/comfyclaw/actions/workflows/ci.yml/badge.svg)](https://github.com/davidliuk/comfyclaw/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Agentic harness for self-evolving ComfyUI image-generation workflows.**

ComfyClaw wraps ComfyUI in an LLM agent loop that _grows_ the workflow
topology in response to image quality feedback — adding LoRA loaders,
ControlNet branches, regional conditioning, and hires-fix passes until a
configurable quality threshold is reached.

Use **any LLM provider** supported by [LiteLLM](https://docs.litellm.ai/docs/providers):
Anthropic Claude, OpenAI GPT-4o, Google Gemini, local Ollama models, and 100+
more — including mixing providers (e.g. a fast local model for the agent,
a vision-capable cloud model for the verifier).

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ClawHarness loop                            │
│                                                                     │
│  ┌──────────┐  evolve  ┌──────────────┐  submit  ┌──────────────┐  │
│  │  Agent   │ ───────► │  ComfyUI API │ ───────► │  Image out   │  │
│  │  Agent   │          │  (HTTP/WS)   │          └──────┬───────┘  │
│  │  (LLM)   │          └──────────────┘                 │          │
│  └────┬─────┘                                           │ verify   │
│       │  feedback                                        │          │
│       │ ◄────────────────────────────────────────────── ▼          │
│       │                                          ┌──────────────┐  │
│       │                                          │  Verifier    │  │
│  ┌────┴─────┐  broadcast  ┌──────────────────┐  │  (LLM+vis)   │  │
│  │  Memory  │             │ ComfyClaw-Sync    │  └──────────────┘  │
│  │          │             │ (ComfyUI plugin)  │                    │
│  └──────────┘             │ live graph update │                    │
│                           └──────────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.10+
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) (Desktop or server)
- An API key for your chosen LLM provider (see [Supported providers](#supported-llm-providers))

### 1. Install uv (recommended)

[uv](https://docs.astral.sh/uv/) is the fastest way to manage Python
environments and installs.

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or Homebrew
brew install uv
```

### 2. Install ComfyClaw

```bash
# Clone the repo and install in editable mode (for development)
git clone https://github.com/davidliuk/comfyclaw.git
cd comfyclaw
uv sync --group dev          # installs all dev + runtime deps

# Or install just the runtime (no dev tools)
uv sync

# Or install from PyPI (once published)
uv add comfyclaw             # adds to your project
uv tool install comfyclaw    # installs comfyclaw CLI globally
```

**Pip alternative** (without uv):

```bash
pip install -e ".[sync]"          # from a local clone
pip install "comfyclaw[sync]"     # from PyPI
```

**Dependency extras:**

| Extra | Packages | When needed |
|---|---|---|
| *(none)* | `litellm`, `python-dotenv` | Always |
| `sync` | `websockets>=12` | Live graph updates in ComfyUI canvas |
| `providers` | `anthropic>=0.25` | Direct Anthropic SDK (optional; litellm handles it) |
| `dev` (group) | `pytest`, `ruff`, `mypy`, … | Development & CI |

### 3. Configure environment

```bash
cp .env.example .env
# Open .env and set at minimum:
#   ANTHROPIC_API_KEY=sk-ant-...   (for Anthropic — default provider)
#   OPENAI_API_KEY=sk-...           (for OpenAI)
#   GEMINI_API_KEY=...              (for Google Gemini)
#   (no key needed for local Ollama)
#   COMFYUI_ADDR=127.0.0.1:8188
```

All CLI flags can also be set as environment variables (see [Environment
variables](#environment-variables)).  `.env` is auto-loaded at startup.

### 2. Install the ComfyUI live-sync plugin

The plugin is bundled inside the package.  Install it once, then restart ComfyUI.

```bash
# Automatic (recommended)
comfyclaw install-node

# With an explicit ComfyUI path
uv run comfyclaw install-node --comfyui-dir /fs/nexus-scratch/zli12321/comfy-testing/ComfyUI

# Via environment variable
export COMFYUI_DIR=~/Documents/ComfyUI
comfyclaw install-node

# Find the node source path (for manual copy)
comfyclaw node-path
```

Manual alternative:

```bash
# Symlink (any edits to the package take effect immediately)
ln -s "$(comfyclaw node-path)" ~/Documents/ComfyUI/custom_nodes/ComfyClaw-Sync

# Or copy
cp -r "$(comfyclaw node-path)" ~/Documents/ComfyUI/custom_nodes/ComfyClaw-Sync
```

After installation a status badge appears in the bottom-right corner of the
ComfyUI canvas:
- 🔄 connecting — waiting for the Python sync server
- 🟢 live — connected, ready to receive updates
- ✨ graph updated — agent just pushed a new workflow
- 🔴 disconnected — sync server not running (click to reconfigure URL)

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | API key for Anthropic models (`anthropic/…`) |
| `OPENAI_API_KEY` | — | API key for OpenAI models (`openai/…`) |
| `GEMINI_API_KEY` | — | API key for Google Gemini models (`gemini/…`) |
| `GROQ_API_KEY` | — | API key for Groq models (`groq/…`) |
| `COMFYUI_DIR` | `~/Documents/ComfyUI` | Path to your ComfyUI installation |
| `COMFYUI_ADDR` | `127.0.0.1:8188` | `host:port` of a running ComfyUI server |
| `COMFYCLAW_MODEL` | `anthropic/claude-sonnet-4-5` | LiteLLM model string for the agent |
| `COMFYCLAW_VERIFIER_MODEL` | *(same as model)* | LiteLLM model for the vision verifier |

---

## Supported LLM providers

ComfyClaw uses [LiteLLM](https://docs.litellm.ai/docs/providers) to route to any
provider.  Set the matching environment variable and pass the model string with
the provider prefix:

| Provider | Model string format | Required env var |
|---|---|---|
| **Anthropic** (default) | `anthropic/claude-sonnet-4-5` | `ANTHROPIC_API_KEY` |
| **OpenAI** | `openai/gpt-4o` | `OPENAI_API_KEY` |
| **Google Gemini** | `gemini/gemini-2.0-flash` | `GEMINI_API_KEY` |
| **Groq** | `groq/llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| **Azure OpenAI** | `azure/<deployment-name>` | `AZURE_API_KEY` + `AZURE_API_BASE` |
| **Local Ollama** | `ollama/llama3.1` | *(none — no API key needed)* |
| **Local vLLM** | `openai/my-model` + `--base-url` | *(none)* |

```bash
# Anthropic (default)
export ANTHROPIC_API_KEY=sk-ant-...
comfyclaw run --model anthropic/claude-sonnet-4-5 --workflow w.json --prompt "..."

# OpenAI
export OPENAI_API_KEY=sk-...
comfyclaw run --model openai/gpt-4o --workflow w.json --prompt "..."

# Google Gemini
export GEMINI_API_KEY=...
comfyclaw run --model gemini/gemini-2.0-flash --workflow w.json --prompt "..."

# Fully local — Ollama agent, LLaVA verifier (no API key required)
comfyclaw run --model ollama/llama3.1 --verifier-model ollama/llava \
  --workflow w.json --prompt "..."

# Mix providers: cheap local agent + strong cloud verifier
export ANTHROPIC_API_KEY=sk-ant-...
comfyclaw run --model ollama/llama3.1 \
             --verifier-model anthropic/claude-sonnet-4-5 \
             --workflow w.json --prompt "..."
```

> **Vision requirement**: the `--verifier-model` must support image inputs.
> Good choices: `anthropic/claude-*`, `openai/gpt-4o`, `gemini/gemini-*`, `ollama/llava`.

---

## Quick start

```bash
# 1. Set your provider API key (or put it in .env)
export ANTHROPIC_API_KEY=sk-ant-...    # Anthropic (default)
# export OPENAI_API_KEY=sk-...           # OpenAI alternative
# export GEMINI_API_KEY=...              # Gemini alternative
# (no key needed for local Ollama)

# 2. Install the ComfyUI live-sync plugin (one-time)
uv run comfyclaw install-node

# 3. Run — full agent–generate–verify loop
uv run comfyclaw run \
  --workflow my_workflow_api.json \
  --prompt "a red fox at dawn, photorealistic, DSLR"

# 4. Dry-run — agent patches workflow, skips ComfyUI (no GPU needed)
uv run comfyclaw dry-run \
  --workflow my_workflow_api.json \
  --prompt "a red fox at dawn"
```

If you installed via `uv tool install comfyclaw` or activated the venv, drop
the `uv run` prefix.

Or via `python3 run_comfy_claw.py` at the repo root (no install needed):

```bash
python3 run_comfy_claw.py --prompt "a red fox at dawn" --iterations 3
python3 run_comfy_claw.py --dry-run
```

---

## Sample runs

These two runs use the same `qwen_workflow_api.json` base workflow and Qwen-Image-2512
to illustrate how agent quality affects the outcome.

---

### Case 1 — Claude Sonnet 4.5 · Wildlife photography

```bash
uv run comfyclaw run \
  --workflow qwen_workflow_api.json \
  --prompt "A majestic red fox sitting in a misty ancient forest at dawn, photorealistic wildlife photography" \
  --iterations 2
```

**What the agent did (iteration 1):**

1. Called `read_skill("qwen-image-2512")` → learned model-specific parameters
2. Called `read_skill("photorealistic")` and `read_skill("high-quality")`
3. Called `set_prompt` with an expanded prompt:
   - Rich spatial composition (fox on mossy log, ferns foreground, old-growth trees behind)
   - Lighting details: golden volumetric dawn light, rim lighting, atmospheric mist
   - Technical photography style: 300 mm f/2.8, shallow DoF, National Geographic aesthetic
   - Chinese-language negative prompt (Qwen understands both languages)
4. Set resolution to **1472 × 1104** (Qwen's 4:3 Lightning bucket)
5. Confirmed optimal Lightning settings: `steps=4, cfg=1.0, euler/simple`

**Result:**

| Metric | Value |
|--------|-------|
| Score | **0.89 / 1.00** |
| Threshold | 0.85 |
| Outcome | ✅ Passed — **stopped early after iteration 1** |
| Passed checks | Fox present, red colour, sitting pose, forest, ancient trees, mist, dawn lighting, photorealistic, majestic posture |
| Failed checks | "Wildlife photography style" (minor stylistic gap) |

**Verifier summary:** *"The image is a high-quality, photorealistic rendering of a red fox in a forest environment with excellent lighting, composition, and atmospheric effects."*

---

### Case 2 — Ollama Gemma4 (e4b) · Cyberpunk city

```bash
uv run comfyclaw run \
  --workflow qwen_workflow_api.json \
  --model ollama/gemma4:e4b \
  --verifier-model ollama/gemma4:e4b \
  --image-model "qwen_image_2512_fp8_e4m3fn.safetensors" \
  --iterations 3 \
  --prompt "a futuristic cyberpunk city skyline at night, neon lights, rain, 8k"
```

**What the agent did:**

Gemma4 correctly *planned* multi-step evolutions (ControlNet Canny, hires-fix, prompt
overhaul) but did not reliably execute the corresponding tool calls — `nodes 11→11 (+0)`
in all three iterations. The harness still seeded the correct user prompt before each
iteration via `inject_prompt`, so images were generated with the right subject.

**Result:**

| Iteration | Score | Key verifier notes |
|-----------|-------|--------------------|
| 1 | 0.36 | Cyberpunk atmosphere present; missing neon signs, wet reflections, night lighting |
| 2 | 0.36 | Same structural gap; agent planned ControlNet but did not execute |
| 3 | **0.49** | Rain and reflections now passed; architecture/neon still weak |

**Verifier summary (iter 3):** *"Highly atmospheric piece, perfectly capturing the
cyberpunk aesthetic. Lighting, scale, and neon saturation are excellent."*

**Takeaway:** Gemma4 is a capable *vision verifier* but its tool-call execution is
less reliable than Claude for complex multi-step workflow modifications.  Use a larger
or more capable model for the agent role when precise tool use is required, and
reserve Gemma4 (or another local model) for the `--verifier-model` slot.

---

## CLI reference

```
comfyclaw run           Run the full agent–generate–verify loop
comfyclaw dry-run       Run the agent only (no ComfyUI execution needed)
comfyclaw install-node  Symlink the ComfyClaw-Sync plugin into ComfyUI
comfyclaw node-path     Print the path to the bundled plugin directory
```

### `comfyclaw run` / `comfyclaw dry-run` options

| Flag | Default | Description |
|---|---|---|
| `--workflow PATH` | *(required)* | API-format ComfyUI workflow JSON |
| `--prompt TEXT` | *(required)* | Image generation prompt |
| `--model MODEL` | `anthropic/claude-sonnet-4-5` | LiteLLM model for the agent (e.g. `openai/gpt-4o`, `gemini/gemini-2.0-flash`, `ollama/llama3.1`) |
| `--verifier-model MODEL` | *(same as --model)* | LiteLLM model for the vision verifier (must support images) |
| `--iterations N` | `3` | Maximum agent–generate–verify cycles |
| `--threshold SCORE` | `0.85` | Stop early when verifier score ≥ this value |
| `--sync-port PORT` | `8765` | WebSocket port for live graph sync |
| `--no-sync` | off | Disable live sync (faster; ComfyUI not needed) |
| `--skills-dir DIR` | *(built-in)* | Custom directory of skill SKILL.md files |
| `--reset-each-iter` | off | Reset to base workflow each iteration |
| `--output-dir DIR` | `./comfyclaw_output/` | Where to save the best image |

---

## Python API

### Minimal usage

```python
from comfyclaw import ClawHarness, HarnessConfig

cfg = HarnessConfig(
    # api_key="sk-ant-..."  # optional: or set ANTHROPIC_API_KEY env var
    server_address="127.0.0.1:8188",
    model="anthropic/claude-sonnet-4-5",  # any LiteLLM model string
    max_iterations=3,
    success_threshold=0.85,
)

with ClawHarness.from_workflow_file("workflow_api.json", cfg) as h:
    image_bytes = h.run("a red fox at dawn, photorealistic")

if image_bytes:
    open("output.png", "wb").write(image_bytes)
```

### HarnessConfig fields

```python
@dataclass
class HarnessConfig:
    api_key: str = ""                  # optional; or set provider env var
    server_address: str = "127.0.0.1:8188"
    model: str = "anthropic/claude-sonnet-4-5"  # any LiteLLM model string
    verifier_model: str | None = None  # None = use model; set for vision-specific model
    max_iterations: int = 3
    success_threshold: float = 0.85    # stop early when score ≥ this
    sync_port: int = 8765              # 0 = disable live sync
    skills_dir: str | None = None      # None = use built-in skills
    evolve_from_best: bool = True      # accumulate topology across iters
    max_images: int = 5                # images kept in RAM per run
    score_weights: tuple = (0.6, 0.4) # (requirement, detail) score blend
```

### Topology accumulation

When `evolve_from_best=True` (the default), each iteration starts from the
**best workflow snapshot** produced so far, not the original base.  This means
LoRA nodes added in round 1 persist into round 2, and the agent only adds
incremental improvements.

```
Iteration 1:  base(3 nodes) → agent adds LoRA → 4 nodes   score=0.62
Iteration 2:  starts from 4-node snapshot → agent adds ControlNet → 6 nodes  score=0.81
Iteration 3:  starts from 6-node snapshot → agent adds hires-fix → 8 nodes   score=0.91 ✅
```

Set `evolve_from_best=False` to reset to `base_workflow` each iteration.

### Using WorkflowManager directly

```python
from comfyclaw.workflow import WorkflowManager

wm = WorkflowManager.from_file("workflow_api.json")

# Inspect
print(wm)                                 # repr with node count
print(WorkflowManager.summarize(wm.workflow))  # human-readable table

# Validate
errors = wm.validate()                    # list of dangling link messages

# Modify
nid = wm.add_node("LoraLoader", nickname="My LoRA",
                  lora_name="detail.safetensors",
                  strength_model=0.8, strength_clip=0.8)
wm.connect("1", 0, nid, "model")         # wire checkpoint → LoRA
wm.set_param("3", "steps", 30)           # update KSampler steps
wm.delete_node("2")                      # remove a node

clone = wm.clone()                       # independent deep copy
```

### Using ComfyClient directly

```python
from comfyclaw.client import ComfyClient

client = ComfyClient("127.0.0.1:8188")
print(client.is_alive())                 # True / False

resp   = client.queue_prompt(wm.workflow)
entry  = client.wait_for_completion(resp["prompt_id"], timeout=300)
images = client.collect_images(entry)    # list[bytes]

# Query available models
info = client.object_info("LoraLoader")
loras = info["LoraLoader"]["input"]["required"]["lora_name"][0]
```

---

## How the live sync works

```
Python process                    Browser (ComfyUI)
──────────────────────────────    ─────────────────────────────────
SyncServer (ws://localhost:8765)  ComfyClaw-Sync extension
     │                                     │
     │  { type: "workflow_update",         │
     │    workflow: { "1": {...}, ... } }   │
     │ ──────────────────────────────────► │
     │                                     │
     │                             loadWorkflowIntoCanvas()
     │                               → app.loadApiJson()   (ComfyUI ≥ 0.2)
     │                               → app.loadGraphData() (older versions)
     │                               → app.graph.configure()  (fallback)
```

Every time the agent calls a tool that mutates the workflow (e.g.
`add_lora_loader`, `set_param`, `connect_nodes`), the harness broadcasts the
updated workflow dict over the WebSocket.  The JS extension receives it,
converts it to LiteGraph format if needed, and reloads the canvas — so you
watch the graph evolve in real time as the agent works.

The sync server runs in a background daemon thread, so it never blocks the
main generation loop.

---

## Agent tools

The LLM agent has 14 tools grouped by category:

| Category | Tool | What it does |
|---|---|---|
| Inspect | `inspect_workflow` | Show all nodes and connections as text |
| Inspect | `query_available_models` | List installed LoRA / ControlNet / checkpoint files |
| Basic | `set_param` | Set a scalar input on a node |
| Basic | `add_node` | Append a new node, returns its ID |
| Basic | `connect_nodes` | Wire one node's output to another's input |
| Basic | `delete_node` | Remove a node and clean up dangling links |
| LoRA | `add_lora_loader` | Insert LoraLoader and re-wire model/CLIP consumers |
| ControlNet | `add_controlnet` | Add ControlNetLoader + preprocessor + Apply node |
| Regional | `add_regional_attention` | Split conditioning into foreground/background regions |
| Refinement | `add_hires_fix` | Add LatentUpscaleBy + second KSampler + VAEDecode |
| Refinement | `add_inpaint_pass` | Add targeted inpaint stage for a specific region |
| Skills | `read_skill` | Load full instructions for a named skill on demand |
| Control | `report_evolution_strategy` | Declare plan before making changes |
| Control | `finalize_workflow` | Signal completion and return rationale |

### Agent decision heuristics

The agent is instructed to load a skill's full instructions before applying a technique,
then choose tools based on verifier feedback:

| Verifier finding | Skill to read | Primary tool |
|---|---|---|
| Flat / low-depth background | `controlnet-control` | `add_controlnet` (depth) |
| Blurry edges / lost structure | `controlnet-control` | `add_controlnet` (canny) |
| Wrong human pose / body | `controlnet-control` | `add_controlnet` (pose) |
| Plasticky skin / poor texture | `lora-enhancement` | `add_lora_loader` (detail) |
| Wrong anatomy (hands, fingers) | `lora-enhancement` | `add_lora_loader` (anatomy) |
| Style inconsistency | `lora-enhancement` | `add_lora_loader` (style) |
| Subject and background bleed | `regional-control` | `add_regional_attention` |
| Low resolution / soft fine detail | `hires-fix` | `add_hires_fix` |
| Localised artifact in one area | — | `add_inpaint_pass` |

---

## Skills

ComfyClaw's skills system follows the [Anthropic Agent Skills specification](https://agentskills.dev/specification).
Skills are directories containing a `SKILL.md` file with YAML frontmatter.

**Progressive disclosure** keeps the agent's context lean:

1. **Startup (stage 1)** — only `name` and `description` from each skill's frontmatter
   are surfaced in an `<available_skills>` XML block in the system prompt.
2. **On demand (stage 2)** — when the agent decides to apply a skill it calls
   `read_skill("<name>")`, which loads the full instruction body at that moment.

This means 11 skills can be registered without flooding the context window on every
iteration.

### Built-in skills

| Skill | When the agent activates it |
|---|---|
| `high-quality` | User asks for "high quality", "sharp", "detailed", "8K" |
| `photorealistic` | "photo", "DSLR", "realistic", "cinematic", "RAW" |
| `creative` | "creative", "artistic", "fantasy", "concept art", "surreal" |
| `aesthetic-drawing` | "aesthetic drawing", "masterpiece", "award-winning", "professional art" |
| `creative-drawing` | "cool", "dreamy", "futuristic", "artistic" (prompt upgrade) |
| `lora-enhancement` | Verifier reports texture/lighting/anatomy defects |
| `controlnet-control` | Verifier reports flat background, blurry edges, wrong pose |
| `regional-control` | Subject and background style bleed |
| `hires-fix` | Blurry output, soft detail, base resolution ≤ 768 |
| `spatial` | Prompt involves multiple objects with spatial relationships |
| `text-rendering` | Prompt contains quoted text, "a sign saying…", labels |

### Adding custom skills

Skills follow the [Agent Skills spec](https://agentskills.dev/specification) — a
directory whose name is the skill's `name`, containing a `SKILL.md` with YAML
frontmatter:

```
my_skills/
└── portrait-lighting/
    └── SKILL.md
```

**Minimal `SKILL.md`:**

```markdown
---
name: portrait-lighting
description: >-
  Optimise lighting for portrait photography. Use when the user mentions
  "portrait", "face", "skin", "studio lighting", or asks for flattering
  skin tone rendering.
---

1. Append `, dramatic studio lighting, rim light, catchlights` to the positive prompt.
2. Set KSampler CFG to 8.0–9.0 for stronger lighting contrast.
3. Consider read_skill("controlnet-control") and add a normal-map ControlNet
   for additional skin texture depth.
```

Required frontmatter fields: `name` (kebab-case, must match directory name) and
`description`.  Optional: `license`, `compatibility`, `allowed-tools`, `metadata`.

```bash
comfyclaw run --workflow wf.json --prompt "…" --skills-dir ./my_skills/
```

```python
cfg = HarnessConfig(api_key="…", skills_dir="./my_skills/")
```

### Using SkillManager directly

```python
from comfyclaw.skill_manager import SkillManager

sm = SkillManager()                          # built-in skills
sm = SkillManager("./my_skills/")           # custom dir

# Stage-1: XML block for agent system prompt
print(sm.build_available_skills_xml())

# Stage-2: load full instructions on demand
body = sm.get_body("lora-enhancement")

# Lightweight heuristic matching
relevant = sm.detect_relevant_skills("photorealistic portrait")
# → ['photorealistic']

# List of {name, description, location} dicts
manifest = sm.get_manifest()
```

---

## Workflow format

ComfyClaw uses the **API format** (not the UI format with `nodes` array).

```json
{
  "1": {
    "class_type": "CheckpointLoaderSimple",
    "_meta": { "title": "Load Checkpoint" },
    "inputs": { "ckpt_name": "v1-5-pruned.ckpt" }
  },
  "2": {
    "class_type": "CLIPTextEncode",
    "_meta": { "title": "Positive Prompt" },
    "inputs": { "clip": ["1", 1], "text": "a red fox" }
  }
}
```

To get the API format from ComfyUI: **Workflow → Export (API)** in the menu.

`ClawHarness.from_workflow_file()` also handles:
- Prompt-keyed saves: `{ "prompt": { "1": {...}, ... } }`
- UI format: looks for a sibling `*_api.json` first; falls back to a best-effort conversion

---

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full developer guide, including
pre-commit hooks, CI check mapping, and contribution workflow.

### Quick-reference

```bash
# Bootstrap
uv sync --group dev

# Install git hooks (run once per clone)
uv run pre-commit install                       # commit-stage: ruff + file checks
uv run pre-commit install --hook-type pre-push  # push-stage:   pytest + uv build

# Run all hooks manually
uv run pre-commit run --all-files                        # commit stage
uv run pre-commit run --all-files --hook-stage push      # push stage

# Tests (all offline, ~30 s)
uv run pytest -ra -q

# Lint / format
uv run ruff check --fix . && uv run ruff format .

# Build wheel
uv build
```

---

## Project structure

```
comfyclaw/
├── pyproject.toml
├── README.md
├── comfyclaw/
│   ├── __init__.py          custom_node_path(), public re-exports
│   ├── client.py            ComfyClient — HTTP REST + polling
│   ├── workflow.py          WorkflowManager — graph mutations
│   ├── agent.py             ClawAgent — LLM tool-use loop via LiteLLM (14 tools)
│   ├── verifier.py          ClawVerifier — LLM vision QA via LiteLLM
│   ├── memory.py            ClawMemory — per-run attempt history
│   ├── sync_server.py       SyncServer — WebSocket broadcast thread
│   ├── skill_manager.py     SkillManager — Agent Skills spec loader
│   ├── harness.py           ClawHarness + HarnessConfig
│   ├── cli.py               comfyclaw CLI entry point
│   ├── custom_node/         ← bundled ComfyUI plugin
│   │   ├── __init__.py      ComfyUI extension registration
│   │   └── js/
│   │       └── comfy_claw_sync.js   WebSocket client + canvas reload
│   └── skills/              ← built-in skills (Agent Skills format)
│       ├── high-quality/    SKILL.md with YAML frontmatter
│       ├── photorealistic/
│       ├── creative/
│       ├── aesthetic-drawing/
│       ├── creative-drawing/
│       ├── lora-enhancement/
│       ├── controlnet-control/
│       ├── regional-control/
│       ├── hires-fix/
│       ├── spatial/
│       └── text-rendering/
└── tests/
    ├── conftest.py
    ├── test_workflow.py      23 tests
    ├── test_memory.py        12 tests
    ├── test_skill_manager.py 43 tests
    ├── test_verifier.py      16 tests
    ├── test_agent.py         17 tests
    └── test_harness.py       25 tests
```

---

## Known constraints

- **Apple MPS + FP8 models**: `Float8_e4m3fn` is not supported on Apple Silicon
  MPS. Set `weight_dtype: "default"` in your `UNETLoader` node if you see this
  error. The agent is instructed not to change dtype to `fp16` (invalid option).
- **ComfyUI version**: The sync JS extension tries `app.loadApiJson` (≥ 0.2),
  `app.loadGraphData`, and `app.graph.configure` in order. All three are
  supported.
- **Workflow format**: Only API format is sent to ComfyUI's `/prompt` endpoint.
  UI format workflows are converted on load.
