# ComfyClaw

[![CI](https://github.com/davidliuk/comfyclaw/actions/workflows/ci.yml/badge.svg)](https://github.com/davidliuk/comfyclaw/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Agentic harness for self-evolving ComfyUI image-generation workflows.**

ComfyClaw wraps ComfyUI in a Claude Sonnet agent loop that _grows_ the
workflow topology in response to image quality feedback — adding LoRA loaders,
ControlNet branches, regional conditioning, and hires-fix passes until a
configurable quality threshold is reached.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ClawHarness loop                            │
│                                                                     │
│  ┌──────────┐  evolve  ┌──────────────┐  submit  ┌──────────────┐  │
│  │  Agent   │ ───────► │  ComfyUI API │ ───────► │  Image out   │  │
│  │ (Claude) │          │  (HTTP/WS)   │          └──────┬───────┘  │
│  └────┬─────┘          └──────────────┘                 │          │
│       │  feedback                                        │ verify   │
│       │ ◄────────────────────────────────────────────── ▼          │
│       │                                          ┌──────────────┐  │
│       │                                          │  Verifier    │  │
│  ┌────┴─────┐  broadcast  ┌──────────────────┐  │ (Claude vis) │  │
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
- An [Anthropic API key](https://console.anthropic.com/)

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
| *(none)* | `anthropic`, `python-dotenv` | Always |
| `sync` | `websockets>=12` | Live graph updates in ComfyUI canvas |
| `dev` (group) | `pytest`, `ruff`, `mypy`, … | Development & CI |

### 3. Configure environment

```bash
cp .env.example .env
# Open .env and set at minimum:
#   ANTHROPIC_API_KEY=sk-ant-...
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
comfyclaw install-node --comfyui-dir ~/Documents/ComfyUI

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
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key for Claude |
| `COMFYUI_DIR` | `~/Documents/ComfyUI` | Path to your ComfyUI installation |
| `COMFYUI_ADDR` | `127.0.0.1:8188` | `host:port` of a running ComfyUI server |

---

## Quick start

```bash
# 1. Set your API key (or put it in .env)
export ANTHROPIC_API_KEY=sk-ant-...

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
| `--model NAME` | `claude-sonnet-4-5` | Claude model for agent and verifier |
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
    api_key="sk-ant-...",       # or use ANTHROPIC_API_KEY env var
    server_address="127.0.0.1:8188",
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
    api_key: str                       # Anthropic API key
    server_address: str = "127.0.0.1:8188"
    model: str = "claude-sonnet-4-5"
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

The Claude agent has 13 tools grouped by category:

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
| Control | `report_evolution_strategy` | Declare plan before making changes |
| Control | `finalize_workflow` | Signal completion and return rationale |

### Agent decision heuristics

The agent is instructed to choose tools based on verifier feedback:

| Verifier finding | Agent action |
|---|---|
| Flat / low-depth background | `add_controlnet` (depth model) |
| Blurry edges / lost structure | `add_controlnet` (canny model) |
| Wrong human pose / body | `add_controlnet` (pose model) |
| Plasticky skin / poor texture | `add_lora_loader` (detail/texture LoRA) |
| Wrong anatomy (hands, fingers) | `add_lora_loader` (anatomy LoRA) or `add_inpaint_pass` |
| Style inconsistency | `add_lora_loader` (style LoRA) |
| Subject and background bleed | `add_regional_attention` |
| Low resolution / soft fine detail | `add_hires_fix` |
| Localised artifact in one area | `add_inpaint_pass` |

---

## Skills

Skills are SKILL.md files that provide the agent with domain-specific guidance.
The agent auto-detects which skills are relevant to the prompt via keyword
matching and injects their full `## Instructions` section into the user message.

Built-in skills (in `comfyclaw/skills/`):

| Skill | Trigger keywords |
|---|---|
| `high_quality` | quality, sharp, detailed, crisp |
| `photorealistic` | photorealistic, photo, DSLR, realistic |
| `creative` | creative, artistic, stylized |
| `aesthetic_drawing` | drawing, illustration, anime, sketch |
| `lora_enhancement` | detail, texture, anatomy, style, enhance |
| `controlnet_control` | structure, depth, pose, canny, edges |
| `regional_control` | region, foreground, background, separate |
| `hires_fix` | high-res, hires, upscale, detail |
| `spatial` | spatial, composition, layout |
| `text_rendering` | text, typography, lettering |
| `creative_drawing` | creative drawing, concept art |

### Adding custom skills

Create a directory with a `SKILL.md` and point the harness to it:

```
my_skills/
└── portrait_lighting/
    └── SKILL.md
```

`SKILL.md` format:

```markdown
# Skill: Portrait Lighting

## Description
Optimize lighting for portrait photography. Trigger on: portrait, face, skin, lighting.

## Instructions

### Steps
1. Add "dramatic studio lighting, rim light" to the positive prompt.
2. Increase CFG to 8–9 for stronger lighting contrast.
3. Consider add_controlnet with a normal map for skin texture.
```

```bash
comfyclaw run --workflow wf.json --prompt "…" --skills-dir ./my_skills/
```

```python
cfg = HarnessConfig(api_key="…", skills_dir="./my_skills/")
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

### Running the tests

```bash
# All 103 tests (fully offline — Anthropic API mocked)
uv run pytest

# Verbose, specific module
uv run pytest tests/test_workflow.py -v
uv run pytest -k "topology_accumulation" -v

# Stop on first failure
uv run pytest -x
```

### Linting and formatting

```bash
uv run ruff check .           # lint
uv run ruff check --fix .     # auto-fix
uv run ruff format .          # format
uv run mypy comfyclaw/        # type-check (optional)
```

### Building a wheel

```bash
uv build                      # produces dist/comfyclaw-*.whl
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
│   ├── agent.py             ClawAgent — Claude tool-use loop
│   ├── verifier.py          ClawVerifier — Claude vision QA
│   ├── memory.py            ClawMemory — per-run attempt history
│   ├── sync_server.py       SyncServer — WebSocket broadcast thread
│   ├── skill_manager.py     SkillManager — SKILL.md loader & matcher
│   ├── harness.py           ClawHarness + HarnessConfig
│   ├── cli.py               comfyclaw CLI entry point
│   ├── custom_node/         ← bundled ComfyUI plugin
│   │   ├── __init__.py      ComfyUI extension registration
│   │   └── js/
│   │       └── comfy_claw_sync.js   WebSocket client + canvas reload
│   └── skills/              ← built-in skill SKILL.md files
│       ├── high_quality/
│       ├── photorealistic/
│       ├── lora_enhancement/
│       ├── controlnet_control/
│       ├── regional_control/
│       ├── hires_fix/
│       └── …
└── tests/
    ├── conftest.py
    ├── test_workflow.py      23 tests
    ├── test_memory.py        12 tests
    ├── test_skill_manager.py 19 tests
    ├── test_verifier.py      16 tests
    ├── test_agent.py         17 tests
    └── test_harness.py       16 tests
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
