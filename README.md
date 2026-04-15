
<div align="center">


# <img src="assets/logo.png" width="40" style="vertical-align: -25%; margin-right: -5px;"> GEMS: Agent-Native Multimodal Generation with Memory and Skills

<a href="https://arxiv.org/abs/2603.28088"><img src="https://img.shields.io/badge/arXiv-paper-b31b1b?logo=arxiv&logoColor=white" alt="Paper"></a>&nbsp;&nbsp;<a href="https://gems-gen.github.io"><img src="https://img.shields.io/badge/%F0%9F%8C%90%20Project-Page-2563eb" alt="Project Page"></a>&nbsp;&nbsp;
<a href="https://huggingface.co/papers/2603.28088"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-ffc107" alt="Paper"></a>



![Main Image](assets/main.png)

</div>


### Project Overview



```text
GEMS/
├── agent/
│   ├── server/                 # start server
│   │   ├── kimi.sh             # Kimi-K2.5
│   │   ├── qwen_image.py       # Qwen-Image-2512
│   │   └── z_image.py          # Z-Image-Turbo
│   ├── skills/
│   │   ├── aesthetic_drawing
│   │   │   └── SKILL.md
│   │   ├── creative_drawing
│   │   │   └── SKILL.md
│   │   └── ...
│   ├── base_agent.py           # base Interfaces
│   └── GEMS.py                 # core implementation
├── eval/                       # evalation for tasks
│   ├── ArtiMuse/
│   ├── CREA/
│   ├── GenEval2.py
│   └── ...
└── ...
```


### Quick Start

```bash
git clone https://github.com/lcqysl/GEMS.git
cd GEMS
pip install requests litellm torch diffusers transformers fastapi uvicorn accelerate tqdm
```

---

### Setup: MLLM

GEMS uses an MLLM for reasoning, verification, and prompt refinement. Two options:

#### Option A — Cloud API via LiteLLM (recommended)

Supports Claude, GPT-4o, Gemini, and [any model LiteLLM covers](https://docs.litellm.ai/docs/providers). The default config uses **Claude Sonnet 4.6**.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

The model is set in `agent/base_agent.py`:
```python
LITELLM_MODEL = "anthropic/claude-sonnet-4-6"
```
Change this to switch providers (e.g. `"openai/gpt-4o"`, `"gemini/gemini-2.0-flash"`).

#### Option B — Self-hosted via SGLang

To run [Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) locally (requires 8× GPU):

```bash
pip install sglang

MODEL_PATH=/path/to/Kimi-K2.5 bash agent/server/kimi.sh
# Starts on http://localhost:30000
```

Then set `mllm_url` in `infer.py` and switch `base_agent.py` back to using the OpenAI-compatible client.

---

### Setup: Image Generation Server

GEMS supports [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) and [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) as generators.

**Start the Qwen-Image server:**

```bash
MODEL_PATH=/path/to/Qwen-Image-2512 NUM_GPUS=1 python agent/server/qwen_image.py
# Starts on http://localhost:8000
```

- `NUM_GPUS`: number of GPUs to use (default: 1). Each GPU runs an independent worker; requests are load-balanced across them.
- `MODEL_PATH`: local path to the downloaded model weights.

**Start the Z-Image-Turbo server** (faster, 9 steps vs 50):

```bash
MODEL_PATH=/path/to/Z-Image-Turbo NUM_GPUS=1 python agent/server/z_image.py
# Starts on http://localhost:8001
```

**Verify the server is running:**

```bash
curl -X POST "http://localhost:8000/generate?prompt=a+cat+on+a+rooftop" --output test.png
```

---

### Infer

Edit `infer.py` to set your image generation server URL, then run:

```python
# infer.py
gen_url = "http://localhost:8000/generate"   # Qwen-Image (port 8000) or Z-Image-Turbo (port 8001)
max_iterations = 5
```

```bash
python infer.py
```

Output is saved to `infer_results/test_output.png`.

**How it works:** GEMS decomposes the prompt into verification questions, generates an image, checks each requirement, and iteratively refines the prompt based on failures — repeating up to `max_iterations` rounds.

---

### Evaluation

Images are first generated with GEMS, then scored with task-specific methods.

**GenEval2:**

```bash
python eval/GenEval2.py \
    --name my_run \
    --agent gems \
    --max_iterations 5
```

Set `gen_url` and `mllm_url` at the top of `eval/GenEval2.py` before running.

**CREA:**

```bash
python eval/CREA/CREA.py \
    --name my_run \
    --agent gems \
    --max_iterations 5 \
    --n_samples 25
```

**ArtiMuse:**

```bash
python eval/ArtiMuse/gen_artimuse.py \
    --gen_url http://localhost:8000/generate \
    --mllm_url http://localhost:30000/v1 \
    --max_iterations 5
```

**Note:** Occasional server errors (e.g., timeouts) may result in missing outputs for a few tasks. Simply re-run — the scripts automatically skip already-completed items.

We provide full evaluation code for **CREA** and **ArtiMuse**. For other tasks, evaluations follow their official settings.


### Skills
![Skill](assets/skill_demo.png)

Our Skills are summarized from previous works and tested on downstream tasks. You can also add your own by referring to `agent/skills`.

Each skill should be organized as follows:

```text
agent/skills/
└── <skill_id>/             # Unique folder name (used as Skill ID)
    └── SKILL.md            # Skill definition file
```
The `SkillManager` parses `SKILL.md` using regular expressions. To ensure your skill is recognized correctly, please follow this template:

```markdown
# Skill: <Name>

## Description
Provide a concise summary of what this skill does. 

## Instructions
Provide detailed domain-specific guidance, prompts, or constraints here. 
The code will capture all content remaining below this header.
```

### Citation
If you find our work useful, please consider citing:
```code
@article{he2026gems,
  title={GEMS: Agent-Native Multimodal Generation with Memory and Skills},
  author={He, Zefeng and Huang, Siyuan and Qu, Xiaoye and Li, Yafu and Zhu, Tong and Cheng, Yu and Yang, Yang},
  journal={arXiv preprint arXiv:2603.28088},
  year={2026}
}
```