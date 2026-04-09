# Contributing to ComfyClaw

Thank you for your interest in contributing!  This document covers everything
you need to set up a development environment, run tests, and open a pull
request.

---

## Table of contents

1. [Development setup](#development-setup)
2. [Running tests](#running-tests)
3. [Code style](#code-style)
4. [Adding / editing skills](#adding--editing-skills)
5. [Adding agent tools](#adding-agent-tools)
6. [Pull request workflow](#pull-request-workflow)
7. [Reporting bugs](#reporting-bugs)

---

## Development setup

ComfyClaw uses [**uv**](https://docs.astral.sh/uv/) for dependency management.

### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or via Homebrew
brew install uv
```

### 2. Clone and bootstrap

```bash
git clone https://github.com/davidliuk/comfyclaw.git
cd comfyclaw

# Install the package + all dev dependencies into an isolated .venv
uv sync --group dev

# Activate (optional — uv run works without activating)
source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
```

### 3. Configure your environment

```bash
cp .env.example .env
# Edit .env — at minimum set ANTHROPIC_API_KEY
```

### 4. Install the ComfyUI plugin (optional, for live-sync testing)

```bash
uv run comfyclaw install-node
# Then restart ComfyUI
```

---

## Running tests

All tests are fully offline — Anthropic API calls are mocked with `pytest-mock`.

```bash
# Run all tests
uv run pytest

# Verbose output
uv run pytest -v

# Run a specific test file
uv run pytest tests/test_workflow.py -v

# Run a specific test class or function
uv run pytest -k "TestClone"
uv run pytest -k "test_topology_accumulation"

# Stop on first failure
uv run pytest -x
```

### Test structure

| File | Coverage |
|---|---|
| `tests/test_workflow.py` | `WorkflowManager` — add/connect/delete/validate/clone |
| `tests/test_memory.py` | `ClawMemory` — record, best, image cap |
| `tests/test_skill_manager.py` | `SkillManager` — load, instructions, detect |
| `tests/test_verifier.py` | `ClawVerifier` — encode once, JPEG/PNG, region issues |
| `tests/test_agent.py` | `ClawAgent` — tool dispatch, LoRA rewire, regional |
| `tests/test_harness.py` | `ClawHarness` — dry-run, early stop, topology accum. |

---

## Code style

```bash
# Lint (check only)
uv run ruff check .

# Auto-fix lint issues
uv run ruff check --fix .

# Format
uv run ruff format .

# Type-check (optional, non-blocking in CI)
uv run mypy comfyclaw/
```

Style rules are in `pyproject.toml` under `[tool.ruff]`.  The CI linting step
will fail PRs with unfixed ruff errors.  Formatting is checked but auto-applied
locally.

---

## Adding / editing skills

Skills are Markdown files in `comfyclaw/skills/<skill_id>/SKILL.md`.

**Required sections:**

```markdown
# Skill: <Human-Readable Name>

## Description
<1–3 sentences describing what this skill does and when it applies.>
Trigger on: <comma-separated trigger keywords>.

## Instructions

### Steps
1. …
2. …
```

Rules:
- `## Description` is shown to the agent as a one-liner in the manifest.
- `## Instructions` is injected in full when the skill matches the prompt.
- Keywords in the Description line ("Trigger on: …") are used by
  `SkillManager.detect_relevant_skills()` for keyword matching.
- Add a test for your skill's keywords in `tests/test_skill_manager.py`.

---

## Adding agent tools

Agent tools are defined in `comfyclaw/agent.py`:

1. Add an entry to the `_TOOLS` list (follows the Anthropic `tool` schema).
2. Add a `case "your_tool_name":` branch in `ClawAgent._dispatch()`.
3. Implement the logic as a `_your_tool` private method.
4. Add decision guidance to `_SYSTEM_PROMPT` (when to pick this tool).
5. Add a test in `tests/test_agent.py` that mocks Anthropic and verifies the
   workflow mutation.

---

## Pull request workflow

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feat/my-feature
   ```
2. Make your changes, add tests, update `CHANGELOG.md` under `[Unreleased]`.
3. Run the full test + lint suite locally:
   ```bash
   uv run pytest && uv run ruff check . && uv run ruff format --check .
   ```
4. Push and open a PR against `main`.  The CI will run automatically.
5. Address reviewer feedback.  Squash merge is preferred.

### Commit style

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add add_ip_adapter agent tool
fix: correct msg.workflow key in JS sync extension
docs: expand skill authoring guide
test: add harness evolution log tests
refactor: extract _load_dotenv into config module
```

---

## Reporting bugs

Please open a [GitHub Issue](https://github.com/davidliuk/comfyclaw/issues) with:

- ComfyClaw version (`comfyclaw --version` once implemented, or git commit)
- Python version and OS
- ComfyUI version
- Minimal reproduction: workflow JSON, prompt, and error message / traceback
