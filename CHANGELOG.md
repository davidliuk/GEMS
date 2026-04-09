# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.1.0] — 2025-04-08

Initial public release extracted from the `vision-harness` research monorepo.

### Added

**Package structure**
- Standalone `comfyclaw/` Python package, installable via `pip` or `uv`.
- `pyproject.toml` with hatchling build backend, uv dependency groups, ruff/mypy config.
- `uv.lock` + `.python-version` (3.13) for reproducible environments.
- `.env.example` template covering all configuration variables.
- MIT `LICENSE`.

**Core modules**
- `client.py` — `ComfyClient`: HTTP REST + polling against the ComfyUI API.
- `workflow.py` — `WorkflowManager`: add / connect / delete / validate / clone nodes.
- `agent.py` — `ClawAgent`: Claude Sonnet tool-use loop with 13 tools for workflow evolution.
- `verifier.py` — `ClawVerifier`: Claude vision verifier with region-level analysis and configurable score weights.
- `memory.py` — `ClawMemory`: per-run attempt history with configurable image-bytes cap.
- `sync_server.py` — `SyncServer`: thread-safe WebSocket broadcast server.
- `skill_manager.py` — `SkillManager`: SKILL.md loader with description/instructions parsing and keyword-based skill detection.
- `harness.py` — `ClawHarness` + `HarnessConfig`: orchestrator with topology accumulation, early stopping, and context manager support.
- `cli.py` — `comfyclaw` CLI with `run`, `dry-run`, `install-node`, `node-path` sub-commands; reads all config from env vars / `.env`.

**Agent tools (13)**
`inspect_workflow`, `query_available_models`, `set_param`, `add_node`,
`connect_nodes`, `delete_node`, `add_lora_loader`, `add_controlnet`,
`add_regional_attention`, `add_hires_fix`, `add_inpaint_pass`,
`report_evolution_strategy`, `finalize_workflow`.

**Built-in skills (11)**
`high_quality`, `photorealistic`, `creative`, `aesthetic_drawing`,
`lora_enhancement`, `controlnet_control`, `regional_control`, `hires_fix`,
`spatial`, `text_rendering`, `creative_drawing`.

**ComfyUI live-sync plugin**
- `comfyclaw/custom_node/` bundled inside the Python package.
- `comfy_claw_sync.js` v1.1: WebSocket client with auto-reconnect, three-method canvas reload (loadApiJson / loadGraphData / configure), status badge.
- Fixed: JS was reading `msg.data` but Python sends `msg.workflow`.
- `comfyclaw install-node` symlinks the bundled plugin; `comfyclaw node-path` prints its location.

**Tests** — 103 tests, fully offline (Anthropic mocked)
- `test_workflow.py` (23), `test_memory.py` (12), `test_skill_manager.py` (19),
  `test_verifier.py` (16), `test_agent.py` (17), `test_harness.py` (16).

### Fixed
- `WorkflowManager.delete_node`: added `str()` coerce for node-ID comparison.
- `ClawVerifier`: base64-encodes image bytes **once** before parallel threads (was re-encoding per question).
- `ClawVerifier`: detects JPEG vs PNG from magic bytes (was hardcoded `image/png`).
- `SkillManager.get_instructions()`: now returns full `## Instructions` body (was dead method).
- `ClawAgent._add_regional_attention`: guards `node.get("_meta") or {}` to prevent `KeyError`.
- `SyncServer._clients`: protected with `threading.Lock` (was accessed unsafely across threads).
- `ClawHarness`: removed `sys.path.insert` hack; all imports within the package.

[Unreleased]: https://github.com/davidliuk/comfyclaw/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/davidliuk/comfyclaw/releases/tag/v0.1.0
