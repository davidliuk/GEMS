"""
ClawHarness — top-level orchestrator for the self-evolving generate–verify loop.

Usage (context-manager)::

    cfg = HarnessConfig(api_key="sk-ant-...", max_iterations=4)
    with ClawHarness.from_workflow_file("workflow_api.json", cfg) as h:
        image_bytes = h.run("a red fox at dawn, photorealistic")

Topology accumulation
---------------------
When ``evolve_from_best=True`` (the default) each iteration starts from the
**best workflow snapshot** produced so far rather than resetting to the
original base workflow.  This means LoRA / ControlNet nodes added in round 1
persist into round 2, and the agent only needs to add *incremental* upgrades.
Set ``evolve_from_best=False`` to revert to the old reset-each-iteration
behaviour.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .agent import ClawAgent
from .client import ComfyClient
from .memory import ClawMemory
from .sync_server import SyncServer
from .verifier import ClawVerifier, VerifierResult
from .workflow import WorkflowManager

# Error messages that indicate a transient infrastructure fault in ComfyUI
# (broken pipe from tqdm/progress-bar writing to a closed stderr, etc.).
# These are NOT workflow logic errors — the agent should not attempt a repair;
# instead, harness should retry the same workflow after a short pause.
_INFRA_ERROR_SIGNALS = (
    "[Errno 32] Broken pipe",
    "BrokenPipeError",
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class HarnessConfig:
    """
    All tuning knobs for a ``ClawHarness`` run in one place.

    Parameters
    ----------
    api_key               : Anthropic API key (required).
    server_address        : ComfyUI HTTP address, e.g. ``"127.0.0.1:8188"``.
    model                 : Claude model for both agent and verifier.
    max_iterations        : Maximum agent–generate–verify cycles.
    success_threshold     : Stop early when verifier score reaches this value.
    sync_port             : WebSocket port for live UI sync; 0 to disable.
    skills_dir            : Path to SKILL.md directory; ``None`` uses built-in skills.
    evolve_from_best      : Start each iteration from the best previous workflow.
    max_images            : Max images kept in RAM across attempts (see ClawMemory).
    score_weights         : ``(req_weight, detail_weight)`` for verifier score blend.
    image_model           : Pin the ComfyUI checkpoint / UNET to this name.
                            Must be the **exact filename** as reported by ComfyUI
                            (e.g. ``"qwen_image_2512_fp8_e4m3fn.safetensors"``),
                            not a HuggingFace-style path.  ``None`` leaves the
                            workflow's existing model untouched.
    max_repair_attempts   : When ComfyUI rejects a workflow (HTTP 4xx / execution
                            error), the agent gets up to this many chances to
                            inspect the error and fix the topology before the
                            iteration is abandoned.  Set to 0 to disable repairs.
    """

    api_key: str = ""
    server_address: str = "127.0.0.1:8188"
    model: str = "anthropic/claude-sonnet-4-5"
    verifier_model: str | None = None
    max_iterations: int = 3
    success_threshold: float = 0.85
    sync_port: int = 8765
    skills_dir: str | None = None
    evolve_from_best: bool = True
    max_images: int = 5
    score_weights: tuple[float, float] = field(default_factory=lambda: (0.6, 0.4))
    image_model: str | None = None
    max_repair_attempts: int = 2
    verifier_mode: str = "vlm"  # "vlm", "human", or "hybrid"
    """
    Pin the image-generation model (checkpoint / UNET) used by ComfyUI.

    When set, this model name is written into every loader node
    (``CheckpointLoaderSimple``, ``UNETLoader``, etc.) in the workflow
    at startup and after each topology evolution, so the agent cannot
    accidentally swap it out.

    Examples::

        image_model = "qwen_image_2512_fp8_e4m3fn.safetensors"  # exact ComfyUI filename
        image_model = "realisticVisionV51.safetensors"           # local checkpoint
        image_model = None   # do not override — use whatever the workflow has
    """


# ---------------------------------------------------------------------------
# Evolution log
# ---------------------------------------------------------------------------


@dataclass
class EvolutionEntry:
    iteration: int
    node_count_before: int
    node_count_after: int
    node_ids_added: list[str]
    rationale: str
    verifier_score: float | None = None

    def summary(self) -> str:
        diff = self.node_count_after - self.node_count_before
        sign = "+" if diff >= 0 else ""
        added = ", ".join(self.node_ids_added) or "none"
        return (
            f"  Iter {self.iteration}: nodes {self.node_count_before}→{self.node_count_after} "
            f"({sign}{diff}), added=[{added}], score={self.verifier_score}"
        )


class EvolutionLog:
    def __init__(self) -> None:
        self.entries: list[EvolutionEntry] = []

    def record(self, entry: EvolutionEntry) -> None:
        self.entries.append(entry)

    def format(self) -> str:
        if not self.entries:
            return "  (no entries yet)"
        return "\n".join(e.summary() for e in self.entries)

    def __len__(self) -> int:
        return len(self.entries)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class ClawHarness:
    """
    Orchestrates the ClawAgent → ComfyUI → ClawVerifier self-evolving loop.

    Prefer constructing via ``from_workflow_file`` or ``from_workflow_dict``.
    Use as a context manager to ensure the sync server is stopped cleanly.
    """

    def __init__(self, base_workflow: dict, config: HarnessConfig) -> None:
        self.base_workflow = copy.deepcopy(base_workflow)
        self.config = config

        # Apply any pinned image model to the base workflow immediately,
        # so it is the starting point for every iteration.
        if config.image_model:
            # Warn early if the value looks like a HuggingFace path rather than
            # a ComfyUI filename — ComfyUI only accepts exact filenames and will
            # return HTTP 400 otherwise.
            im = config.image_model
            if "/" in im and not any(
                im.endswith(ext) for ext in (".safetensors", ".ckpt", ".pt", ".gguf", ".bin")
            ):
                log.warning(
                    "image_model=%r looks like a HuggingFace path, not a ComfyUI filename. "
                    "ComfyUI requires the exact local filename (e.g. 'qwen_image_2512_fp8_e4m3fn.safetensors'). "
                    "The workflow submission will likely fail with HTTP 400.",
                    im,
                )
            wm = WorkflowManager(self.base_workflow)
            updated = wm.apply_image_model(config.image_model)
            self.base_workflow = wm.workflow
            if updated:
                log.info(
                    "Pinned image model %r on %d loader node(s): %s",
                    config.image_model,
                    len(updated),
                    ", ".join(f"[{nid}].{p}" for nid, p in updated),
                )
            else:
                log.warning(
                    "image_model=%r set but no loader nodes found in workflow; "
                    "the model pin will have no effect.",
                    config.image_model,
                )

        self._client = ComfyClient(config.server_address)
        self._sync = SyncServer(port=config.sync_port) if config.sync_port else None
        self._memory = ClawMemory(max_images=config.max_images)
        self._evolution_log = EvolutionLog()

        self._agent = ClawAgent(
            api_key=config.api_key,
            model=config.model,
            server_address=config.server_address,
            skills_dir=config.skills_dir,
            on_change=self._on_workflow_change,
            pinned_image_model=config.image_model,
        )

        vlm_verifier = ClawVerifier(
            api_key=config.api_key,
            model=config.verifier_model or config.model,
            score_weights=config.score_weights,
        )

        mode = config.verifier_mode
        if mode == "human":
            from .human_verifier import HumanVerifier

            self._verifier = HumanVerifier(
                sync_server=self._sync,
                timeout=600.0,
            )
            log.info("Verifier mode: human (feedback via ComfyUI panel or terminal)")
        elif mode == "hybrid":
            from .human_verifier import HybridVerifier

            self._verifier = HybridVerifier(
                vlm_verifier=vlm_verifier,
                sync_server=self._sync,
                timeout=600.0,
            )
            log.info("Verifier mode: hybrid (VLM + human override)")
        else:
            self._verifier = vlm_verifier
            log.info("Verifier mode: vlm")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> ClawHarness:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._sync:
            self._sync.start()

    def stop(self) -> None:
        if self._sync:
            self._sync.stop()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, prompt: str, dry_run: bool = False) -> bytes | None:
        """
        Run the self-evolving generate–verify loop.

        Parameters
        ----------
        prompt  : Image generation prompt.
        dry_run : If ``True``, skip actual ComfyUI execution.

        Returns
        -------
        Raw image bytes of the best result, or ``None`` on failure / dry-run.
        """
        cfg = self.config
        log.info("Starting run: %r", prompt)
        print(f"\n{'=' * 60}")
        print(f"[ClawHarness] Run: {prompt!r}")
        print(f"{'=' * 60}")

        self._memory.clear()
        self._evolution_log = EvolutionLog()
        best_image: bytes | None = None
        best_score = -1.0
        best_workflow_snapshot: dict | None = None
        last_result: VerifierResult | None = None

        for iteration in range(1, cfg.max_iterations + 1):
            print(f"\n--- Iteration {iteration}/{cfg.max_iterations} ---")

            # ── Choose starting workflow ───────────────────────────────────
            if cfg.evolve_from_best and best_workflow_snapshot is not None:
                start_wf = copy.deepcopy(best_workflow_snapshot)
            else:
                start_wf = copy.deepcopy(self.base_workflow)

            wm = WorkflowManager(start_wf)

            # Prepare sync: reset to empty so every subsequent broadcast
            # produces add_node diffs (not a single full snapshot).
            if self._sync:
                self._sync.reset(empty=True)

            # Broadcast base workflow nodes one-by-one so the ComfyUI
            # canvas shows them appearing incrementally.
            if self._sync and start_wf:
                partial: dict = {}
                for nid in sorted(start_wf.keys(), key=lambda k: int(k)):
                    partial[nid] = copy.deepcopy(start_wf[nid])
                    self._sync.broadcast(copy.deepcopy(partial))

            # Seed the user's prompt into every CLIPTextEncode-family node
            # connected to a sampler's positive input.
            pos_injected, _ = wm.inject_prompt(positive=prompt)
            if pos_injected:
                print(f"[ClawHarness] 📝 Seeded user prompt into encoder node(s) {pos_injected}")
                if self._sync:
                    self._sync.broadcast(wm.to_dict())

            node_ids_before = set(wm.workflow.keys())

            # ── Agent evolves the workflow ─────────────────────────────────
            verifier_feedback = self._build_feedback(last_result)
            memory_summary = (
                self._memory.format_history_for_agent() if self._memory.attempts else None
            )

            print("[ClawHarness] 🤖 Agent is evolving the workflow…")
            rationale = self._agent.plan_and_patch(
                workflow_manager=wm,
                original_prompt=prompt,
                verifier_feedback=verifier_feedback,
                memory_summary=memory_summary,
                iteration=iteration,
            )

            node_ids_after = set(wm.workflow.keys())
            added_ids = sorted(node_ids_after - node_ids_before)
            evo = EvolutionEntry(
                iteration=iteration,
                node_count_before=len(node_ids_before),
                node_count_after=len(node_ids_after),
                node_ids_added=added_ids,
                rationale=rationale,
            )
            if added_ids:
                new_classes = [wm.workflow[nid].get("class_type", "?") for nid in added_ids]
                print(f"[ClawHarness] 🔧 Added nodes {added_ids} → {new_classes}")

            # Re-apply pinned model after agent evolution — the agent may have
            # added new loader nodes (e.g. a LoRA) and we must ensure the
            # primary checkpoint / UNET still points at the configured model.
            if cfg.image_model:
                wm.apply_image_model(cfg.image_model)

            self._on_workflow_change(wm.workflow)

            # ── Dry-run mode ───────────────────────────────────────────────
            if dry_run:
                print("[ClawHarness] ⏭  dry_run=True — skipping ComfyUI execution.")
                print(f"[ClawHarness] Final workflow ({len(wm)} nodes):")
                print(json.dumps(wm.workflow, indent=2)[:3000])
                self._evolution_log.record(evo)
                return None

            # ── Submit with repair loop ────────────────────────────────────
            # When ComfyUI rejects a workflow (HTTP 4xx / execution error),
            # the agent gets up to cfg.max_repair_attempts chances to inspect
            # the error message and fix the topology before this iteration is
            # abandoned.
            prompt_id: str | None = None
            submission_error: str | None = None

            for repair_round in range(cfg.max_repair_attempts + 1):
                label = (
                    "Submitting"
                    if repair_round == 0
                    else f"Repair {repair_round}/{cfg.max_repair_attempts}"
                )
                print(f"[ClawHarness] 🚀 {label} to ComfyUI…")

                # On repair rounds let the agent fix the workflow in-place.
                if repair_round > 0:
                    repair_feedback = self._build_repair_feedback(submission_error, last_result)
                    self._agent.plan_and_patch(
                        workflow_manager=wm,
                        original_prompt=prompt,
                        verifier_feedback=repair_feedback,
                        iteration=iteration,
                    )
                    if cfg.image_model:
                        wm.apply_image_model(cfg.image_model)
                    self._on_workflow_change(wm.workflow)

                try:
                    queue_resp = self._client.queue_prompt(wm.workflow)
                    prompt_id = queue_resp["prompt_id"]
                    submission_error = None
                    if repair_round > 0:
                        print(f"[ClawHarness] ✅ Repair {repair_round} accepted by ComfyUI.")
                    break
                except Exception as exc:
                    submission_error = str(exc)
                    print(
                        f"[ClawHarness] ❌ {'Repair' if repair_round > 0 else 'Queue'} error: {exc}"
                    )

            if prompt_id is None:
                self._record_error(
                    iteration, wm.workflow, submission_error or "unknown queue error"
                )
                self._evolution_log.record(evo)
                continue

            # ── Wait for completion ────────────────────────────────────────
            try:
                history = self._client.wait_for_completion(prompt_id, timeout=600)
            except TimeoutError as exc:
                print(f"[ClawHarness] ❌ Timeout: {exc}")
                self._record_error(iteration, wm.workflow, str(exc))
                self._evolution_log.record(evo)
                continue

            # ── Handle ComfyUI execution-time error ────────────────────────
            if "error" in history:
                exec_error = history["error"]
                print(f"[ClawHarness] ❌ ComfyUI execution error: {exec_error}")

                # ── Infra fault (BrokenPipe from tqdm stderr) — not a workflow bug
                # Retry the SAME workflow once after a short pause; do NOT ask the
                # agent to repair anything.
                if any(sig in exec_error for sig in _INFRA_ERROR_SIGNALS):
                    print(
                        "[ClawHarness] ⚠  Transient infrastructure error detected "
                        "(BrokenPipe / progress-bar stderr flush). Waiting 5 s then "
                        "retrying the same workflow once."
                    )
                    time.sleep(5)
                    try:
                        rq_retry = self._client.queue_prompt(wm.workflow)
                        retry_pid = rq_retry["prompt_id"]
                        print(f"[ClawHarness] 🔄 Infra-retry submitted ({retry_pid}).")
                        history = self._client.wait_for_completion(retry_pid, timeout=600)
                    except Exception as infra_exc:
                        print(f"[ClawHarness] ❌ Infra-retry exception: {infra_exc}")
                        self._record_error(iteration, wm.workflow, str(infra_exc))
                        self._evolution_log.record(evo)
                        last_result = None
                        continue

                    if "error" in history:
                        infra_msg = history["error"]
                        print(f"[ClawHarness] ❌ Infra-retry also failed: {infra_msg}")
                        self._record_error(iteration, wm.workflow, infra_msg)
                        self._evolution_log.record(evo)
                        last_result = None
                        continue
                    # Retry succeeded — fall through to image collection below.
                    print("[ClawHarness] ✅ Infra-retry succeeded.")

                else:
                    # ── Workflow-logic error — let the agent repair the topology
                    repaired_prompt_id: str | None = None
                    exec_submission_error: str | None = exec_error

                    for repair_round in range(1, cfg.max_repair_attempts + 1):
                        print(
                            f"[ClawHarness] 🔧 Execution repair {repair_round}/{cfg.max_repair_attempts}…"
                        )
                        repair_feedback = self._build_repair_feedback(
                            exec_submission_error, last_result
                        )
                        self._agent.plan_and_patch(
                            workflow_manager=wm,
                            original_prompt=prompt,
                            verifier_feedback=repair_feedback,
                            iteration=iteration,
                        )
                        if cfg.image_model:
                            wm.apply_image_model(cfg.image_model)
                        self._on_workflow_change(wm.workflow)

                        try:
                            rq = self._client.queue_prompt(wm.workflow)
                            repaired_prompt_id = rq["prompt_id"]
                            exec_submission_error = None
                            print(f"[ClawHarness] ✅ Execution repair {repair_round} accepted.")
                            break
                        except Exception as exc2:
                            exec_submission_error = str(exc2)
                            print(
                                f"[ClawHarness] ❌ Execution repair {repair_round} failed: {exc2}"
                            )

                    if repaired_prompt_id is None:
                        self._record_error(
                            iteration, wm.workflow, exec_submission_error or exec_error
                        )
                        self._evolution_log.record(evo)
                        last_result = None
                        continue

                    # Wait for the repaired workflow to finish.
                    try:
                        history = self._client.wait_for_completion(repaired_prompt_id, timeout=600)
                    except TimeoutError as exc:
                        print(f"[ClawHarness] ❌ Timeout after repair: {exc}")
                        self._record_error(iteration, wm.workflow, str(exc))
                        self._evolution_log.record(evo)
                        continue

                    if "error" in history:
                        msg = history["error"]
                        print(f"[ClawHarness] ❌ ComfyUI error after repair: {msg}")
                        self._record_error(iteration, wm.workflow, msg)
                        self._evolution_log.record(evo)
                        last_result = None
                        continue

            images = self._client.collect_images(history)
            if not images:
                print("[ClawHarness] ⚠  No images in output — check workflow.")
                self._evolution_log.record(evo)
                continue

            image_bytes = images[0]
            print(f"[ClawHarness] 🖼  Got image ({len(image_bytes):,} bytes)")

            # ── Verify ────────────────────────────────────────────────────
            print("[ClawHarness] 🔍 Verifying image…")
            result = self._verifier.verify(image_bytes, prompt, iteration=iteration)
            last_result = result
            print(f"[ClawHarness] Score: {result.score:.2f}")
            print(result.format_feedback())

            if result.score > best_score:
                best_score = result.score
                best_image = image_bytes
                best_workflow_snapshot = wm.to_dict()

            evo.verifier_score = result.score
            self._evolution_log.record(evo)

            # ── Record in memory ──────────────────────────────────────────
            experience = self._summarize_experience(prompt, result.passed, result.failed, rationale)
            self._memory.record(
                iteration=iteration,
                workflow_snapshot=wm.to_dict(),
                verifier_score=result.score,
                passed=result.passed,
                failed=result.failed,
                experience=experience,
                image_bytes=image_bytes,
            )

            # ── Early stop ────────────────────────────────────────────────
            if result.score >= cfg.success_threshold:
                print(
                    f"[ClawHarness] ✅ Score {result.score:.2f} ≥ threshold "
                    f"{cfg.success_threshold} — stopping early."
                )
                break

        self._print_summary(best_score)
        return best_image

    # ------------------------------------------------------------------
    # Callbacks & helpers
    # ------------------------------------------------------------------

    def _on_workflow_change(self, workflow: dict) -> None:
        if self._sync:
            self._sync.broadcast(workflow)

    def _build_repair_feedback(
        self, error_msg: str | None, last_result: VerifierResult | None
    ) -> str:
        """
        Feedback passed to the agent when ComfyUI rejected the workflow.

        Puts the raw error front-and-centre so the agent can fix the exact
        broken connection or invalid parameter before the next submission.
        """
        lines = [
            "## ⚠️ ComfyUI Rejected the Workflow — Repair Required",
            "",
            "Your last workflow submission was rejected with the following error:",
            f"```\n{error_msg or '(no error details)'}\n```",
            "",
            "**Repair protocol (follow in order):**",
            "1. Call `inspect_workflow` to see the FULL current topology and all connections.",
            "2. Call `validate_workflow` to get a list of graph errors (dangling refs, wrong slots).",
            "3. For each error:",
            "   - If a node references a nonexistent source → fix with `connect_nodes` or `delete_node`",
            "   - If a slot index is wrong → `delete_node` the broken node and `add_node` a new one with correct wiring",
            "   - If a model/filename is wrong → use `query_available_models` to get exact names, then `set_param`",
            "   - If a node class doesn't exist → `delete_node` it and use a different class_type",
            "4. Call `validate_workflow` again to confirm all issues are resolved.",
            "5. Call `finalize_workflow` (it will auto-validate and block if still broken).",
            "",
            "**IMPORTANT:** Do NOT just add new nodes on top of broken ones — `delete_node` the",
            "broken node first, then `add_node` a replacement with correct connections.",
            "",
            "**Output slot reference:**",
            "  CheckpointLoaderSimple → slot 0: MODEL, slot 1: CLIP, slot 2: VAE",
            "  UNETLoader / CLIPLoader / VAELoader → slot 0 only",
            "  KSampler → slot 0: LATENT",
            "  VAEDecode → slot 0: IMAGE",
            "  CLIPTextEncode → slot 0: CONDITIONING",
        ]
        if last_result:
            lines += [
                "",
                "── Previous Verifier Feedback (for context) ──",
                last_result.format_feedback(),
            ]
        return "\n".join(lines)

    def _build_feedback(self, result: VerifierResult | None) -> str | None:
        if result is None:
            return None
        lines = [result.format_feedback()]
        if self._evolution_log.entries:
            lines.append("\n── Evolution History ──")
            lines.append(self._evolution_log.format())
        lines.append(
            "\nChoose the single highest-impact structural upgrade from the "
            "evolution_suggestions above. Declare it with report_evolution_strategy first."
        )
        return "\n".join(lines)

    def _record_error(self, iteration: int, workflow: dict, msg: str) -> None:
        self._memory.record(
            iteration=iteration,
            workflow_snapshot=workflow,
            verifier_score=0.0,
            passed=[],
            failed=[f"Execution error: {msg}"],
            experience=f"Workflow failed: {msg}. Inspect and fix before next attempt.",
        )

    def _summarize_experience(
        self, prompt: str, passed: list[str], failed: list[str], rationale: str
    ) -> str:
        try:
            msg = (
                f"Summarize in ≤80 words. Focus on what worked, failed, and the key lesson.\n\n"
                f"Prompt: {prompt}\nPassed: {', '.join(passed) or 'none'}\n"
                f"Failed: {', '.join(failed) or 'none'}\nAgent rationale: {rationale}"
            )
            return self._verifier.complete(msg, max_tokens=200)
        except Exception as exc:
            return f"Summary unavailable: {exc}"

    def _print_summary(self, best_score: float) -> None:
        print("\n[ClawHarness] ── Evolution Summary ──")
        print(self._evolution_log.format())
        print(f"[ClawHarness] Best score: {best_score:.2f}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def evolution_log(self) -> EvolutionLog:
        return self._evolution_log

    @property
    def memory(self) -> ClawMemory:
        return self._memory

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_workflow_file(cls, path: str, config: HarnessConfig) -> ClawHarness:
        """
        Load a workflow from a JSON file.

        Handles:
        - API-format dict (keys are numeric strings with ``class_type``)
        - Prompt-keyed save (``{"prompt": {...}}``)
        - UI-format with ``nodes`` list (attempts sibling ``*_api.json`` first)
        """
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        if isinstance(data, dict) and "prompt" in data and isinstance(data["prompt"], dict):
            data = data["prompt"]
        elif isinstance(data, dict) and "nodes" in data and isinstance(data["nodes"], list):
            api_data = _try_sibling_api(path)
            if api_data is not None:
                print("[ClawHarness] Using sibling API-format workflow.")
                data = api_data
            else:
                print("[ClawHarness] ⚠  UI-format workflow; converting (widget names approximate).")
                data = _ui_to_api(data)

        return cls(base_workflow=data, config=config)

    @classmethod
    def from_workflow_dict(cls, workflow: dict, config: HarnessConfig) -> ClawHarness:
        return cls(base_workflow=workflow, config=config)


# ---------------------------------------------------------------------------
# UI → API conversion helpers
# ---------------------------------------------------------------------------


def _try_sibling_api(ui_path: str) -> dict | None:
    stem = Path(ui_path).stem
    parent = Path(ui_path).parent
    alt_stem = stem.removesuffix("_2512")
    for candidate in [parent / f"{stem}_api.json", parent / f"{alt_stem}_api.json"]:
        if candidate.exists():
            with open(candidate, encoding="utf-8") as fh:
                return json.load(fh)
    return None


def _ui_to_api(ui_data: dict) -> dict:
    """
    Best-effort conversion of ComfyUI UI-format to API format.
    Widget values are stored under ``__widget_N`` placeholder keys.
    """
    link_map: dict[int, list] = {}
    for lk in ui_data.get("links", []):
        if len(lk) >= 3:
            link_map[lk[0]] = [lk[1], lk[2]]

    api: dict[str, dict] = {}
    for node in ui_data.get("nodes", []):
        nid = str(node["id"])
        class_type = node.get("type", "Unknown")
        inputs: dict = {}

        for inp in node.get("inputs", []):
            link_id = inp.get("link")
            if link_id is not None and link_id in link_map:
                src = link_map[link_id]
                inputs[inp.get("name", "input")] = [str(src[0]), src[1]]

        for i, val in enumerate(node.get("widgets_values", [])):
            inputs[f"__widget_{i}"] = val

        api[nid] = {
            "class_type": class_type,
            "_meta": {"title": node.get("title", class_type)},
            "inputs": inputs,
        }
    return api
