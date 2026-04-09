"""Unit tests for ClawHarness (ComfyClient and Anthropic mocked)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from comfyclaw.harness import ClawHarness, EvolutionEntry, HarnessConfig

# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def cfg() -> HarnessConfig:
    return HarnessConfig(
        api_key="sk-ant-test",
        server_address="127.0.0.1:9999",  # nothing listening here
        max_iterations=3,
        success_threshold=0.9,
        sync_port=0,  # disable sync server
        evolve_from_best=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_agent(rationale: str = "test rationale") -> MagicMock:
    agent = MagicMock()
    agent.plan_and_patch.return_value = rationale
    return agent


def _mock_verifier(score: float = 0.7) -> MagicMock:
    from comfyclaw.verifier import RequirementCheck, VerifierResult

    result = VerifierResult(
        score=score,
        checks=[RequirementCheck("Q1?", "yes", True)],
        passed=["Q1?"],
        failed=[],
        overall_assessment=f"Score {score}",
        evolution_suggestions=[],
    )
    verifier = MagicMock()
    verifier.verify.return_value = result
    verifier.model = "claude-test"
    verifier.client = MagicMock()
    verifier.client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="Brief experience.")]
    )
    return verifier


def _make_harness(
    base_workflow: dict,
    cfg: HarnessConfig,
    agent: MagicMock | None = None,
    verifier: MagicMock | None = None,
    client: MagicMock | None = None,
) -> ClawHarness:
    h = ClawHarness(base_workflow, cfg)
    h._agent = agent or _mock_agent()
    h._verifier = verifier or _mock_verifier()
    h._client = client or MagicMock()
    return h


def _mock_comfy_client(image_bytes: bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20) -> MagicMock:
    client = MagicMock()
    client.queue_prompt.return_value = {"prompt_id": "test-prompt-id"}
    client.wait_for_completion.return_value = {
        "outputs": {"7": {"images": [{"filename": "test.png", "subfolder": "", "type": "output"}]}}
    }
    client.collect_images.return_value = [image_bytes]
    return client


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_returns_none(self, minimal_workflow: dict, cfg: HarnessConfig) -> None:
        h = _make_harness(minimal_workflow, cfg)
        result = h.run("a red fox", dry_run=True)
        assert result is None

    def test_dry_run_no_http_calls(self, minimal_workflow: dict, cfg: HarnessConfig) -> None:
        mock_client = MagicMock()
        h = _make_harness(minimal_workflow, cfg, client=mock_client)
        h.run("a red fox", dry_run=True)
        mock_client.queue_prompt.assert_not_called()

    def test_dry_run_agent_is_called(self, minimal_workflow: dict, cfg: HarnessConfig) -> None:
        mock_agent = _mock_agent()
        h = _make_harness(minimal_workflow, cfg, agent=mock_agent)
        h.run("a red fox", dry_run=True)
        mock_agent.plan_and_patch.assert_called_once()


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class TestEarlyStop:
    def test_stops_at_threshold(self, minimal_workflow: dict, cfg: HarnessConfig) -> None:
        mock_verifier = _mock_verifier(score=0.95)  # above threshold 0.9
        mock_client = _mock_comfy_client()
        h = _make_harness(minimal_workflow, cfg, verifier=mock_verifier, client=mock_client)
        h.run("a fox")
        # Agent should have been called exactly once
        assert h._agent.plan_and_patch.call_count == 1

    def test_runs_all_iters_when_below_threshold(
        self, minimal_workflow: dict, cfg: HarnessConfig
    ) -> None:
        mock_verifier = _mock_verifier(score=0.5)  # always below threshold
        mock_client = _mock_comfy_client()
        h = _make_harness(minimal_workflow, cfg, verifier=mock_verifier, client=mock_client)
        h.run("a fox")
        assert h._agent.plan_and_patch.call_count == cfg.max_iterations


# ---------------------------------------------------------------------------
# Topology accumulation
# ---------------------------------------------------------------------------


class TestTopologyAccumulation:
    def test_evolve_from_best_starts_from_best_snapshot(
        self, minimal_workflow: dict, cfg: HarnessConfig
    ) -> None:
        """
        With evolve_from_best=True, iteration 2 should receive the workflow
        snapshot that was produced by iteration 1 (when iter 1 had a higher score).
        """
        # Track what workflow the agent sees each iteration
        seen_node_counts: list[int] = []

        def track_patch(workflow_manager, **kwargs):
            seen_node_counts.append(len(workflow_manager))
            # Simulate adding a node on iter 1
            if len(seen_node_counts) == 1:
                workflow_manager.add_node("LoraLoader", lora_name="test.lora")
            return "rationale"

        mock_agent = MagicMock()
        mock_agent.plan_and_patch.side_effect = track_patch

        # Iter 1 score = 0.7, iter 2 score = 0.5 (so best is iter 1)
        from comfyclaw.verifier import VerifierResult

        iter_scores = [0.7, 0.5]
        score_iter = iter(iter_scores)

        def make_result():
            s = next(score_iter, 0.5)
            return VerifierResult(
                score=s,
                checks=[],
                passed=[],
                failed=[],
                overall_assessment=f"{s}",
                evolution_suggestions=[],
            )

        mock_verifier = MagicMock()
        mock_verifier.verify.side_effect = lambda *a, **kw: make_result()
        mock_verifier.model = "test"
        mock_verifier.client = MagicMock()
        mock_verifier.client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="exp")]
        )

        mock_client = _mock_comfy_client()
        cfg.max_iterations = 2
        cfg.success_threshold = 1.0  # never stop early
        h = _make_harness(
            minimal_workflow, cfg, agent=mock_agent, verifier=mock_verifier, client=mock_client
        )
        h.run("test")

        # Iter 1 starts with base workflow (3 nodes)
        assert seen_node_counts[0] == 3
        # Iter 2 should start from iter 1's result (3 base + 1 added = 4)
        assert seen_node_counts[1] == 4

    def test_reset_each_iter_ignores_best(self, minimal_workflow: dict, cfg: HarnessConfig) -> None:
        """With evolve_from_best=False every iteration starts from base."""
        cfg.evolve_from_best = False
        seen_node_counts: list[int] = []

        def track_patch(workflow_manager, **kwargs):
            seen_node_counts.append(len(workflow_manager))
            workflow_manager.add_node("LoraLoader", lora_name="x.lora")
            return "r"

        mock_agent = MagicMock()
        mock_agent.plan_and_patch.side_effect = track_patch
        mock_client = _mock_comfy_client()
        cfg.max_iterations = 2
        cfg.success_threshold = 1.0
        h = _make_harness(minimal_workflow, cfg, agent=mock_agent, client=mock_client)
        h.run("test")

        # Both iterations should start with 3 nodes (base)
        assert seen_node_counts[0] == 3
        assert seen_node_counts[1] == 3


# ---------------------------------------------------------------------------
# Evolution log
# ---------------------------------------------------------------------------


class TestEvolutionLog:
    def test_log_populated_after_run(self, minimal_workflow: dict, cfg: HarnessConfig) -> None:
        mock_client = _mock_comfy_client()
        cfg.max_iterations = 2
        cfg.success_threshold = 1.0
        h = _make_harness(minimal_workflow, cfg, client=mock_client)
        h.run("test")
        assert len(h.evolution_log) == 2

    def test_added_nodes_tracked(self, minimal_workflow: dict, cfg: HarnessConfig) -> None:
        def add_node_patch(workflow_manager, **kwargs):
            workflow_manager.add_node("LoraLoader", lora_name="x.safetensors")
            return "r"

        mock_agent = MagicMock()
        mock_agent.plan_and_patch.side_effect = add_node_patch
        mock_client = _mock_comfy_client()
        cfg.max_iterations = 1
        h = _make_harness(minimal_workflow, cfg, agent=mock_agent, client=mock_client)
        h.run("test")
        entry = h.evolution_log.entries[0]
        assert len(entry.node_ids_added) == 1

    def test_entry_summary_format(self) -> None:
        entry = EvolutionEntry(
            iteration=1,
            node_count_before=5,
            node_count_after=7,
            node_ids_added=["6", "7"],
            rationale="Added LoRA and ControlNet.",
            verifier_score=0.72,
        )
        s = entry.summary()
        assert "Iter 1" in s
        assert "5→7" in s
        assert "0.72" in s


# ---------------------------------------------------------------------------
# ComfyUI error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_comfy_error_recorded_in_memory(
        self, minimal_workflow: dict, cfg: HarnessConfig
    ) -> None:
        mock_client = MagicMock()
        mock_client.queue_prompt.return_value = {"prompt_id": "pid"}
        mock_client.wait_for_completion.return_value = {
            "error": "ComfyUI execution error: Float8_e4m3fn MPS issue"
        }
        cfg.max_iterations = 1
        cfg.max_repair_attempts = 0  # disable repairs for this test
        h = _make_harness(minimal_workflow, cfg, client=mock_client)
        h.run("test")
        assert len(h.memory) == 1
        assert h.memory.attempts[0].verifier_score == 0.0
        assert "error" in h.memory.attempts[0].failed[0].lower()

    def test_queue_exception_continues_loop(
        self, minimal_workflow: dict, cfg: HarnessConfig
    ) -> None:
        mock_client = MagicMock()
        mock_client.queue_prompt.side_effect = Exception("Connection refused")
        cfg.max_iterations = 1
        cfg.max_repair_attempts = 0  # disable repairs for this test
        h = _make_harness(minimal_workflow, cfg, client=mock_client)
        result = h.run("test")
        assert result is None  # no image produced, but no crash


# ---------------------------------------------------------------------------
# Repair loop
# ---------------------------------------------------------------------------


class TestRepairLoop:
    def test_queue_error_triggers_agent_repair(
        self, minimal_workflow: dict, cfg: HarnessConfig
    ) -> None:
        """When queue_prompt raises on the first attempt the agent should be
        called a second time (repair round 1) before the iteration is abandoned."""
        mock_client = MagicMock()
        # First call raises, second call succeeds (repair accepted)
        mock_client.queue_prompt.side_effect = [
            Exception("HTTP 400: bad node"),
            {"prompt_id": "pid-repaired"},
        ]
        mock_client.wait_for_completion.return_value = {"outputs": {}}
        mock_client.collect_images.return_value = []

        cfg.max_iterations = 1
        cfg.max_repair_attempts = 1
        mock_agent = _mock_agent()
        h = _make_harness(minimal_workflow, cfg, agent=mock_agent, client=mock_client)
        h.run("test")

        # Agent called twice: once for the normal evolution, once for the repair
        assert mock_agent.plan_and_patch.call_count == 2

    def test_repair_feedback_contains_error_message(
        self, minimal_workflow: dict, cfg: HarnessConfig
    ) -> None:
        """The feedback given to the agent during repair must include the
        exact error message returned by ComfyUI."""
        received_feedbacks: list[str] = []

        def capture_patch(workflow_manager, verifier_feedback=None, **kwargs):
            if verifier_feedback is not None:
                received_feedbacks.append(verifier_feedback)
            return "rationale"

        mock_client = MagicMock()
        mock_client.queue_prompt.side_effect = [
            Exception("return_type_mismatch: vae slot wrong"),
            {"prompt_id": "pid"},
        ]
        mock_client.wait_for_completion.return_value = {"outputs": {}}
        mock_client.collect_images.return_value = []

        cfg.max_iterations = 1
        cfg.max_repair_attempts = 1
        mock_agent = MagicMock()
        mock_agent.plan_and_patch.side_effect = capture_patch
        h = _make_harness(minimal_workflow, cfg, agent=mock_agent, client=mock_client)
        h.run("test")

        # At least one feedback must contain the error string
        assert any("return_type_mismatch" in fb for fb in received_feedbacks)
        assert any("Repair Required" in fb for fb in received_feedbacks)

    def test_repair_exhausted_records_error(
        self, minimal_workflow: dict, cfg: HarnessConfig
    ) -> None:
        """If all repair attempts also fail, the iteration should be abandoned
        and the error recorded in memory without crashing."""
        mock_client = MagicMock()
        mock_client.queue_prompt.side_effect = Exception("persistent error")

        cfg.max_iterations = 1
        cfg.max_repair_attempts = 2
        h = _make_harness(minimal_workflow, cfg, client=mock_client)
        result = h.run("test")

        assert result is None
        assert len(h.memory) == 1
        assert "persistent error" in h.memory.attempts[0].failed[0]
        # Agent called 1 (normal) + 2 (repairs) = 3 times
        assert h._agent.plan_and_patch.call_count == 3

    def test_repair_success_produces_image(
        self, minimal_workflow: dict, cfg: HarnessConfig
    ) -> None:
        """A successful repair should result in a verified image being returned."""
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20

        mock_client = MagicMock()
        mock_client.queue_prompt.side_effect = [
            Exception("HTTP 400"),
            {"prompt_id": "pid-ok"},
        ]
        mock_client.wait_for_completion.return_value = {
            "outputs": {"7": {"images": [{"filename": "t.png", "subfolder": "", "type": "output"}]}}
        }
        mock_client.collect_images.return_value = [png]

        cfg.max_iterations = 1
        cfg.max_repair_attempts = 1
        h = _make_harness(minimal_workflow, cfg, client=mock_client)
        result = h.run("test")

        assert result == png

    def test_execution_error_triggers_repair(
        self, minimal_workflow: dict, cfg: HarnessConfig
    ) -> None:
        """An execution-time error from ComfyUI (not HTTP 400) should also
        trigger the repair loop."""
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
        call_count = {"n": 0}

        def queue_side_effect(_wf):
            call_count["n"] += 1
            return {"prompt_id": f"pid-{call_count['n']}"}

        completion_responses = [
            {"error": "ComfyUI execution error: node 10 failed"},  # first attempt
            {
                "outputs": {
                    "7": {"images": [{"filename": "t.png", "subfolder": "", "type": "output"}]}
                }
            },  # repair
        ]
        completion_iter = iter(completion_responses)

        mock_client = MagicMock()
        mock_client.queue_prompt.side_effect = queue_side_effect
        mock_client.wait_for_completion.side_effect = lambda *a, **kw: next(completion_iter)
        mock_client.collect_images.return_value = [png]

        cfg.max_iterations = 1
        cfg.max_repair_attempts = 1
        h = _make_harness(minimal_workflow, cfg, client=mock_client)
        result = h.run("test")

        assert result == png
        # Agent: 1 normal + 1 execution repair = 2 calls
        assert h._agent.plan_and_patch.call_count == 2

    def test_build_repair_feedback_content(
        self, minimal_workflow: dict, cfg: HarnessConfig
    ) -> None:
        """_build_repair_feedback should include the error message and fix hints."""
        h = ClawHarness(minimal_workflow, cfg)
        fb = h._build_repair_feedback("vae slot mismatch", last_result=None)

        assert "vae slot mismatch" in fb
        assert "Repair Required" in fb
        assert "inspect_workflow" in fb
        assert "finalize_workflow" in fb


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_context_manager_calls_start_stop(
        self, minimal_workflow: dict, cfg: HarnessConfig
    ) -> None:
        h = ClawHarness(minimal_workflow, cfg)
        h._sync = MagicMock()
        with h:
            h._sync.start.assert_called_once()
        h._sync.stop.assert_called_once()


# ---------------------------------------------------------------------------
# from_workflow_file
# ---------------------------------------------------------------------------


class TestFromWorkflowFile:
    def test_loads_api_format(self, tmp_path, minimal_workflow: dict, cfg: HarnessConfig) -> None:
        wf_path = tmp_path / "wf.json"
        wf_path.write_text(json.dumps(minimal_workflow))
        h = ClawHarness.from_workflow_file(str(wf_path), cfg)
        assert h.base_workflow == minimal_workflow

    def test_loads_prompt_keyed_format(
        self, tmp_path, minimal_workflow: dict, cfg: HarnessConfig
    ) -> None:
        wf_path = tmp_path / "wf.json"
        wf_path.write_text(json.dumps({"prompt": minimal_workflow}))
        h = ClawHarness.from_workflow_file(str(wf_path), cfg)
        assert h.base_workflow == minimal_workflow


# ---------------------------------------------------------------------------
# Pinned image model
# ---------------------------------------------------------------------------


@pytest.fixture()
def workflow_with_loader() -> dict:
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "original_model.safetensors"},
        },
        "2": {"class_type": "KSampler", "inputs": {"steps": 20, "model": ["1", 0]}},
        "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
    }


class TestPinnedImageModel:
    def test_base_workflow_gets_model_applied_on_init(
        self, workflow_with_loader: dict, cfg: HarnessConfig
    ) -> None:
        cfg.image_model = "Qwen/Qwen-Image-2512"
        h = ClawHarness(workflow_with_loader, cfg)
        assert h.base_workflow["1"]["inputs"]["ckpt_name"] == "Qwen/Qwen-Image-2512"

    def test_original_workflow_not_mutated(
        self, workflow_with_loader: dict, cfg: HarnessConfig
    ) -> None:
        cfg.image_model = "new_model.safetensors"
        ClawHarness(workflow_with_loader, cfg)
        # The caller's dict must not be mutated
        assert workflow_with_loader["1"]["inputs"]["ckpt_name"] == "original_model.safetensors"

    def test_no_pin_leaves_model_unchanged(
        self, workflow_with_loader: dict, cfg: HarnessConfig
    ) -> None:
        cfg.image_model = None
        h = ClawHarness(workflow_with_loader, cfg)
        assert h.base_workflow["1"]["inputs"]["ckpt_name"] == "original_model.safetensors"

    def test_pin_passed_to_agent(self, workflow_with_loader: dict, cfg: HarnessConfig) -> None:
        cfg.image_model = "Qwen/Qwen-Image-2512"
        h = ClawHarness(workflow_with_loader, cfg)
        assert h._agent.pinned_image_model == "Qwen/Qwen-Image-2512"

    def test_model_reapplied_after_agent_adds_loader(
        self, workflow_with_loader: dict, cfg: HarnessConfig
    ) -> None:
        """Agent adds a second CheckpointLoaderSimple; pin must cover it too."""
        cfg.image_model = "pinned.safetensors"
        cfg.max_iterations = 1
        h = _make_harness(workflow_with_loader, cfg)

        def _agent_adds_loader(workflow_manager, **_kw):
            # Simulate agent adding a new loader node
            workflow_manager.add_node(
                "CheckpointLoaderSimple",
                nickname="AgentAddedLoader",
                ckpt_name="agent_chose_this.safetensors",
            )
            return "added loader"

        h._agent.plan_and_patch = _agent_adds_loader
        # Mock client to avoid real ComfyUI calls — return empty history
        h._client = MagicMock()
        h._client.queue_prompt.return_value = {"prompt_id": "pid"}
        h._client.wait_for_completion.return_value = {"outputs": {}}
        h._client.collect_images.return_value = []
        h.run("test prompt")

        # All CheckpointLoaderSimple nodes must be pinned after the loop
        for node in h.base_workflow.values():
            if node.get("class_type") == "CheckpointLoaderSimple":
                assert node["inputs"]["ckpt_name"] == "pinned.safetensors", (
                    f"Expected pinned model but got {node['inputs']['ckpt_name']!r}"
                )
