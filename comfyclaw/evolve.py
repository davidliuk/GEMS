"""
SkillEvolver — Self-evolution engine for ComfyClaw's skill system.

Runs benchmark batches, clusters failure patterns, proposes skill mutations
(create/update/merge/delete), validates on a held-out set, and commits or
rolls back based on score improvement.

Topology complexity regularisation
-----------------------------------
When ``complexity_penalty > 0`` the effective score used for acceptance
decisions is::

    adjusted = raw_score - complexity_penalty * max(0, node_count - baseline_nodes) / baseline_nodes

This implements an Occam's-razor prior — penalising unnecessarily complex
workflows that are more likely to break.

Usage::

    evolver = SkillEvolver(skills_dir="skills/", benchmark_config=cfg)
    report = evolver.run_cycle(train_prompts, val_prompts)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import litellm

from .skill_manager import SkillManager, _EVOLVED_SKILLS_DIR
from .skill_store import SkillStore

log = logging.getLogger(__name__)

_MAX_LLM_RETRIES = 2


# ── Data types ─────────────────────────────────────────────────────────────


@dataclass
class FailureCluster:
    """A group of related failures across benchmark prompts."""

    name: str
    description: str
    failure_count: int
    affected_prompts: list[str]
    mean_score: float
    example_feedback: list[str]
    existing_skill: str | None = None


@dataclass
class MutationProposal:
    """A proposed change to the skill set."""

    mutation_type: str  # create | update | merge | delete
    target_skills: list[str]
    rationale: str
    failure_cluster: str
    proposed_changes: dict[str, str] = field(default_factory=dict)
    pre_score: float = 0.0
    post_score: float | None = None
    accepted: bool = False
    test_prompts: list[str] = field(default_factory=list)
    test_scores: list[float] = field(default_factory=list)


@dataclass
class EvolutionReport:
    """Summary of one evolution cycle."""

    cycle: int
    pre_mean_score: float
    post_mean_score: float
    mutations_proposed: int
    mutations_accepted: int
    mutations_rejected: int
    failure_clusters: list[FailureCluster]
    mutations: list[MutationProposal]
    duration_s: float = 0.0
    complexity_penalty_applied: float = 0.0

    def summary(self) -> str:
        delta = self.post_mean_score - self.pre_mean_score
        sign = "+" if delta >= 0 else ""
        lines = [
            f"Cycle {self.cycle}: score {self.pre_mean_score:.4f} -> "
            f"{self.post_mean_score:.4f} ({sign}{delta:.4f})",
            f"  Clusters: {len(self.failure_clusters)}",
            f"  Mutations: {self.mutations_proposed} proposed, "
            f"{self.mutations_accepted} accepted, "
            f"{self.mutations_rejected} rejected",
            f"  Duration: {self.duration_s:.1f}s",
        ]
        if self.complexity_penalty_applied > 0:
            lines.append(f"  Complexity penalty: {self.complexity_penalty_applied:.4f}")
        return "\n".join(lines)


# ── Prompts for LLM-driven analysis ───────────────────────────────────────

_CLUSTER_FAILURES_PROMPT = """\
You are an expert at analyzing image generation failures. Given the following
benchmark results (prompt, score, verifier feedback), cluster the failures
into categories.

Results:
{results_json}

Available skills: {skill_names}

Return ONLY a JSON array (no markdown fences, no explanation before/after):
[
  {{
    "name": "cluster_name_snake_case",
    "description": "What characterizes this failure pattern",
    "prompt_indices": [0, 3, 7],
    "existing_skill": "skill-name or null if no skill covers this"
  }}
]

Rules:
- Focus on actionable clusters where a skill could help.
- Ignore one-off failures (cluster must have >=2 prompts).
- Keep cluster names as snake_case identifiers.
- Return valid JSON only — no trailing commas, no comments.
"""

_PROPOSE_MUTATION_PROMPT = """\
You are a skill evolution engine. Given this failure cluster and the current
skill set, propose a skill mutation to address the failures.

Failure cluster: {cluster_json}

Current skills and their descriptions:
{skills_manifest}

Mutation types:
- "create": New skill for an uncovered failure pattern
- "update": Improve an existing skill's instructions
- "merge": Combine overlapping skills
- "delete": Remove a skill that is not helping

Return ONLY a JSON object (no markdown fences, no explanation before/after):
{{
  "mutation_type": "create|update|merge|delete",
  "target_skills": ["skill-name"],
  "rationale": "Why this mutation addresses the cluster",
  "proposed_changes": {{
    "name": "skill-name",
    "description": "One-line skill description (no newlines)",
    "body": "Full SKILL.md body with instructions"
  }}
}}

Rules:
- For "update", provide the improved body text.
- For "merge", provide the merged skill content.
- For "delete", target_skills lists which to remove; proposed_changes can be empty.
- The "body" field MUST be a single JSON string — escape newlines as \\n.
- Keep the body concise (under 3000 chars) with actionable node-level instructions.
- Return valid JSON only — no trailing commas, no comments.
"""

_GENERATE_TEST_PROMPTS_PROMPT = """\
Given this skill that was just created/updated for an image generation agent,
generate exactly 3 test prompts that exercise the skill's core capability.

Skill name: {skill_name}
Failure cluster it addresses: {cluster_name}
Affected prompts from the cluster: {affected_prompts}

Return ONLY a JSON array of 3 strings (no markdown fences):
["prompt 1", "prompt 2", "prompt 3"]

Rules:
- Each prompt should be a realistic image generation request.
- Prompts should test the specific failure pattern the skill addresses.
- Keep prompts concise (under 20 words each).
"""


# ── SkillEvolver ───────────────────────────────────────────────────────────


class SkillEvolver:
    """Autonomous skill evolution engine.

    Evolved skills (created, updated, merged) are written to a separate
    ``evolved_skills_dir`` so they don't mix with pre-defined skills.

    Parameters
    ----------
    skills_dir : Path to the pre-defined skills directory (read-only source).
    evolved_skills_dir : Path where evolved skills are stored.  Defaults to
                         the built-in ``skills_evolved/`` directory.
    llm_model  : LLM for failure analysis and mutation proposal.
    api_key    : API key for the LLM provider.
    min_improvement : Minimum score improvement to accept a mutation.
    max_mutations_per_cycle : Maximum mutations to attempt per cycle.
    complexity_penalty : Per-excess-node score penalty (0 = disabled).
    baseline_nodes : Reference node count for complexity penalty.
    """

    def __init__(
        self,
        skills_dir: str | Path | None = None,
        evolved_skills_dir: str | Path | None = None,
        llm_model: str = "anthropic/claude-sonnet-4-5",
        api_key: str = "",
        min_improvement: float = 0.02,
        max_mutations_per_cycle: int = 3,
        complexity_penalty: float = 0.02,
        baseline_nodes: int = 10,
    ) -> None:
        self.evolved_skills_dir = Path(evolved_skills_dir) if evolved_skills_dir else _EVOLVED_SKILLS_DIR
        self.evolved_skills_dir.mkdir(parents=True, exist_ok=True)
        self.store = SkillStore(self.evolved_skills_dir)
        self.skill_manager = SkillManager(skills_dir, evolved_skills_dir=str(self.evolved_skills_dir))
        self.llm_model = llm_model
        self.api_key = api_key
        self.min_improvement = min_improvement
        self.max_mutations_per_cycle = max_mutations_per_cycle
        self.complexity_penalty = complexity_penalty
        self.baseline_nodes = baseline_nodes

        if api_key:
            os.environ.setdefault("ANTHROPIC_API_KEY", api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        results: list[dict[str, Any]],
        run_validation_fn: Any | None = None,
        cycle: int = 1,
    ) -> EvolutionReport:
        """Execute one evolution cycle.

        Parameters
        ----------
        results : List of dicts with keys ``prompt``, ``score``, ``feedback``,
                  ``passed``, ``failed`` from a benchmark run.
        run_validation_fn : Optional callable that re-runs benchmark and
                           returns mean score.  Signature: ``() -> float``.
        cycle : Cycle number for logging.

        Returns
        -------
        EvolutionReport summarising what happened.
        """
        t0 = time.time()
        pre_mean = self._mean_score(results)
        log.info("Cycle %d: pre-evolution mean score = %.4f", cycle, pre_mean)

        # Step 1: Cluster failures
        failures = [r for r in results if r.get("score", 0) < 0.7]
        if not failures:
            log.info("No significant failures to address (all scores >= 0.7)")
            return EvolutionReport(
                cycle=cycle,
                pre_mean_score=pre_mean,
                post_mean_score=pre_mean,
                mutations_proposed=0,
                mutations_accepted=0,
                mutations_rejected=0,
                failure_clusters=[],
                mutations=[],
                duration_s=time.time() - t0,
            )

        clusters = self._cluster_failures(failures)
        log.info("Found %d failure clusters", len(clusters))

        # Step 2: Propose mutations
        mutations: list[MutationProposal] = []
        for cluster in clusters[: self.max_mutations_per_cycle]:
            proposal = self._propose_mutation(cluster)
            if proposal:
                proposal.pre_score = pre_mean
                mutations.append(proposal)

        # Step 3: Generate test prompts for each mutation
        for mutation in mutations:
            test_prompts = self._generate_test_prompts(mutation, clusters)
            mutation.test_prompts = test_prompts

        # Step 4: Apply and validate each mutation
        accepted = 0
        rejected = 0
        post_mean = pre_mean
        total_penalty = 0.0

        for mutation in mutations:
            log.info(
                "Applying mutation: %s on %s",
                mutation.mutation_type,
                mutation.target_skills,
            )
            try:
                self._apply_mutation(mutation)

                if run_validation_fn:
                    val_score = run_validation_fn()
                    penalty = self._compute_complexity_penalty(results)
                    adjusted_score = val_score - penalty
                    total_penalty += penalty

                    mutation.post_score = adjusted_score
                    improvement = adjusted_score - pre_mean

                    if improvement >= self.min_improvement:
                        mutation.accepted = True
                        accepted += 1
                        post_mean = adjusted_score
                        log.info(
                            "Mutation accepted: +%.4f (%.4f -> %.4f, penalty=%.4f)",
                            improvement, pre_mean, adjusted_score, penalty,
                        )
                        self.skill_manager = SkillManager(evolved_skills_dir=str(self.evolved_skills_dir))
                    else:
                        self._rollback_mutation(mutation)
                        rejected += 1
                        log.info(
                            "Mutation rejected: improvement %.4f < threshold %.4f",
                            improvement, self.min_improvement,
                        )
                else:
                    # No validation function — apply sanity checks before accepting.
                    # Verify the skill was actually created/updated on disk.
                    if self._verify_mutation_on_disk(mutation):
                        mutation.accepted = True
                        accepted += 1
                        self.skill_manager = SkillManager(evolved_skills_dir=str(self.evolved_skills_dir))
                        log.info("Mutation accepted (disk-verified, no validation fn)")
                    else:
                        self._rollback_mutation(mutation)
                        rejected += 1
                        log.warning("Mutation rejected: failed disk verification")

            except Exception as exc:
                log.error("Mutation failed: %s", exc)
                try:
                    self._rollback_mutation(mutation)
                except Exception:
                    pass
                rejected += 1

        return EvolutionReport(
            cycle=cycle,
            pre_mean_score=pre_mean,
            post_mean_score=post_mean,
            mutations_proposed=len(mutations),
            mutations_accepted=accepted,
            mutations_rejected=rejected,
            failure_clusters=clusters,
            mutations=mutations,
            duration_s=time.time() - t0,
            complexity_penalty_applied=total_penalty,
        )

    def run_multi_cycle(
        self,
        results: list[dict[str, Any]],
        run_validation_fn: Any | None = None,
        max_cycles: int = 10,
        convergence_threshold: float = 0.01,
    ) -> list[EvolutionReport]:
        """Run multiple evolution cycles until convergence."""
        reports: list[EvolutionReport] = []
        recent_improvements: list[float] = []

        for cycle in range(1, max_cycles + 1):
            report = self.run_cycle(results, run_validation_fn, cycle=cycle)
            reports.append(report)
            log.info(report.summary())

            delta = report.post_mean_score - report.pre_mean_score
            recent_improvements.append(delta)

            # Check convergence: avg improvement over last 3 cycles
            if len(recent_improvements) >= 3:
                avg_recent = sum(recent_improvements[-3:]) / 3
                if avg_recent < convergence_threshold:
                    log.info(
                        "Converged: avg improvement %.4f < threshold %.4f",
                        avg_recent, convergence_threshold,
                    )
                    break

            if report.mutations_accepted == 0 and report.mutations_proposed > 0:
                log.info("No mutations accepted this cycle — stopping early.")
                break

        return reports

    # ------------------------------------------------------------------
    # Private: failure clustering
    # ------------------------------------------------------------------

    def _cluster_failures(self, failures: list[dict]) -> list[FailureCluster]:
        """Use the LLM to cluster failures by pattern (with retry)."""
        results_for_llm = []
        for i, f in enumerate(failures[:30]):
            results_for_llm.append({
                "index": i,
                "prompt": f.get("prompt", ""),
                "score": f.get("score", 0),
                "failed_requirements": f.get("failed", [])[:5],
                "feedback_snippet": str(f.get("feedback", ""))[:200],
            })

        skill_names = ", ".join(self.store.list_skills())

        prompt = _CLUSTER_FAILURES_PROMPT.format(
            results_json=json.dumps(results_for_llm, indent=2),
            skill_names=skill_names or "(none)",
        )

        raw_clusters = self._llm_json_call(
            prompt, max_tokens=2048, expect_array=True
        )
        if raw_clusters is None:
            return []

        clusters: list[FailureCluster] = []
        for rc in raw_clusters:
            indices = rc.get("prompt_indices", [])
            affected = [
                failures[i]["prompt"]
                for i in indices
                if i < len(failures)
            ]
            scores = [
                failures[i].get("score", 0)
                for i in indices
                if i < len(failures)
            ]
            feedback = [
                str(failures[i].get("feedback", ""))[:100]
                for i in indices[:3]
                if i < len(failures)
            ]

            clusters.append(FailureCluster(
                name=rc.get("name", "unknown"),
                description=rc.get("description", ""),
                failure_count=len(indices),
                affected_prompts=affected,
                mean_score=sum(scores) / len(scores) if scores else 0.0,
                example_feedback=feedback,
                existing_skill=rc.get("existing_skill"),
            ))

        clusters.sort(key=lambda c: c.failure_count, reverse=True)
        return clusters

    # ------------------------------------------------------------------
    # Private: mutation proposal
    # ------------------------------------------------------------------

    def _propose_mutation(self, cluster: FailureCluster) -> MutationProposal | None:
        """Ask the LLM to propose a skill mutation for a failure cluster (with retry)."""
        manifest = "\n".join(
            f"- {s['name']}: {s['description']}"
            for s in self.skill_manager.get_manifest()
        )

        cluster_json = json.dumps({
            "name": cluster.name,
            "description": cluster.description,
            "failure_count": cluster.failure_count,
            "mean_score": cluster.mean_score,
            "example_feedback": cluster.example_feedback,
            "existing_skill": cluster.existing_skill,
            "affected_prompts": cluster.affected_prompts[:5],
        }, indent=2)

        prompt = _PROPOSE_MUTATION_PROMPT.format(
            cluster_json=cluster_json,
            skills_manifest=manifest or "(no skills)",
        )

        data = self._llm_json_call(prompt, max_tokens=4000, expect_array=False)
        if data is None:
            return None

        return MutationProposal(
            mutation_type=data.get("mutation_type", "update"),
            target_skills=data.get("target_skills", []),
            rationale=data.get("rationale", ""),
            failure_cluster=cluster.name,
            proposed_changes=data.get("proposed_changes", {}),
        )

    # ------------------------------------------------------------------
    # Private: apply / rollback mutations
    # ------------------------------------------------------------------

    def _apply_mutation(self, mutation: MutationProposal) -> None:
        """Apply a mutation to the skill store."""
        changes = mutation.proposed_changes
        mt = mutation.mutation_type

        if mt == "create":
            self.store.create_skill(
                name=changes.get("name", f"auto-{mutation.failure_cluster}"),
                description=changes.get("description", "Auto-generated skill"),
                body=changes.get("body", ""),
                metadata={"origin": "self-evolve", "cluster": mutation.failure_cluster},
            )

        elif mt == "update":
            for skill_name in mutation.target_skills:
                self.store.update_skill(
                    name=skill_name,
                    description=changes.get("description"),
                    body=changes.get("body"),
                )

        elif mt == "merge":
            self.store.merge_skills(
                names=mutation.target_skills,
                merged_name=changes.get("name", mutation.target_skills[0]),
                merged_description=changes.get("description", "Merged skill"),
                merged_body=changes.get("body", ""),
                delete_originals=True,
            )

        elif mt == "delete":
            for skill_name in mutation.target_skills:
                self.store.delete_skill(skill_name)

        else:
            raise ValueError(f"Unknown mutation type: {mt}")

    def _rollback_mutation(self, mutation: MutationProposal) -> None:
        """Rollback a mutation by restoring from the latest snapshot."""
        mt = mutation.mutation_type

        if mt == "create":
            name = mutation.proposed_changes.get(
                "name", f"auto-{mutation.failure_cluster}"
            )
            try:
                self.store.delete_skill(name)
            except FileNotFoundError:
                pass

        elif mt == "update":
            for skill_name in mutation.target_skills:
                try:
                    self.store.rollback_skill(skill_name)
                except FileNotFoundError:
                    pass

        elif mt == "merge":
            # Restore originals from snapshots
            for skill_name in mutation.target_skills:
                try:
                    self.store.rollback_skill(skill_name)
                except FileNotFoundError:
                    pass
            merged = mutation.proposed_changes.get("name")
            if merged and merged not in mutation.target_skills:
                try:
                    self.store.delete_skill(merged)
                except FileNotFoundError:
                    pass

        elif mt == "delete":
            for skill_name in mutation.target_skills:
                try:
                    self.store.rollback_skill(skill_name)
                except FileNotFoundError:
                    pass

    # ------------------------------------------------------------------
    # LLM call with retry
    # ------------------------------------------------------------------

    def _llm_json_call(
        self,
        prompt: str,
        max_tokens: int = 2048,
        expect_array: bool = False,
    ) -> Any | None:
        """Call the LLM and parse JSON from response, retrying on parse failure."""
        last_error: Exception | None = None
        for attempt in range(_MAX_LLM_RETRIES + 1):
            try:
                messages: list[dict] = [{"role": "user", "content": prompt}]
                if attempt > 0:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Your previous response had a JSON parse error: {last_error}. "
                            "Please return ONLY valid JSON with no markdown fences or extra text."
                        ),
                    })

                resp = litellm.completion(
                    model=self.llm_model,
                    max_tokens=max_tokens,
                    messages=messages,
                )
                text = (resp.choices[0].message.content or "").strip()

                # Strip markdown fences if present
                text = re.sub(r"^```(?:json)?\s*\n?", "", text)
                text = re.sub(r"\n?```\s*$", "", text)

                if expect_array:
                    m = re.search(r"\[.*\]", text, re.DOTALL)
                    return json.loads(m.group() if m else text)
                else:
                    m = re.search(r"\{.*\}", text, re.DOTALL)
                    return json.loads(m.group() if m else text)

            except (json.JSONDecodeError, AttributeError) as exc:
                last_error = exc
                log.warning(
                    "JSON parse failed (attempt %d/%d): %s",
                    attempt + 1, _MAX_LLM_RETRIES + 1, exc,
                )
            except Exception as exc:
                log.error("LLM call failed: %s", exc)
                return None

        log.error("All %d JSON parse attempts failed", _MAX_LLM_RETRIES + 1)
        return None

    # ------------------------------------------------------------------
    # Complexity penalty
    # ------------------------------------------------------------------

    def _compute_complexity_penalty(self, results: list[dict]) -> float:
        """Compute average complexity penalty across results."""
        if self.complexity_penalty <= 0:
            return 0.0
        total = 0.0
        count = 0
        for r in results:
            nc = r.get("node_count", 0)
            if nc > 0:
                excess = max(0, nc - self.baseline_nodes)
                total += self.complexity_penalty * excess / max(self.baseline_nodes, 1)
                count += 1
        return total / count if count else 0.0

    # ------------------------------------------------------------------
    # Test prompt generation
    # ------------------------------------------------------------------

    def _generate_test_prompts(
        self,
        mutation: MutationProposal,
        clusters: list[FailureCluster],
    ) -> list[str]:
        """Generate test prompts for a mutation to enable grounded validation."""
        cluster = next(
            (c for c in clusters if c.name == mutation.failure_cluster), None
        )
        if not cluster:
            return []

        prompt = _GENERATE_TEST_PROMPTS_PROMPT.format(
            skill_name=mutation.proposed_changes.get("name", mutation.failure_cluster),
            cluster_name=cluster.name,
            affected_prompts=json.dumps(cluster.affected_prompts[:5]),
        )

        result = self._llm_json_call(prompt, max_tokens=512, expect_array=True)
        if result and isinstance(result, list):
            return [str(p) for p in result[:5]]
        return cluster.affected_prompts[:3]

    # ------------------------------------------------------------------
    # Disk verification
    # ------------------------------------------------------------------

    def _verify_mutation_on_disk(self, mutation: MutationProposal) -> bool:
        """Verify a mutation was successfully applied by checking disk state."""
        mt = mutation.mutation_type
        changes = mutation.proposed_changes

        if mt == "create":
            name = changes.get("name", f"auto-{mutation.failure_cluster}")
            skill_md = self.evolved_skills_dir / name / "SKILL.md"
            if not skill_md.exists():
                log.warning("Created skill %r not found on disk", name)
                return False
            content = skill_md.read_text(encoding="utf-8")
            if len(content) < 50:
                log.warning("Created skill %r is suspiciously short (%d chars)", name, len(content))
                return False
            return True

        elif mt == "update":
            for skill_name in mutation.target_skills:
                skill_md = self.evolved_skills_dir / skill_name / "SKILL.md"
                if not skill_md.exists():
                    log.warning("Updated skill %r not found on disk", skill_name)
                    return False
            return True

        elif mt == "delete":
            for skill_name in mutation.target_skills:
                skill_dir = self.evolved_skills_dir / skill_name
                if skill_dir.exists():
                    log.warning("Deleted skill %r still exists on disk", skill_name)
                    return False
            return True

        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_score(results: list[dict]) -> float:
        scores = [r.get("score", 0.0) for r in results]
        return sum(scores) / len(scores) if scores else 0.0
