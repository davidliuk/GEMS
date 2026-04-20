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

from .skill_manager import SkillManager, _EVOLVED_SKILLS_ROOT
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
class SuccessCluster:
    """A group of high-scoring results sharing a common strategy pattern."""

    name: str
    description: str
    success_count: int
    affected_prompts: list[str]
    mean_score: float
    example_strategies: list[str]
    key_tools: list[str] = field(default_factory=list)
    existing_skill: str | None = None


@dataclass
class MutationProposal:
    """A proposed change to the skill set."""

    mutation_type: str  # create | update | merge | delete | reinforce
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
    success_clusters: list[SuccessCluster] = field(default_factory=list)
    reinforce_mutations: int = 0

    def summary(self) -> str:
        delta = self.post_mean_score - self.pre_mean_score
        sign = "+" if delta >= 0 else ""
        lines = [
            f"Cycle {self.cycle}: score {self.pre_mean_score:.4f} -> "
            f"{self.post_mean_score:.4f} ({sign}{delta:.4f})",
            f"  Failure clusters: {len(self.failure_clusters)}",
            f"  Success clusters: {len(self.success_clusters)}",
            f"  Mutations: {self.mutations_proposed} proposed, "
            f"{self.mutations_accepted} accepted, "
            f"{self.mutations_rejected} rejected"
            f" ({self.reinforce_mutations} reinforce)",
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

# ---------------------------------------------------------------------------
# Description-writing guide — shared across *every* code path that synthesises
# a SKILL.md ``description``.  Keep it here so there's one source of truth and
# all writers (SkillEvolver mutations + run_benchmark's learned-errors /
# learned-successes synthesizers) benefit from the same triggering advice.
#
# Why this matters: the ``description`` field is the ONLY part of a skill the
# agent sees in its system prompt at plan time.  Vague "this skill is about X"
# descriptions cause chronic under-triggering — agents know the skill exists
# but never call ``read_skill`` on it.  The triggering playbook below is
# lifted (and adapted for the image-gen domain) from Anthropic's official
# ``skill-creator`` skill's "Description Optimization" section.
# ---------------------------------------------------------------------------

DESCRIPTION_WRITING_GUIDE = """\
Description-writing guide (THIS IS THE MOST IMPORTANT FIELD):
The ``description`` is the ONLY thing the agent sees about this skill in its
system prompt.  The full body is NOT loaded until the agent calls
``read_skill``.  Today's LLMs routinely under-trigger skills — if your
description only states what the skill IS, the agent will skip it.  You MUST
make the description do TWO things in a single line:

  1. Say briefly WHAT the skill contains (a capability summary).
  2. Say PUSHILY WHEN to consult it — give concrete trigger signals the agent
     will actually encounter at plan time (prompt keywords, error strings,
     verifier feedback phrases, workflow states, number of objects, etc.).

Use imperative phrasing: "Consult when…", "Read this before…",
"Apply whenever…", "Use this for any prompt containing…".  It is OK —
encouraged, even — to be a little pushy: include synonyms and closely
related phrasings so the agent notices even when the user phrasing varies.

GOOD (note the explicit WHEN clauses):
  "Counting-and-composition playbook for >=4 distinct objects. Consult
   whenever the prompt contains a number word ('three', 'four', 'six',
   'seven', 'eight') AND two or more object types ('and', 'with'), OR when
   the verifier flags missing/extra objects; includes proven node orderings
   and region-split strategies that hit 0.956+ on geneval2."

  "ComfyUI validation-error recovery kit. Read this BEFORE editing any
   workflow when the previous attempt raised 'exception_during_inner_validation',
   'prompt_outputs_failed_validation', 'conditioning_to_strength > 1.0', or
   broken 'positive'/'CONDITIONING' edges; includes the exact node-level fixes
   that unblocked those errors in past runs."

BAD (vague, states WHAT but not WHEN — agent will skip these):
  "Proven strategies for high-scoring workflows."
  "ComfyUI error prevention guide."
  "Helper skill for the agent."

Additional rules:
- Single line, no embedded newlines.
- Start with the capability, then pivot to the trigger clause ("Consult …",
  "Read … when …").
- Name at least 2–3 concrete trigger signals (words, phrases, error strings,
  workflow states).  The more specific, the better.
- Length target: 30–80 words.  Longer than the old one-liners, but still
  one line of text.
"""


_PROPOSE_MUTATION_PROMPT = """\
You are a skill evolution engine. Given this cluster and the current
skill set, propose a skill mutation.

Cluster: {cluster_json}

Built-in skills (curated, do NOT duplicate or override these):
{builtin_skills_manifest}

Evolved skills (auto-generated, can be updated/merged/deleted):
{evolved_skills_manifest}

Mutation types:
- "create": New skill for an uncovered pattern
- "update": Improve an existing skill's instructions
- "merge": Combine overlapping skills
- "delete": Remove a skill that is not helping
- "reinforce": Codify a successful strategy as a reusable skill or strengthen an existing skill with proven techniques

Return ONLY a JSON object (no markdown fences, no explanation before/after):
{{
  "mutation_type": "create|update|merge|delete|reinforce",
  "target_skills": ["skill-name"],
  "rationale": "Why this mutation addresses the cluster",
  "proposed_changes": {{
    "name": "skill-name",
    "description": "<what it does>. <pushy 'when to consult' signals> (see guide below)",
    "body": "Full SKILL.md body with instructions",
    "tags": ["optional", "extra", "tags"]
  }}
}}

""" + DESCRIPTION_WRITING_GUIDE + """
Tag guidance:
- ``proposed_changes.tags`` is OPTIONAL.  The runner will automatically attach
  ``agent``, ``model:<short>``, and ``bench:<short>`` tags based on the current
  run context — you do NOT need to include those.
- Only add a tag here when it meaningfully narrows applicability, e.g.:
  * ``"topic:counting"`` for counting-specific skills
  * ``"topic:spatial"`` for spatial-layout skills
  * ``"topic:text-rendering"`` for text-in-image skills
- Never add tags that reference specific models/benchmarks — those are injected.

Rules:
- NEVER create a skill that restates what a built-in skill already covers. \
Built-in skills are authoritative — evolved skills must add NEW knowledge on top.
- If the cluster's failures relate to a built-in skill's domain, prefer "update" on \
an existing evolved skill or "create" a NARROW evolved skill that references the \
built-in skill by name (e.g. "Apply regional-control first, then additionally ...").
- The body MUST start with a "## Complements" section listing built-in skills the \
agent should read first, e.g. "## Complements\\nRead `regional-control` and `spatial` \
before applying this skill."
- Only target evolved skills for "update", "merge", or "delete" — never modify built-in skills.
- For "reinforce", codify proven successful strategies into a new or existing skill. \
Include specific tool sequences, parameter values, and node configurations that worked.
- The "body" field MUST be a single JSON string — escape newlines as \\n.
- Keep the body concise (under 3000 chars) with actionable node-level instructions.
- Return valid JSON only — no trailing commas, no comments.
"""

_CLUSTER_SUCCESSES_PROMPT = """\
You are an expert at analyzing image generation successes. Given the following
high-scoring benchmark results (prompt, score, tools used, rationale),
identify reusable strategy patterns that the agent can apply to future prompts.

Results:
{results_json}

Available skills: {skill_names}

Return ONLY a JSON array (no markdown fences, no explanation before/after):
[
  {{
    "name": "pattern_name_snake_case",
    "description": "What strategy made these prompts succeed",
    "prompt_indices": [0, 2, 5],
    "key_tools": ["tool1", "tool2"],
    "existing_skill": "skill-name or null if no existing skill covers this"
  }}
]

Rules:
- Focus on actionable patterns an agent can replicate on new prompts.
- Ignore trivially easy prompts (cluster must have >=3 prompts).
- Look for common tool sequences, node configurations, or parameter strategies.
- If a pattern already matches an existing skill, set existing_skill to that skill name.
- Keep cluster names as snake_case identifiers.
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
    success_threshold : Minimum score to consider a result a "success" for
                        pattern extraction (default 0.9).
    auto_tags :         Tags automatically attached to every evolved skill
                        created/updated in this cycle (e.g.
                        ``["model:longcat", "bench:geneval2"]``).  These are
                        merged with any LLM-proposed tags and with the
                        auto-injected ``"agent"`` tag.  This is what lets
                        ``SkillManager.build_available_skills_xml`` filter
                        evolved skills by ``model:<short>`` and ``bench:<short>``.
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
        success_threshold: float = 0.9,
        auto_tags: list[str] | None = None,
    ) -> None:
        self.evolved_skills_dir = Path(evolved_skills_dir) if evolved_skills_dir else _EVOLVED_SKILLS_ROOT
        self.evolved_skills_dir.mkdir(parents=True, exist_ok=True)
        self.store = SkillStore(self.evolved_skills_dir)
        self.skill_manager = SkillManager(skills_dir, evolved_skills_dir=str(self.evolved_skills_dir))
        self.llm_model = llm_model
        self.api_key = api_key
        self.min_improvement = min_improvement
        self.max_mutations_per_cycle = max_mutations_per_cycle
        self.complexity_penalty = complexity_penalty
        self.baseline_nodes = baseline_nodes
        self.success_threshold = success_threshold
        self.auto_tags: list[str] = list(auto_tags or [])

        if api_key:
            from .agent import _set_llm_api_key
            _set_llm_api_key(api_key, llm_model)

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

        # Step 1a: Cluster failures
        failures = [r for r in results if r.get("score", r.get("best_score", 0)) < 0.7]
        failure_clusters: list[FailureCluster] = []
        if failures:
            failure_clusters = self._cluster_failures(failures)
            log.info("Found %d failure clusters", len(failure_clusters))
        else:
            log.info("No significant failures to address (all scores >= 0.7)")

        # Step 1b: Cluster successes
        successes = [
            r for r in results
            if r.get("score", r.get("best_score", 0)) >= self.success_threshold
        ]
        success_clusters: list[SuccessCluster] = []
        if successes:
            success_clusters = self._cluster_successes(successes)
            log.info("Found %d success clusters from %d high-scoring results",
                     len(success_clusters), len(successes))

        if not failure_clusters and not success_clusters:
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

        # Step 2: Propose mutations from both failures and successes
        mutations: list[MutationProposal] = []
        reinforce_count = 0

        for cluster in failure_clusters[: self.max_mutations_per_cycle]:
            proposal = self._propose_mutation(cluster)
            if proposal:
                proposal.pre_score = pre_mean
                mutations.append(proposal)

        remaining_slots = max(0, self.max_mutations_per_cycle - len(mutations))
        reinforce_slots = max(1, remaining_slots) if success_clusters else 0
        for cluster in success_clusters[:reinforce_slots]:
            proposal = self._propose_reinforce_mutation(cluster)
            if proposal:
                proposal.pre_score = pre_mean
                mutations.append(proposal)
                reinforce_count += 1

        # Step 3: Generate test prompts (only useful when validation is available)
        if run_validation_fn:
            for mutation in mutations:
                test_prompts = self._generate_test_prompts(mutation, failure_clusters)
                mutation.test_prompts = test_prompts

        # Step 4: Apply and validate each mutation.
        #
        # We advance ``running_baseline`` after every accepted mutation so the
        # next mutation's ``improvement`` is measured against the NEW state of
        # the skill library, not the (now stale) cycle-start baseline.  Without
        # this, a single earlier boost would let subsequent no-op mutations
        # clear ``min_improvement`` for free.
        accepted = 0
        rejected = 0
        running_baseline = pre_mean
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
                    improvement = adjusted_score - running_baseline

                    if improvement >= self.min_improvement:
                        mutation.accepted = True
                        accepted += 1
                        running_baseline = adjusted_score
                        post_mean = adjusted_score
                        log.info(
                            "Mutation accepted: +%.4f (%.4f -> %.4f, penalty=%.4f)",
                            improvement, running_baseline - improvement,
                            adjusted_score, penalty,
                        )
                        self.skill_manager = SkillManager(evolved_skills_dir=str(self.evolved_skills_dir))
                    else:
                        self._rollback_mutation(mutation)
                        rejected += 1
                        log.info(
                            "Mutation rejected: improvement %.4f < threshold %.4f "
                            "(baseline=%.4f, adjusted=%.4f)",
                            improvement, self.min_improvement,
                            running_baseline, adjusted_score,
                        )
                else:
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
            failure_clusters=failure_clusters,
            mutations=mutations,
            duration_s=time.time() - t0,
            complexity_penalty_applied=total_penalty,
            success_clusters=success_clusters,
            reinforce_mutations=reinforce_count,
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
                "score": f.get("score", f.get("best_score", 0)),
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
                failures[i].get("score", failures[i].get("best_score", 0))
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
    # Private: success clustering
    # ------------------------------------------------------------------

    def _cluster_successes(self, successes: list[dict]) -> list[SuccessCluster]:
        """Use the LLM to cluster high-scoring results by strategy pattern."""
        results_for_llm = []
        for i, s in enumerate(successes[:30]):
            results_for_llm.append({
                "index": i,
                "prompt": s.get("prompt", ""),
                "score": s.get("score", s.get("best_score", 0)),
                "passed_requirements": s.get("passed", [])[:5],
                "tools_used": s.get("tools_used", []),
                "rationale_snippet": str(s.get("rationale", ""))[:200],
            })

        skill_names = ", ".join(self.store.list_skills())

        prompt = _CLUSTER_SUCCESSES_PROMPT.format(
            results_json=json.dumps(results_for_llm, indent=2),
            skill_names=skill_names or "(none)",
        )

        raw_clusters = self._llm_json_call(
            prompt, max_tokens=2048, expect_array=True
        )
        if raw_clusters is None:
            return []

        clusters: list[SuccessCluster] = []
        for rc in raw_clusters:
            indices = rc.get("prompt_indices", [])
            affected = [
                successes[i].get("prompt", "")
                for i in indices
                if i < len(successes)
            ]
            scores = [
                successes[i].get("score", successes[i].get("best_score", 0))
                for i in indices
                if i < len(successes)
            ]
            strategies = [
                str(successes[i].get("rationale", successes[i].get("passed", [])))[:100]
                for i in indices[:3]
                if i < len(successes)
            ]

            if len(affected) < 3:
                continue

            clusters.append(SuccessCluster(
                name=rc.get("name", "unknown_pattern"),
                description=rc.get("description", ""),
                success_count=len(indices),
                affected_prompts=affected,
                mean_score=sum(scores) / len(scores) if scores else 0.0,
                example_strategies=strategies,
                key_tools=rc.get("key_tools", []),
                existing_skill=rc.get("existing_skill"),
            ))

        clusters.sort(key=lambda c: c.success_count, reverse=True)
        return clusters

    def _build_split_manifest(self) -> tuple[str, str]:
        """Return (builtin_manifest, evolved_manifest) for the mutation prompt."""
        evolved_names = self.skill_manager.evolved_skill_names
        builtin_lines, evolved_lines = [], []
        for s in self.skill_manager.get_manifest():
            line = f"- {s['name']}: {s['description']}"
            if s["name"] in evolved_names:
                evolved_lines.append(line)
            else:
                builtin_lines.append(line)
        return (
            "\n".join(builtin_lines) or "(none)",
            "\n".join(evolved_lines) or "(none)",
        )

    def _propose_reinforce_mutation(
        self, cluster: SuccessCluster
    ) -> MutationProposal | None:
        """Ask the LLM to propose a 'reinforce' mutation from a success cluster.

        Only evolved skills may be modified (built-in skills are authoritative).
        If the cluster points at a BUILT-IN skill via ``existing_skill``, we
        coerce ``mutation_type`` to ``create`` so the new evolved skill
        complements (rather than tries to update) the built-in one.  If it
        points at an EVOLVED skill, we coerce to ``update`` and lock the
        target so the LLM can't accidentally spawn a duplicate.
        """
        builtin_manifest, evolved_manifest = self._build_split_manifest()

        existing = cluster.existing_skill
        existing_is_evolved = bool(existing) and existing in self.skill_manager.evolved_skill_names
        existing_is_builtin = bool(existing) and not existing_is_evolved

        cluster_json = json.dumps({
            "name": cluster.name,
            "description": cluster.description,
            "type": "success_pattern",
            "success_count": cluster.success_count,
            "mean_score": cluster.mean_score,
            "key_tools": cluster.key_tools,
            "example_strategies": cluster.example_strategies,
            "existing_skill": existing,
            "existing_skill_kind": (
                "evolved" if existing_is_evolved
                else "builtin" if existing_is_builtin
                else None
            ),
            "affected_prompts": cluster.affected_prompts[:5],
        }, indent=2)

        guidance = (
            "\n\nThis cluster represents SUCCESSFUL strategies, not failures. "
            "Prefer mutation_type 'reinforce' to codify what worked."
        )
        if existing_is_evolved:
            guidance += (
                f"\nAn EVOLVED skill named '{existing}' already partially covers this — "
                "use mutation_type='update' with target_skills=['{existing}'] "
                "to strengthen it, do not create a duplicate."
            ).format(existing=existing)
        elif existing_is_builtin:
            guidance += (
                f"\nThe BUILT-IN skill '{existing}' already covers this domain. "
                "Built-in skills are authoritative and must not be modified. "
                "Use mutation_type='create' to add a NARROW evolved skill whose "
                f"body starts with '## Complements\\nRead `{existing}` first.'"
            )

        prompt = _PROPOSE_MUTATION_PROMPT.format(
            cluster_json=cluster_json,
            builtin_skills_manifest=builtin_manifest,
            evolved_skills_manifest=evolved_manifest,
        ) + guidance

        data = self._llm_json_call(prompt, max_tokens=4000, expect_array=False)
        if data is None:
            return None

        mutation_type = data.get("mutation_type", "reinforce")
        target_skills = data.get("target_skills", []) or []

        # Enforce the constraints described in the docstring even if the LLM
        # disregarded the guidance above.
        if existing_is_evolved:
            mutation_type = "update"
            target_skills = [existing]
        elif existing_is_builtin and mutation_type in ("update", "merge", "delete"):
            # Never touch the built-in skill — fall back to a complementary create.
            mutation_type = "create"
            target_skills = []

        return MutationProposal(
            mutation_type=mutation_type,
            target_skills=target_skills,
            rationale=data.get("rationale", ""),
            failure_cluster=cluster.name,
            proposed_changes=data.get("proposed_changes", {}),
        )

    # ------------------------------------------------------------------
    # Private: mutation proposal
    # ------------------------------------------------------------------

    def _get_recently_mutated_skills(self) -> list[str]:
        """Return skill names that have been mutated, ordered by version count (most first)."""
        versions_dir = self.evolved_skills_dir / ".versions"
        if not versions_dir.exists():
            return []
        from collections import Counter
        counts: Counter[str] = Counter()
        for f in versions_dir.iterdir():
            if f.suffix == ".md":
                parts = f.stem.split("__")
                if parts:
                    counts[parts[0]] += 1
        return [name for name, _ in counts.most_common()]

    def _propose_mutation(self, cluster: FailureCluster) -> MutationProposal | None:
        """Ask the LLM to propose a skill mutation for a failure cluster (with retry)."""
        builtin_manifest, evolved_manifest = self._build_split_manifest()

        cluster_json = json.dumps({
            "name": cluster.name,
            "description": cluster.description,
            "failure_count": cluster.failure_count,
            "mean_score": cluster.mean_score,
            "example_feedback": cluster.example_feedback,
            "existing_skill": cluster.existing_skill,
            "affected_prompts": cluster.affected_prompts[:5],
        }, indent=2)

        recently_mutated = self._get_recently_mutated_skills()
        diversity_hint = ""
        if recently_mutated:
            diversity_hint = (
                f"\n\nDiversity guidance: These skills have already been mutated heavily: "
                f"{', '.join(recently_mutated)}. "
                "Prefer creating a NEW skill or updating a DIFFERENT existing skill "
                "unless this cluster is clearly about the same narrow topic. "
                "Broader coverage across different skills is more valuable than "
                "repeated refinement of one skill."
            )

        prompt = _PROPOSE_MUTATION_PROMPT.format(
            cluster_json=cluster_json,
            builtin_skills_manifest=builtin_manifest,
            evolved_skills_manifest=evolved_manifest,
        ) + diversity_hint

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

    def _effective_tags(self, mutation: MutationProposal) -> list[str]:
        """Merge ``auto_tags`` (from the runner) with the LLM's proposed tags.

        The ``SkillStore`` always re-injects ``"agent"``, but we still include
        any partition tags here so the stored SKILL.md has
        ``tags: [agent, model:<short>, bench:<short>, topic:<whatever>]``.
        """
        proposed = mutation.proposed_changes.get("tags") or []
        if not isinstance(proposed, (list, tuple)):
            proposed = []
        return list(self.auto_tags) + [str(t) for t in proposed]

    def _apply_mutation(self, mutation: MutationProposal) -> None:
        """Apply a mutation to the skill store."""
        changes = mutation.proposed_changes
        mt = mutation.mutation_type
        effective_tags = self._effective_tags(mutation)

        if mt in ("create", "reinforce"):
            skill_name = changes.get("name", f"auto-{mutation.failure_cluster}")
            skill_exists = (self.evolved_skills_dir / skill_name / "SKILL.md").exists()
            if skill_exists and mt == "reinforce":
                self.store.update_skill(
                    name=skill_name,
                    description=changes.get("description"),
                    body=changes.get("body"),
                    tags=effective_tags,
                )
            else:
                origin = "self-evolve-success" if mt == "reinforce" else "self-evolve"
                self.store.create_skill(
                    name=skill_name,
                    description=changes.get("description", "Auto-generated skill"),
                    body=changes.get("body", ""),
                    metadata={"origin": origin, "cluster": mutation.failure_cluster},
                    tags=effective_tags,
                )

        elif mt == "update":
            for skill_name in mutation.target_skills:
                self.store.update_skill(
                    name=skill_name,
                    description=changes.get("description"),
                    body=changes.get("body"),
                    tags=effective_tags,
                )

        elif mt == "merge":
            self.store.merge_skills(
                names=mutation.target_skills,
                merged_name=changes.get("name", mutation.target_skills[0]),
                merged_description=changes.get("description", "Merged skill"),
                merged_body=changes.get("body", ""),
                delete_originals=True,
                tags=effective_tags,
            )

        elif mt == "delete":
            for skill_name in mutation.target_skills:
                self.store.delete_skill(skill_name)

        else:
            raise ValueError(f"Unknown mutation type: {mt}")

    def _rollback_mutation(self, mutation: MutationProposal) -> None:
        """Rollback a mutation by restoring from the latest snapshot."""
        mt = mutation.mutation_type

        if mt in ("create", "reinforce"):
            name = mutation.proposed_changes.get(
                "name", f"auto-{mutation.failure_cluster}"
            )
            try:
                self.store.rollback_skill(name)
            except FileNotFoundError:
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

        if mt in ("create", "reinforce"):
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
        scores = [r.get("score", r.get("best_score", 0.0)) for r in results]
        return sum(scores) / len(scores) if scores else 0.0
