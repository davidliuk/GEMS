"""
SkillManager — Agent Skills loader conforming to the Anthropic Agent Skills spec.

Spec summary (https://agentskills.dev/specification):
  • Each skill lives in a directory whose name matches the skill's ``name`` field.
  • ``SKILL.md`` starts with YAML frontmatter (required: ``name``, ``description``).
  • Progressive disclosure:
      - Stage 1: only ``name`` + ``description`` are shown to the agent at startup
        via an ``<available_skills>`` XML block in the system prompt.
      - Stage 2: the agent calls ``read_skill`` to load the full body on demand.

Usage::

    sm = SkillManager()                       # built-in skills
    sm = SkillManager("/path/to/my_skills")   # custom skills dir

    xml = sm.build_available_skills_xml()     # → XML for system prompt
    body = sm.get_body("lora-enhancement")    # → instructions string
    relevant = sm.detect_relevant_skills("photorealistic fox") # → [name, …]
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import NamedTuple

import yaml

# ---------------------------------------------------------------------------
# Built-in + evolved skills directory layout
# ---------------------------------------------------------------------------
#
# Evolved skills are written per (model, benchmark) pair by the benchmark
# runner, following the convention::
#
#     comfyclaw/evolved_skills/<model_short>_<bench_short>/[<agent_slug>/]<skill>/SKILL.md
#
# e.g. ``evolved_skills/longcat_geneval2/`` contains skills LongCat learned on
# GenEval2 and must *not* be loaded when the agent is pinned to Qwen or
# running on DPG-Bench.  Use :func:`evolved_dir_for` (below) as the single
# source of truth for this naming convention.

_BUILTIN_SKILLS_DIR = Path(__file__).resolve().parent / "skills"
_EVOLVED_SKILLS_ROOT = Path(__file__).resolve().parent / "evolved_skills"


def evolved_dir_for(
    model: str,
    benchmark: str,
    agent_name: str | None = None,
    root: str | Path | None = None,
) -> Path:
    """Return the evolved-skills directory for a given (model, benchmark) pair.

    Parameters
    ----------
    model :
        Model short name (e.g. ``"longcat"``, ``"qwen"``, ``"z-image-turbo"``),
        as declared in ``experiments/models/<model>.yaml`` under ``short_name``.
    benchmark :
        Benchmark short name (e.g. ``"geneval2"``, ``"dpg-bench"``).
    agent_name :
        Optional LLM/agent slug used to sub-partition evolved skills between
        agents that may have different strengths (e.g. ``"gpt-5-4"``).  When
        ``None`` (the default), returns the shared bench directory.
    root :
        Base directory containing the per-bench sub-folders.  Defaults to the
        packaged ``comfyclaw/evolved_skills/`` location.

    The returned path is not required to exist — callers that load skills can
    use :class:`SkillManager` which silently treats missing dirs as empty.
    """
    base = Path(root).resolve() if root else _EVOLVED_SKILLS_ROOT
    leaf = base / f"{model}_{benchmark}"
    if agent_name:
        leaf = leaf / agent_name
    return leaf


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class SkillProperties(NamedTuple):
    """
    Parsed SKILL.md frontmatter.

    Attributes
    ----------
    name          : kebab-case skill name (matches directory name).
    description   : What the skill does and when to use it.
    location      : Absolute path to SKILL.md.
    license       : License string (optional).
    compatibility : Environment requirements (optional).
    allowed_tools : Space-delimited pre-approved tool list (optional, experimental).
    metadata      : Arbitrary key→value map (optional).
    tags          : Classification tags for filtering (e.g. ``["agent"]``,
                    ``["agent", "model:longcat"]``, ``["meta"]``).
    """

    name: str
    description: str
    location: Path
    license: str | None = None
    compatibility: str | None = None
    allowed_tools: str | None = None
    metadata: dict[str, str] | None = None
    tags: list[str] | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_skill_md(skill_dir: Path) -> tuple[SkillProperties, str]:
    """
    Parse a ``SKILL.md`` file and return ``(SkillProperties, body_text)``.

    The body text is the Markdown content after the closing ``---``.

    Raises
    ------
    ValueError
        If the file is missing, has no frontmatter, or is missing required fields.
    """
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        skill_md = skill_dir / "skill.md"
    if not skill_md.exists():
        raise ValueError(f"SKILL.md not found in {skill_dir}")

    content = skill_md.read_text(encoding="utf-8")

    if not content.startswith("---"):
        raise ValueError(
            f"{skill_md}: SKILL.md must start with YAML frontmatter (---). "
            "Add frontmatter with at minimum 'name' and 'description'."
        )

    parts = content.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"{skill_md}: frontmatter not closed with ---")

    try:
        fm = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{skill_md}: invalid YAML frontmatter: {exc}") from exc

    if not isinstance(fm, dict):
        raise ValueError(f"{skill_md}: frontmatter must be a YAML mapping")

    for field in ("name", "description"):
        if field not in fm or not str(fm[field]).strip():
            raise ValueError(f"{skill_md}: missing required frontmatter field: {field!r}")

    # Validate name matches directory name
    declared_name = str(fm["name"]).strip()
    if declared_name != skill_dir.name:
        raise ValueError(
            f"{skill_md}: frontmatter 'name' ({declared_name!r}) does not match "
            f"directory name ({skill_dir.name!r})"
        )

    meta = fm.get("metadata")
    if isinstance(meta, dict):
        meta = {str(k): str(v) for k, v in meta.items()}
    else:
        meta = None

    raw_tags = fm.get("tags")
    tags = list(raw_tags) if isinstance(raw_tags, list) else None

    props = SkillProperties(
        name=declared_name,
        description=str(fm["description"]).strip(),
        location=skill_md.resolve(),
        license=fm.get("license"),
        compatibility=fm.get("compatibility"),
        allowed_tools=fm.get("allowed-tools"),
        metadata=meta,
        tags=tags,
    )
    body = parts[2].strip()
    return props, body


def _iter_skill_dirs(root: Path, max_depth: int = 4) -> list[Path]:
    """Recursively find every directory under *root* that contains a SKILL.md.

    A directory is treated as a *skill* directory as soon as it contains
    ``SKILL.md`` (or ``skill.md``); recursion stops there.  Intermediate
    directories that don't hold a SKILL.md — e.g. ``evolved_skills/<bench>/``
    or ``evolved_skills/<bench>/<agent>/`` — are traversed transparently, so
    layouts like::

        evolved_skills/longcat_geneval2/api-retry/SKILL.md
        evolved_skills/longcat_geneval2/gpt-5.4-smoke/my-skill/SKILL.md

    are both discovered.  Hidden directories (``.versions/`` snapshots, etc.)
    are skipped.  ``max_depth`` guards against runaway recursion / symlink
    cycles.
    """
    results: list[Path] = []

    def _walk(current: Path, depth: int) -> None:
        if depth > max_depth or not current.is_dir():
            return
        try:
            entries = sorted(current.iterdir())
        except OSError:
            return
        for entry in entries:
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if (entry / "SKILL.md").exists() or (entry / "skill.md").exists():
                results.append(entry)
            else:
                _walk(entry, depth + 1)

    _walk(root, 0)
    return results


# ---------------------------------------------------------------------------
# SkillManager
# ---------------------------------------------------------------------------


class SkillManager:
    """
    Loads Agent Skills from a directory of skill sub-directories.

    Progressive disclosure is implemented as follows:

    * At startup, call :meth:`build_available_skills_xml` to generate an
      ``<available_skills>`` XML block for the agent's system prompt.  Only
      ``name`` and ``description`` are exposed here — the full instructions are
      **not** loaded.
    * When the agent wants to use a skill it calls the ``read_skill`` tool
      (provided by :class:`~comfyclaw.agent.ClawAgent`).  That tool calls
      :meth:`get_body` to load the full Markdown body on demand.

    Parameters
    ----------
    skills_dir :
        Path to the primary (pre-defined) skill sub-directories.  Pass
        ``None`` to use the built-in skills bundled with the package.
    evolved_skills_dir :
        Path to evolved/learned skills that are generated during benchmarks
        and self-evolution cycles.  When provided, this path is loaded
        verbatim.  Set to ``""`` to explicitly disable.  Default ``None``
        loads no evolved skills **unless** ``model`` + ``benchmark`` are
        supplied (see below).
    model, benchmark :
        Convenience: when both are supplied and ``evolved_skills_dir`` is
        not, the evolved directory is resolved via :func:`evolved_dir_for`
        to ``comfyclaw/evolved_skills/<model>_<benchmark>/``.  This is how
        evolved skills stay partitioned by the (model, bench) pair that
        produced them — e.g. ``SkillManager(model="longcat", benchmark="geneval2")``
        loads only the LongCat-on-GenEval2 skills, not LongCat-on-DPG-Bench
        or Qwen-on-GenEval2.
    agent_name :
        Optional LLM/agent slug; if supplied together with ``model`` +
        ``benchmark``, picks the per-agent sub-folder.
    """

    def __init__(
        self,
        skills_dir: str | Path | None = None,
        evolved_skills_dir: str | Path | None = None,
        *,
        model: str | None = None,
        benchmark: str | None = None,
        agent_name: str | None = None,
    ) -> None:
        root = Path(skills_dir) if skills_dir else _BUILTIN_SKILLS_DIR
        self._root = root.resolve()

        # Resolve evolved_skills_dir with the following precedence:
        #   explicit ""                    → disabled
        #   explicit path                  → use verbatim
        #   model + benchmark supplied     → derive via evolved_dir_for()
        #   otherwise                      → disabled (no silent pan-bench load)
        if evolved_skills_dir == "":
            self._evolved_root: Path | None = None
        elif evolved_skills_dir is not None:
            self._evolved_root = Path(evolved_skills_dir).resolve()
        elif model and benchmark:
            self._evolved_root = evolved_dir_for(model, benchmark, agent_name).resolve()
        else:
            self._evolved_root = None

        self._cache: dict[str, tuple[SkillProperties, str]] = {}
        self._props: dict[str, SkillProperties] = {}
        self._evolved_names: set[str] = set()
        self._load_all()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        """Load all skills from the primary and evolved directories.

        Evolved skills are loaded second; if an evolved skill has the same
        name as a pre-defined one, the evolved version takes precedence
        (and a warning is emitted so silent overrides are visible).

        Both roots are walked recursively via :func:`_iter_skill_dirs`, so
        nested layouts such as ``evolved_skills/<bench>/<agent>/<skill>/``
        are supported transparently.
        """
        import warnings

        dirs: list[tuple[Path, bool]] = [(self._root, False)]
        if self._evolved_root and self._evolved_root.is_dir():
            dirs.append((self._evolved_root, True))

        for root, is_evolved in dirs:
            if not root.is_dir():
                continue
            for skill_dir in _iter_skill_dirs(root):
                try:
                    props, body = _parse_skill_md(skill_dir)
                    if is_evolved:
                        existing_tags = list(props.tags or [])
                        if "agent" not in existing_tags:
                            existing_tags.append("agent")
                        props = props._replace(tags=existing_tags)
                    if is_evolved and props.name in self._props and props.name not in self._evolved_names:
                        warnings.warn(
                            f"[SkillManager] evolved skill {props.name!r} "
                            f"overrides built-in at {skill_dir}",
                            stacklevel=2,
                        )
                    self._cache[props.name] = (props, body)
                    self._props[props.name] = props
                    if is_evolved:
                        self._evolved_names.add(props.name)
                except ValueError as exc:
                    warnings.warn(
                        f"[SkillManager] Skipping skill at {skill_dir}: {exc}",
                        stacklevel=2,
                    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def skill_names(self) -> list[str]:
        """Sorted list of loaded skill names."""
        return sorted(self._props)

    @property
    def evolved_skill_names(self) -> set[str]:
        """Names of skills loaded from the evolved skills directory."""
        return set(self._evolved_names)

    def get_properties(self, name: str) -> SkillProperties:
        """Return :class:`SkillProperties` for *name* (raises ``KeyError`` if missing)."""
        return self._props[name]

    def get_body(self, name: str) -> str:
        """
        Return the full Markdown body (instructions) for *name*.

        This is the **stage-2** progressive-disclosure load — only called when
        the agent explicitly requests a skill's instructions via ``read_skill``.

        Raises ``KeyError`` if the skill does not exist.
        """
        return self._cache[name][1]

    def build_available_skills_xml(self, include_tags: set[str] | None = None) -> str:
        """
        Generate the ``<available_skills>`` XML block for agent system prompts.

        Follows the Anthropic-recommended format — only ``name``, ``description``,
        and ``location`` are exposed (stage-1 progressive disclosure).

        Parameters
        ----------
        include_tags :
            When provided, skills are filtered by their ``tags``.  For skills
            with a ``model:*`` tag, at least one of those model tags must appear
            in *include_tags* (so model-specific skills are only shown when
            their model is active).  For all other skills, any tag overlap with
            *include_tags* suffices.  Skills without tags are excluded.
            Pass ``None`` (default) to include all skills regardless of tags.

        Example output::

            <available_skills>
            <skill>
            <name>lora-enhancement</name>
            <description>Insert LoRA adapter nodes …</description>
            <location>/path/to/lora-enhancement/SKILL.md</location>
            </skill>
            …
            </available_skills>
        """
        if not self._props:
            return "<available_skills>\n</available_skills>"

        lines = ["<available_skills>"]
        for name in self.skill_names:
            props = self._props[name]
            if include_tags is not None:
                skill_tags = set(props.tags or [])
                model_tags = {t for t in skill_tags if t.startswith("model:")}
                if model_tags:
                    if not model_tags & include_tags:
                        continue
                elif not skill_tags & include_tags:
                    continue
            lines += [
                "<skill>",
                f"<name>{html.escape(props.name)}</name>",
                f"<description>{html.escape(props.description)}</description>",
                f"<location>{html.escape(str(props.location))}</location>",
                "</skill>",
            ]
        lines.append("</available_skills>")
        return "\n".join(lines)

    def detect_relevant_skills(self, prompt: str) -> list[str]:
        """
        Return skill names likely relevant to *prompt*.

        For built-in skills, matches by keyword overlap between description
        and prompt.  For evolved skills, additionally matches against the
        skill name (kebab-case words) and metadata cluster name, since
        evolved skill descriptions tend to use abstract technique language
        that rarely overlaps with concrete prompt words.
        """
        prompt_lower = prompt.lower()
        prompt_words = set(re.findall(r"[a-z]{4,}", prompt_lower))
        matched: set[str] = set()

        for name, props in self._props.items():
            desc_keywords = set(re.findall(r"[a-z]{4,}", props.description.lower()))
            if desc_keywords & prompt_words:
                matched.add(name)
                continue

            if name in self._evolved_names:
                name_words = set(name.replace("-", " ").split())
                cluster = (props.metadata or {}).get("cluster", "")
                cluster_words = set(re.findall(r"[a-z]{4,}", cluster.lower()))
                if (name_words & prompt_words) or (cluster_words & prompt_words):
                    matched.add(name)

        return sorted(matched)

    def get_manifest(self) -> list[dict[str, str]]:
        """
        Return a list of ``{name, description, location}`` dicts for all skills.

        Useful for building custom prompt formats outside the XML standard.
        """
        return [
            {
                "name": p.name,
                "description": p.description,
                "location": str(p.location),
            }
            for p in self._props.values()
        ]
