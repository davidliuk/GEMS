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
# Built-in skills directory
# ---------------------------------------------------------------------------

_BUILTIN_SKILLS_DIR = Path(__file__).resolve().parent / "skills"


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
    """

    name: str
    description: str
    location: Path
    license: str | None = None
    compatibility: str | None = None
    allowed_tools: str | None = None
    metadata: dict[str, str] | None = None


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

    props = SkillProperties(
        name=declared_name,
        description=str(fm["description"]).strip(),
        location=skill_md.resolve(),
        license=fm.get("license"),
        compatibility=fm.get("compatibility"),
        allowed_tools=fm.get("allowed-tools"),
        metadata=meta,
    )
    body = parts[2].strip()
    return props, body


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
        Path to a directory of skill sub-directories.  Pass ``None`` to use
        the built-in skills bundled with the package.
    """

    def __init__(self, skills_dir: str | Path | None = None) -> None:
        root = Path(skills_dir) if skills_dir else _BUILTIN_SKILLS_DIR
        self._root = root.resolve()
        self._cache: dict[str, tuple[SkillProperties, str]] = {}
        self._props: dict[str, SkillProperties] = {}
        self._load_all()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        """Load all skills from the root directory (frontmatter + body)."""
        if not self._root.is_dir():
            return
        for skill_dir in sorted(self._root.iterdir()):
            if not skill_dir.is_dir():
                continue
            try:
                props, body = _parse_skill_md(skill_dir)
                self._cache[props.name] = (props, body)
                self._props[props.name] = props
            except ValueError as exc:
                # Don't crash — warn and skip malformed skills
                import warnings

                warnings.warn(
                    f"[SkillManager] Skipping skill {skill_dir.name}: {exc}", stacklevel=2
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def skill_names(self) -> list[str]:
        """Sorted list of loaded skill names."""
        return sorted(self._props)

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

    def build_available_skills_xml(self) -> str:
        """
        Generate the ``<available_skills>`` XML block for agent system prompts.

        Follows the Anthropic-recommended format — only ``name``, ``description``,
        and ``location`` are exposed (stage-1 progressive disclosure).

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
        Return skill names whose description keywords appear in *prompt*.

        This is a lightweight heuristic for pre-seeding suggestions when the
        agent has not yet decided which skills to activate.  Returns names in
        alphabetical order.
        """
        prompt_lower = prompt.lower()
        matched: list[str] = []
        for name, props in self._props.items():
            # Extract significant words (≥4 chars) from the description
            keywords = set(re.findall(r"[a-z]{4,}", props.description.lower()))
            # Check for overlap with the prompt
            prompt_words = set(re.findall(r"[a-z]{4,}", prompt_lower))
            if keywords & prompt_words:
                matched.append(name)
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
