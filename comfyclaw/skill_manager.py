"""
SkillManager — discovers and loads SKILL.md files.

Each skill directory contains a SKILL.md with two required sections::

    # Skill: <name>

    ## Description
    <one or two paragraph description + trigger keywords>

    ## Instructions
    <detailed step-by-step guidance for the agent>

The manager surfaces both sections to the agent:
- ``get_manifest()`` — compact list of all skills (description only) for the system prompt
- ``get_instructions(skill_id)`` — full Instructions section injected when a skill is relevant
- ``detect_relevant_skills(prompt)`` — keyword match against descriptions
"""
from __future__ import annotations

import importlib.resources
import re
from pathlib import Path
from typing import NamedTuple


class Skill(NamedTuple):
    skill_id: str          # directory name, e.g. "high_quality"
    name: str              # parsed from "# Skill: <name>"
    description: str       # ## Description section body
    instructions: str      # ## Instructions section body
    keywords: list[str]    # lowercased trigger words extracted from description


_SECTION_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def _parse_skill_md(text: str, skill_id: str) -> Skill:
    # Title
    title_m = re.search(r"^#\s+Skill:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    name = title_m.group(1).strip() if title_m else skill_id

    # Split into sections
    sections: dict[str, str] = {}
    parts = _SECTION_RE.split(text)
    # parts = [pre-first-section, header1, body1, header2, body2, ...]
    for i in range(1, len(parts) - 1, 2):
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections[header.lower()] = body

    description = sections.get("description", "")
    instructions = sections.get("instructions", "")

    # Extract trigger keywords: quoted phrases and standalone CamelCase words
    raw = description.lower()
    # Simple heuristic: words from first sentence plus any quoted tokens
    words = re.findall(r'["\u201c\u201d]([^"\u201c\u201d]+)["\u201c\u201d]|(\b\w{4,}\b)', raw)
    keywords = list({
        w.strip().lower()
        for pair in words
        for w in pair
        if w.strip()
    })

    return Skill(
        skill_id=skill_id,
        name=name,
        description=description,
        instructions=instructions,
        keywords=keywords,
    )


class SkillManager:
    """
    Load and query agent skills from a directory of ``<skill_id>/SKILL.md`` files.

    Falls back to the bundled ``comfyclaw/skills/`` package data when
    *skills_dir* is ``None``.

    Parameters
    ----------
    skills_dir : Path to a directory containing skill subdirectories.
                 Pass ``None`` to use the built-in skills.
    """

    def __init__(self, skills_dir: str | Path | None = None) -> None:
        self._skills: dict[str, Skill] = {}
        if skills_dir is not None:
            self._load_from_dir(Path(skills_dir))
        else:
            self._load_builtin()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_from_dir(self, directory: Path) -> None:
        if not directory.is_dir():
            return
        for skill_dir in sorted(directory.iterdir()):
            md_path = skill_dir / "SKILL.md"
            if skill_dir.is_dir() and md_path.exists():
                text = md_path.read_text(encoding="utf-8")
                skill = _parse_skill_md(text, skill_dir.name)
                self._skills[skill.skill_id] = skill

    def _load_builtin(self) -> None:
        """Load skills bundled inside the package (comfyclaw/skills/)."""
        try:
            # Python 3.9+ importlib.resources.files
            pkg_skills = importlib.resources.files("comfyclaw").joinpath("skills")
            base = Path(str(pkg_skills))
        except Exception:
            base = Path(__file__).parent / "skills"

        if base.is_dir():
            self._load_from_dir(base)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_manifest(self) -> str:
        """
        Return a compact listing of all skills suitable for a system prompt.

        Format::

            [high_quality] High Quality
              Boost detail and sharpness…

            [photorealistic] Photorealistic
              Use DSLR-style prompt…
        """
        if not self._skills:
            return "(no skills loaded)"
        lines: list[str] = []
        for skill in self._skills.values():
            first_line = (skill.description.splitlines() or [""])[0][:120]
            lines.append(f"[{skill.skill_id}] {skill.name}")
            lines.append(f"  {first_line}")
        return "\n".join(lines)

    def get_instructions(self, skill_id: str) -> str:
        """
        Return the full Instructions section of a skill.

        Returns an empty string if the skill is not found or has no Instructions.
        """
        skill = self._skills.get(skill_id)
        return skill.instructions if skill else ""

    def detect_relevant_skills(self, prompt: str) -> list[str]:
        """
        Return skill IDs whose keywords appear in *prompt* (case-insensitive).

        Ordered by number of keyword matches (most relevant first).
        """
        prompt_lower = prompt.lower()
        scored: list[tuple[int, str]] = []
        for sid, skill in self._skills.items():
            hits = sum(1 for kw in skill.keywords if kw in prompt_lower)
            if hits:
                scored.append((hits, sid))
        scored.sort(reverse=True)
        return [sid for _, sid in scored]

    def get_relevant_instructions(self, prompt: str) -> str:
        """
        Return the concatenated Instructions sections for skills relevant to *prompt*.

        Returns empty string if none match.
        """
        relevant = self.detect_relevant_skills(prompt)
        if not relevant:
            return ""
        parts: list[str] = []
        for sid in relevant:
            instr = self.get_instructions(sid)
            if instr:
                parts.append(f"### Skill: {self._skills[sid].name}\n\n{instr}")
        return "\n\n---\n\n".join(parts)

    @property
    def skill_ids(self) -> list[str]:
        return list(self._skills.keys())

    def __len__(self) -> int:
        return len(self._skills)

    def __repr__(self) -> str:
        return f"<SkillManager skills={self.skill_ids}>"
