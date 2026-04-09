"""Unit tests for SkillManager."""
from __future__ import annotations

from pathlib import Path

import pytest

from comfyclaw.skill_manager import SkillManager, _parse_skill_md

_SKILL_A = """\
# Skill: High Quality

## Description
Boost image sharpness and detail. Trigger on keywords: quality, sharp, detailed.

## Instructions

### Steps
1. Increase sampling steps to 30+.
2. Add detail-focused negative prompt terms.
3. Set CFG to 7.0–8.0 for crisper output.
"""

_SKILL_B = """\
# Skill: Photorealistic

## Description
Produce DSLR-style photorealistic images. Trigger on: photorealistic, photo, DSLR, realistic.

## Instructions

### Steps
1. Prefix prompt with "photograph, DSLR,".
2. Sampler: dpmpp_2m with karras.
3. Steps ≥ 25.
"""


@pytest.fixture()
def skills_dir(tmp_path: Path) -> Path:
    """Create two fake skill directories."""
    for skill_id, content in [("high_quality", _SKILL_A), ("photorealistic", _SKILL_B)]:
        d = tmp_path / skill_id
        d.mkdir()
        (d / "SKILL.md").write_text(content, encoding="utf-8")
    return tmp_path


class TestLoad:
    def test_loads_from_dir(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        assert len(sm) == 2
        assert "high_quality" in sm.skill_ids
        assert "photorealistic" in sm.skill_ids

    def test_empty_dir_loads_nothing(self, tmp_path: Path) -> None:
        sm = SkillManager(tmp_path)
        assert len(sm) == 0

    def test_nonexistent_dir_loads_nothing(self, tmp_path: Path) -> None:
        sm = SkillManager(tmp_path / "does_not_exist")
        assert len(sm) == 0

    def test_ignores_files_without_skill_md(self, tmp_path: Path) -> None:
        (tmp_path / "readme.txt").write_text("hello")
        sm = SkillManager(tmp_path)
        assert len(sm) == 0


class TestGetManifest:
    def test_manifest_contains_skill_ids(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        manifest = sm.get_manifest()
        assert "[high_quality]" in manifest
        assert "[photorealistic]" in manifest

    def test_manifest_contains_first_description_line(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        manifest = sm.get_manifest()
        assert "sharpness" in manifest.lower() or "detail" in manifest.lower()

    def test_empty_manifest_message(self, tmp_path: Path) -> None:
        sm = SkillManager(tmp_path)
        assert "no skills" in sm.get_manifest().lower()


class TestGetInstructions:
    def test_returns_instructions_section(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        instr = sm.get_instructions("high_quality")
        assert "Increase sampling steps" in instr

    def test_returns_empty_for_missing_skill(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        assert sm.get_instructions("nonexistent") == ""

    def test_instructions_not_in_description(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        desc = sm._skills["high_quality"].description
        instr = sm._skills["high_quality"].instructions
        # Instructions section body must not bleed into description
        assert "Increase sampling" not in desc
        assert "Increase sampling" in instr


class TestDetectRelevantSkills:
    def test_matches_photorealistic_prompt(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        result = sm.detect_relevant_skills("a photorealistic portrait photo")
        assert "photorealistic" in result

    def test_matches_quality_prompt(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        result = sm.detect_relevant_skills("high quality detailed artwork")
        assert "high_quality" in result

    def test_no_match_returns_empty(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        result = sm.detect_relevant_skills("xxxyyy zzz12345")
        assert result == []

    def test_most_relevant_comes_first(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        # This prompt has more hits for photorealistic than high_quality
        result = sm.detect_relevant_skills("photorealistic DSLR photo, realistic portrait")
        if len(result) > 1:
            assert result[0] == "photorealistic"


class TestGetRelevantInstructions:
    def test_returns_combined_instructions(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        text = sm.get_relevant_instructions("high quality photorealistic photo")
        assert "Increase sampling steps" in text  # from high_quality
        assert "Prefix prompt" in text             # from photorealistic

    def test_empty_when_no_match(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        assert sm.get_relevant_instructions("xyzzy123") == ""


class TestParseSkillMd:
    def test_parses_name(self) -> None:
        skill = _parse_skill_md(_SKILL_A, "high_quality")
        assert skill.name == "High Quality"

    def test_parses_description(self) -> None:
        skill = _parse_skill_md(_SKILL_A, "high_quality")
        assert "sharpness" in skill.description.lower() or "detail" in skill.description.lower()

    def test_parses_instructions(self) -> None:
        skill = _parse_skill_md(_SKILL_A, "high_quality")
        assert "Increase sampling steps" in skill.instructions

    def test_extracts_keywords(self) -> None:
        skill = _parse_skill_md(_SKILL_A, "high_quality")
        assert any("quality" in kw or "sharp" in kw or "detail" in kw for kw in skill.keywords)

    def test_skill_id_used_as_name_fallback(self) -> None:
        skill = _parse_skill_md("## Description\nHello.", "my_skill")
        assert skill.name == "my_skill"
