"""Unit tests for SkillManager (Anthropic Agent Skills format)."""

from __future__ import annotations

from pathlib import Path

import pytest

from comfyclaw.skill_manager import SkillManager, SkillProperties, _parse_skill_md

# ---------------------------------------------------------------------------
# Sample SKILL.md content (valid frontmatter format)
# ---------------------------------------------------------------------------

_SKILL_HQ = """\
---
name: high-quality
description: >-
  Boost perceptual sharpness and detail. Use when the user asks for "high quality",
  "sharp", or "detailed" images.
license: MIT
metadata:
  version: "0.1.0"
---

1. Append `, masterpiece, best quality, highly detailed` to the positive prompt.
2. Set KSampler steps to at least 20.
3. Set CFG to 7.0.
"""

_SKILL_PHOTO = """\
---
name: photorealistic
description: >-
  Produce DSLR-style photorealistic images. Use when the user mentions "photo",
  "photograph", "realistic", "DSLR", or "cinematic".
compatibility: ComfyClaw agent — requires KSampler.
---

1. Append `, RAW photo, DSLR, photorealistic` to the positive prompt.
2. Use dpmpp_2m with karras scheduler.
3. Set steps to 30.
"""

_SKILL_MINIMAL = """\
---
name: creative
description: Add creative artistic flair. Use for creative or artistic prompts.
---

Append `, concept art, vivid colors` to the positive prompt.
"""

_SKILL_NO_FRONTMATTER = """\
# Old Skill Format

## Description
This skill lacks frontmatter.

## Instructions
Nothing useful here.
"""

_SKILL_MISSING_NAME = """\
---
description: A skill without a name field.
---

Some body text.
"""

_SKILL_MISSING_DESCRIPTION = """\
---
name: broken-skill
---

Some body text.
"""

_SKILL_WRONG_DIR_NAME = """\
---
name: wrong-name
description: The name does not match the directory name.
---

Body.
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def skills_dir(tmp_path: Path) -> Path:
    """Create two valid skill directories with proper frontmatter."""
    for skill_id, content in [
        ("high-quality", _SKILL_HQ),
        ("photorealistic", _SKILL_PHOTO),
    ]:
        d = tmp_path / skill_id
        d.mkdir()
        (d / "SKILL.md").write_text(content, encoding="utf-8")
    return tmp_path


@pytest.fixture()
def full_skills_dir(tmp_path: Path) -> Path:
    """Three valid skills."""
    for skill_id, content in [
        ("high-quality", _SKILL_HQ),
        ("photorealistic", _SKILL_PHOTO),
        ("creative", _SKILL_MINIMAL),
    ]:
        d = tmp_path / skill_id
        d.mkdir()
        (d / "SKILL.md").write_text(content, encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# TestLoad
# ---------------------------------------------------------------------------


class TestLoad:
    def test_loads_from_dir(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        assert len(sm.skill_names) == 2
        assert "high-quality" in sm.skill_names
        assert "photorealistic" in sm.skill_names

    def test_empty_dir_loads_nothing(self, tmp_path: Path) -> None:
        sm = SkillManager(tmp_path)
        assert sm.skill_names == []

    def test_nonexistent_dir_loads_nothing(self, tmp_path: Path) -> None:
        sm = SkillManager(tmp_path / "does_not_exist")
        assert sm.skill_names == []

    def test_ignores_plain_files_in_root(self, tmp_path: Path) -> None:
        (tmp_path / "readme.txt").write_text("hello")
        sm = SkillManager(tmp_path)
        assert sm.skill_names == []

    def test_skips_dir_without_skill_md(
        self, tmp_path: Path, recwarn: pytest.WarningsChecker
    ) -> None:
        (tmp_path / "empty-skill").mkdir()
        sm = SkillManager(tmp_path)
        assert sm.skill_names == []

    def test_skips_malformed_skill_with_warning(self, tmp_path: Path) -> None:
        bad = tmp_path / "no-fm"
        bad.mkdir()
        (bad / "SKILL.md").write_text(_SKILL_NO_FRONTMATTER)
        with pytest.warns(UserWarning, match="Skipping skill"):
            sm = SkillManager(tmp_path)
        assert sm.skill_names == []

    def test_skips_missing_name(self, tmp_path: Path) -> None:
        d = tmp_path / "missing-name"
        d.mkdir()
        (d / "SKILL.md").write_text(_SKILL_MISSING_NAME)
        with pytest.warns(UserWarning, match="Skipping skill"):
            sm = SkillManager(tmp_path)
        assert sm.skill_names == []

    def test_skips_missing_description(self, tmp_path: Path) -> None:
        d = tmp_path / "broken-skill"
        d.mkdir()
        (d / "SKILL.md").write_text(_SKILL_MISSING_DESCRIPTION)
        with pytest.warns(UserWarning, match="Skipping skill"):
            sm = SkillManager(tmp_path)
        assert sm.skill_names == []

    def test_skips_name_mismatch(self, tmp_path: Path) -> None:
        d = tmp_path / "actual-dir-name"
        d.mkdir()
        (d / "SKILL.md").write_text(_SKILL_WRONG_DIR_NAME)
        with pytest.warns(UserWarning, match="Skipping skill"):
            sm = SkillManager(tmp_path)
        assert sm.skill_names == []

    def test_accepts_lowercase_skill_md(self, tmp_path: Path) -> None:
        d = tmp_path / "creative"
        d.mkdir()
        (d / "skill.md").write_text(_SKILL_MINIMAL)  # lowercase filename
        sm = SkillManager(tmp_path)
        assert "creative" in sm.skill_names


# ---------------------------------------------------------------------------
# TestGetProperties
# ---------------------------------------------------------------------------


class TestGetProperties:
    def test_returns_skill_properties(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        props = sm.get_properties("high-quality")
        assert isinstance(props, SkillProperties)
        assert props.name == "high-quality"

    def test_description_populated(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        props = sm.get_properties("high-quality")
        assert "high quality" in props.description.lower() or "sharp" in props.description.lower()

    def test_license_parsed(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        assert sm.get_properties("high-quality").license == "MIT"

    def test_compatibility_parsed(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        assert sm.get_properties("photorealistic").compatibility is not None

    def test_optional_fields_none_when_absent(self, full_skills_dir: Path) -> None:
        sm = SkillManager(full_skills_dir)
        props = sm.get_properties("creative")
        assert props.license is None
        assert props.compatibility is None
        assert props.allowed_tools is None

    def test_metadata_parsed(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        meta = sm.get_properties("high-quality").metadata
        assert isinstance(meta, dict)
        assert meta.get("version") == "0.1.0"

    def test_location_is_absolute_path(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        loc = sm.get_properties("high-quality").location
        assert loc.is_absolute()
        assert loc.name == "SKILL.md"

    def test_raises_keyerror_for_unknown_skill(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        with pytest.raises(KeyError):
            sm.get_properties("nonexistent")


# ---------------------------------------------------------------------------
# TestGetBody
# ---------------------------------------------------------------------------


class TestGetBody:
    def test_returns_body_text(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        body = sm.get_body("high-quality")
        assert "Append" in body or "append" in body or "steps" in body.lower()

    def test_body_does_not_contain_frontmatter(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        body = sm.get_body("high-quality")
        assert "---" not in body
        assert "name:" not in body

    def test_raises_keyerror_for_unknown_skill(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        with pytest.raises(KeyError):
            sm.get_body("nonexistent")

    def test_different_skills_have_different_bodies(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        assert sm.get_body("high-quality") != sm.get_body("photorealistic")


# ---------------------------------------------------------------------------
# TestBuildAvailableSkillsXml
# ---------------------------------------------------------------------------


class TestBuildAvailableSkillsXml:
    def test_produces_available_skills_wrapper(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        xml = sm.build_available_skills_xml()
        assert xml.startswith("<available_skills>")
        assert xml.endswith("</available_skills>")

    def test_contains_skill_names(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        xml = sm.build_available_skills_xml()
        assert "<name>high-quality</name>" in xml
        assert "<name>photorealistic</name>" in xml

    def test_contains_description(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        xml = sm.build_available_skills_xml()
        assert "<description>" in xml

    def test_contains_location(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        xml = sm.build_available_skills_xml()
        assert "<location>" in xml
        assert "SKILL.md" in xml

    def test_body_not_exposed_in_xml(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        xml = sm.build_available_skills_xml()
        # Body instructions should NOT appear in the stage-1 XML
        assert "KSampler" not in xml
        assert "dpmpp_2m" not in xml

    def test_empty_manager_produces_empty_xml(self, tmp_path: Path) -> None:
        sm = SkillManager(tmp_path)
        xml = sm.build_available_skills_xml()
        assert xml == "<available_skills>\n</available_skills>"

    def test_special_chars_escaped(self, tmp_path: Path) -> None:
        d = tmp_path / "my-skill"
        d.mkdir()
        content = (
            '---\nname: my-skill\ndescription: A skill with <special> & "chars".\n---\n\nBody.\n'
        )
        (d / "SKILL.md").write_text(content)
        sm = SkillManager(tmp_path)
        xml = sm.build_available_skills_xml()
        assert "&lt;special&gt;" in xml
        assert "&amp;" in xml


# ---------------------------------------------------------------------------
# TestDetectRelevantSkills
# ---------------------------------------------------------------------------


class TestDetectRelevantSkills:
    def test_matches_photorealistic_prompt(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        result = sm.detect_relevant_skills("a photorealistic portrait photo")
        assert "photorealistic" in result

    def test_matches_quality_prompt(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        result = sm.detect_relevant_skills("high quality detailed artwork")
        assert "high-quality" in result

    def test_no_match_returns_empty(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        result = sm.detect_relevant_skills("xxxyyy zzz12345")
        assert result == []

    def test_returns_sorted_list(self, full_skills_dir: Path) -> None:
        sm = SkillManager(full_skills_dir)
        result = sm.detect_relevant_skills("quality artistic creative photo")
        assert result == sorted(result)


# ---------------------------------------------------------------------------
# TestGetManifest
# ---------------------------------------------------------------------------


class TestGetManifest:
    def test_returns_list_of_dicts(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        manifest = sm.get_manifest()
        assert isinstance(manifest, list)
        assert all(isinstance(m, dict) for m in manifest)

    def test_each_entry_has_required_keys(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        for entry in sm.get_manifest():
            assert "name" in entry
            assert "description" in entry
            assert "location" in entry

    def test_names_match_skill_names(self, skills_dir: Path) -> None:
        sm = SkillManager(skills_dir)
        names = {e["name"] for e in sm.get_manifest()}
        assert names == set(sm.skill_names)


# ---------------------------------------------------------------------------
# TestParseSkillMd (internal helper)
# ---------------------------------------------------------------------------


class TestParseSkillMd:
    def test_parses_name(self, tmp_path: Path) -> None:
        d = tmp_path / "high-quality"
        d.mkdir()
        (d / "SKILL.md").write_text(_SKILL_HQ)
        props, _ = _parse_skill_md(d)
        assert props.name == "high-quality"

    def test_parses_description(self, tmp_path: Path) -> None:
        d = tmp_path / "high-quality"
        d.mkdir()
        (d / "SKILL.md").write_text(_SKILL_HQ)
        props, _ = _parse_skill_md(d)
        assert props.description  # non-empty

    def test_body_separated_from_frontmatter(self, tmp_path: Path) -> None:
        d = tmp_path / "high-quality"
        d.mkdir()
        (d / "SKILL.md").write_text(_SKILL_HQ)
        props, body = _parse_skill_md(d)
        assert "---" not in body
        assert "name:" not in body
        assert "masterpiece" in body or "steps" in body.lower() or "append" in body.lower()

    def test_raises_on_missing_skill_md(self, tmp_path: Path) -> None:
        d = tmp_path / "no-skill"
        d.mkdir()
        with pytest.raises(ValueError, match="SKILL.md not found"):
            _parse_skill_md(d)

    def test_raises_on_no_frontmatter(self, tmp_path: Path) -> None:
        d = tmp_path / "no-fm"
        d.mkdir()
        (d / "SKILL.md").write_text(_SKILL_NO_FRONTMATTER)
        with pytest.raises(ValueError, match="frontmatter"):
            _parse_skill_md(d)

    def test_raises_on_missing_required_field(self, tmp_path: Path) -> None:
        d = tmp_path / "missing-name"
        d.mkdir()
        (d / "SKILL.md").write_text(_SKILL_MISSING_NAME)
        with pytest.raises(ValueError, match="name"):
            _parse_skill_md(d)

    def test_raises_on_name_mismatch(self, tmp_path: Path) -> None:
        d = tmp_path / "actual-dir-name"
        d.mkdir()
        (d / "SKILL.md").write_text(_SKILL_WRONG_DIR_NAME)
        with pytest.raises(ValueError, match="does not match"):
            _parse_skill_md(d)
