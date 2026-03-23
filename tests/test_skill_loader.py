from __future__ import annotations

from pathlib import Path

from core.skill_loader import activate_skill, discover_skills, stub_to_manifest


def test_discover_skills_reads_standard_stub_metadata(tmp_path: Path):
    root = tmp_path / "skills"
    skill = root / "doc-skill"
    (skill / "scripts").mkdir(parents=True)
    (skill / "references").mkdir(parents=True)
    (skill / "assets").mkdir(parents=True)

    (skill / "SKILL.md").write_text(
        """
---
name: doc-skill
description: Create polished .docx reports.
metadata:
  tags: [docx, reports]
  custom: keep-me
---
Use python-docx when available.
""".strip(),
        encoding="utf-8",
    )
    (skill / "scripts" / "build_doc.py").write_text("print('ok')\n", encoding="utf-8")
    (skill / "references" / "guide.md").write_text("# guide\n", encoding="utf-8")
    (skill / "assets" / "sample.txt").write_text("asset\n", encoding="utf-8")

    stubs = discover_skills([root])

    assert [stub.id for stub in stubs] == ["doc-skill"]
    stub = stubs[0]
    assert stub.name == "doc-skill"
    assert stub.description == "Create polished .docx reports."
    assert stub.metadata["custom"] == "keep-me"
    assert sorted(stub.bundled_files) == [
        "assets/sample.txt",
        "references/guide.md",
        "scripts/build_doc.py",
    ]


def test_activate_skill_loads_prompt_and_resource_index(tmp_path: Path):
    root = tmp_path / "skills"
    skill = root / "ops-skill"
    (skill / "scripts").mkdir(parents=True)
    (skill / "references").mkdir(parents=True)

    (skill / "SKILL.md").write_text(
        """
---
name: ops-skill
description: Run an operational workflow.
---
Step 1: load the full instructions only when selected.
""".strip(),
        encoding="utf-8",
    )
    (skill / "scripts" / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (skill / "references" / "notes.md").write_text("notes\n", encoding="utf-8")

    stub = discover_skills([root])[0]
    loaded = activate_skill(stub)

    assert "full instructions" in loaded.instructions_markdown
    assert loaded.scripts == ["scripts/run.sh"]
    assert loaded.resources == ["references/notes.md"]
    assert loaded.capabilities["scripts"] == ["scripts/run.sh"]
    assert loaded.capabilities["resources"] == ["references/notes.md"]


def test_stub_to_manifest_preserves_bundled_metadata(tmp_path: Path):
    root = tmp_path / "skills"
    skill = root / "sample"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        """
---
name: sample
description: Sample skill.
metadata:
  owner: platform
---
Body
""".strip(),
        encoding="utf-8",
    )
    (skill / "helper.txt").write_text("x\n", encoding="utf-8")

    stub = discover_skills([root])[0]
    manifest = stub_to_manifest(stub)

    assert manifest.frontmatter["name"] == "sample"
    assert manifest.metadata["owner"] == "platform"
    assert manifest.bundled_files == ["helper.txt"]
