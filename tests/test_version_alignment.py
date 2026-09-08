"""Guard against version drift across release metadata files.

pyproject.toml and habitus/config.yaml previously fell out of sync (fixed in
4.1.4 — see CHANGELOG.md) and nothing caught it. These tests fail fast if the
package version, add-on manifest version, changelog headers, or the version
declared in CLAUDE.md disagree.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

_VERSION_HEADER_RE = re.compile(r"^## \[(\d+\.\d+\.\d+)\]", re.MULTILINE)
_CLAUDE_MD_VERSION_RE = re.compile(r"^\*\*Version:\*\*\s*(\d+\.\d+\.\d+)", re.MULTILINE)


def _pyproject_version() -> str:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    return data["project"]["version"]


def _addon_manifest_version() -> str:
    data = yaml.safe_load((REPO_ROOT / "habitus" / "config.yaml").read_text())
    return str(data["version"])


def _latest_changelog_version(path: Path) -> str:
    match = _VERSION_HEADER_RE.search(path.read_text())
    assert match, f"No semver '## [X.Y.Z]' entry found in {path}"
    return match.group(1)


def test_pyproject_and_addon_manifest_versions_match() -> None:
    assert (
        _pyproject_version() == _addon_manifest_version()
    ), "pyproject.toml [project].version must match habitus/config.yaml version"


def test_root_changelog_latest_entry_matches_manifest() -> None:
    assert _latest_changelog_version(REPO_ROOT / "CHANGELOG.md") == _addon_manifest_version()


def test_addon_changelog_latest_entry_matches_manifest() -> None:
    assert (
        _latest_changelog_version(REPO_ROOT / "habitus" / "CHANGELOG.md")
        == _addon_manifest_version()
    )


def test_claude_md_declared_version_matches_manifest() -> None:
    text = (REPO_ROOT / "CLAUDE.md").read_text()
    match = _CLAUDE_MD_VERSION_RE.search(text)
    assert match, "No '**Version:** X.Y.Z' line found in CLAUDE.md"
    assert (
        match.group(1) == _addon_manifest_version()
    ), "CLAUDE.md '**Version:**' must match habitus/config.yaml version"
