"""Embed the source commit in climate_indices builds.

Hatchling runs this as a build hook (``[tool.hatch.build.hooks.custom]`` in
``pyproject.toml``). From a Git checkout it writes ``_build_info.json`` -- the
``HEAD`` commit and whether tracked files were modified -- into the sdist and the
wheel, never into ``src/``, so an editable checkout cannot read a stale copy.
From an unpacked sdist, which has no ``.git``, it ships the file the sdist
already carries, so an sdist-to-wheel build keeps the commit.

maturin does not run Hatchling hooks, so the release workflow runs this file as
a script first: ``python hatch_build.py`` writes ``src/climate_indices/_build_info.json``
into its ephemeral checkout for the platform wheels. See ``docs/provenance.md``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess  # nosec B404 # runs only the resolved git binary, with fixed arguments
import sys
import tempfile
from pathlib import Path
from typing import Any

BUILD_INFO = "src/climate_indices/_build_info.json"

try:
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface
except ImportError:  # run as a script, where hatchling need not be installed
    BuildHookInterface = object


def build_info(root: Path) -> dict[str, Any] | None:
    """The ``HEAD`` commit and tracked-file dirty state of ``root``, if it is its own Git checkout."""
    git = shutil.which("git")
    if git is None or not (root / ".git").exists():
        return None
    # GIT_DIR, GIT_WORK_TREE and friends would point git at another repository
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}

    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(  # nosec B603 # fixed arguments, no shell
                [git, "--no-optional-locks", "-C", str(root), *args],
                capture_output=True,
                text=True,
                env=env,
                check=False,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    toplevel = run("rev-parse", "--show-toplevel")
    if toplevel is None or Path(toplevel).resolve() != root.resolve():
        return None
    commit = run("rev-parse", "--verify", "HEAD^{commit}")
    status = run("status", "--porcelain", "--untracked-files=no")
    if commit is None or status is None:
        return None
    return {"git_commit": commit, "git_dirty": bool(status)}


class ProvenanceBuildHook(BuildHookInterface):  # type: ignore[misc]
    """Force-include ``_build_info.json`` in the sdist and the wheel."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if version == "editable":
            return  # an editable install reports its checkout's live Git state
        root = Path(self.root)
        target = BUILD_INFO if self.target_name == "sdist" else BUILD_INFO.removeprefix("src/")
        info = build_info(root)
        if info is not None:
            self._directory = tempfile.mkdtemp(prefix="climate_indices-build-info-")
            generated = Path(self._directory, "_build_info.json")
            generated.write_text(json.dumps(info, sort_keys=True) + "\n", encoding="utf-8")
            build_data["force_include"][str(generated)] = target
        elif (root / BUILD_INFO).is_file():
            build_data["force_include"][str(root / BUILD_INFO)] = target

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        directory = getattr(self, "_directory", None)
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    embedded = build_info(project_root)
    if embedded is None:
        sys.exit(f"{project_root} is not its own Git checkout; no source commit to embed")
    (project_root / BUILD_INFO).write_text(json.dumps(embedded, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {BUILD_INFO}: {embedded}")
