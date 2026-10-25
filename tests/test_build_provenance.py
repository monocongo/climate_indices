"""Builds embed the source commit (``hatch_build.py``), and an installed wheel reports it (#1470).

Each test builds a throwaway copy of the package in its own temporary Git
repository through Hatchling's PEP 517 hooks, offline and without build
isolation, so the real checkout's history is never touched.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("hatchling")

ROOT = Path(__file__).parent.parent
GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(GIT is None, reason="git is not installed")


def _env(**extra: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.org",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.org",
    )
    return {**env, **extra}


def _git(root: Path, *args: str) -> str:
    assert GIT is not None
    return subprocess.run(
        [GIT, "-C", str(root), *args], capture_output=True, text=True, check=True, env=_env()
    ).stdout.strip()


def _repository(root: Path) -> str:
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "snapshot")
    return _git(root, "rev-parse", "HEAD")


def _project(tmp_path: Path) -> tuple[Path, str]:
    """A committed copy of the buildable project; returns its root and HEAD."""
    root = tmp_path / "project"
    ignore = shutil.ignore_patterns("__pycache__", "*.so", "*.pyd", "_build_info.json")
    shutil.copytree(ROOT / "src" / "climate_indices", root / "src" / "climate_indices", ignore=ignore)
    for name in ("pyproject.toml", "hatch_build.py", "README.md", "LICENSE", ".gitignore"):
        shutil.copy2(ROOT / name, root / name)
    return root, _repository(root)


def _build(root: Path, hook: str, outdir: Path) -> Path:
    """Run one PEP 517 hook of the Hatchling backend in ``root``; returns the artifact."""
    outdir.mkdir(parents=True, exist_ok=True)
    code = f"import sys, hatchling.build as backend; print(backend.{hook}(sys.argv[1]))"
    completed = subprocess.run(
        [sys.executable, "-c", code, str(outdir)], cwd=root, capture_output=True, text=True, check=True, env=_env()
    )
    return outdir / completed.stdout.strip().splitlines()[-1]


def _wheel_build_info(wheel: Path) -> dict[str, Any]:
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.endswith("_build_info.json")]
        assert names == ["climate_indices/_build_info.json"], names
        info: dict[str, Any] = json.loads(archive.read(names[0]))
    return info


def test_the_sdist_and_the_wheel_built_from_it_report_the_commit_without_git(tmp_path: Path) -> None:
    root, head = _project(tmp_path)

    sdist = _build(root, "build_sdist", tmp_path / "dist")
    with tarfile.open(sdist) as archive:
        member = archive.extractfile(f"{sdist.name.removesuffix('.tar.gz')}/src/climate_indices/_build_info.json")
        assert member is not None
        assert json.loads(member.read()) == {"git_commit": head, "git_dirty": False}
        # unpack inside an unrelated repository: its HEAD must not replace the sdist's commit
        unrelated = tmp_path / "unrelated"
        unrelated.mkdir()
        (unrelated / "notes.txt").write_text("another project\n")
        other_head = _repository(unrelated)
        # the "data" filter exists from Python 3.10.12; the archive is the one built above
        archive.extractall(unrelated, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    wheel = _build(unrelated / sdist.name.removesuffix(".tar.gz"), "build_wheel", tmp_path / "wheel")
    assert _wheel_build_info(wheel) == {"git_commit": head, "git_dirty": False}

    # install by unpacking the wheel, then import it with no git on PATH and the
    # unrelated repository as the working directory
    site = tmp_path / "site"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(site)
    no_git = tmp_path / "empty-bin"
    no_git.mkdir()
    script = (
        "import json, climate_indices\n"
        "from climate_indices import provenance\n"
        "print(json.dumps({'file': climate_indices.__file__, **provenance.software_identity()}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=unrelated,
        capture_output=True,
        text=True,
        check=True,
        env=_env(PYTHONPATH=str(site), PATH=str(no_git)),
    )
    identity = json.loads(completed.stdout.strip().splitlines()[-1])
    assert Path(identity["file"]).resolve().is_relative_to(site.resolve())
    assert identity["git_commit"] == head != other_head
    assert (identity["git_commit_source"], identity["git_dirty"]) == ("build-metadata", False)
    assert identity["repository_url"] == "https://github.com/monocongo/climate_indices"


def test_a_wheel_built_from_a_modified_checkout_is_marked_dirty(tmp_path: Path) -> None:
    root, head = _project(tmp_path)
    with (root / "src" / "climate_indices" / "__init__.py").open("a") as module:
        module.write("\n# local change\n")
    # a stale, git-ignored copy in src/ is never shipped in place of the fresh one
    (root / "src" / "climate_indices" / "_build_info.json").write_text(json.dumps({"git_commit": "a" * 40}))

    wheel = _build(root, "build_wheel", tmp_path / "dist")

    assert _wheel_build_info(wheel) == {"git_commit": head, "git_dirty": True}


def _hook_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hatch_build", ROOT / "hatch_build.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_hook_embeds_nothing_for_an_editable_install_and_cleans_up(tmp_path: Path) -> None:
    root, head = _project(tmp_path)
    hook = _hook_module().ProvenanceBuildHook(str(root), {}, None, None, str(tmp_path / "out"), "wheel")

    editable: dict[str, Any] = {"force_include": {}}
    hook.initialize("editable", editable)
    standard: dict[str, Any] = {"force_include": {}}
    hook.initialize("standard", standard)

    assert editable == {"force_include": {}}
    [(generated, target)] = standard["force_include"].items()
    assert target == "climate_indices/_build_info.json"
    assert json.loads(Path(generated).read_text()) == {"git_commit": head, "git_dirty": False}
    hook.finalize("standard", standard, "")
    assert not Path(generated).exists()


def test_the_script_writes_build_info_for_maturin_only_in_a_checkout(tmp_path: Path) -> None:
    root, head = _project(tmp_path)
    loose = tmp_path / "loose"
    loose.mkdir()
    shutil.copy2(ROOT / "hatch_build.py", loose / "hatch_build.py")

    subprocess.run([sys.executable, "hatch_build.py"], cwd=root, check=True, capture_output=True, env=_env())
    refused = subprocess.run([sys.executable, "hatch_build.py"], cwd=loose, capture_output=True, text=True, env=_env())

    written = json.loads((root / "src" / "climate_indices" / "_build_info.json").read_text())
    assert written == {"git_commit": head, "git_dirty": False}
    assert _git(root, "status", "--porcelain") == ""  # git-ignored, so the tree stays clean
    assert refused.returncode != 0 and "not its own Git checkout" in refused.stderr
