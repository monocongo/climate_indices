#!/usr/bin/env python3
"""Find and quarantine ECC ("Everything Claude Code") artifacts in coding-agent directories.

ECC (https://github.com/affaan-m/ECC) installs agents, skills, commands, rules,
hooks, and instruction files into user-level agent directories such as
``~/.claude``, ``~/.agents``, ``~/.codex``, and ``~/.pi``. Those files then load
into every session in every repository, next to the project's own
``AGENTS.md``, and a ``~/.claude/AGENTS.md`` is read for any project under your
home directory that has no ``CLAUDE.md``.

The default run is a read-only audit. ``--apply`` moves each *confirmed* ECC
artifact into a timestamped quarantine directory and edits the shared files ECC
merged into (``settings.json`` hooks, ``<!-- BEGIN ECC -->`` blocks, skills
lock entries), keeping a byte copy of every edited file. Nothing is deleted,
and ``--restore`` puts a quarantine back.

Confidence tiers:

confirmed
    Recorded in an ECC install-state file and unchanged since install,
    byte-identical to a file ECC ships, installed from ECC according to a
    skills lock file, or carrying an ECC provenance marker. ``--apply``
    quarantines these.
name-match
    Shares a name with something ECC ships but carries no ECC marker: an older
    ECC version, or occasionally your own file. Quarantined only with
    ``--include-name-matches``, so review the list first.
manual
    Reported with a suggested command and never changed: plugin registries,
    Codex ``config.toml``, MCP servers in ``~/.claude.json``, shell start-up
    files, and ECC's own backups of files it replaced.

Usage::

    python3 scripts/agent_hygiene/audit_agent_home.py              # audit only
    python3 scripts/agent_hygiene/audit_agent_home.py --apply      # quarantine confirmed
    python3 scripts/agent_hygiene/audit_agent_home.py --project ~/src/other-repo
    python3 scripts/agent_hygiene/audit_agent_home.py --restore ~/.agent-quarantine/ecc-20260928-101500
    python3 scripts/agent_hygiene/audit_agent_home.py --refresh-fingerprints ~/src/ECC

Close running Claude Code, Codex, and pi sessions before ``--apply``: they
rewrite ``settings.json`` while they run.

Exit status: 0 when the audit is clean or ``--apply``/``--restore`` finished
without conflicts, 1 when the audit found artifacts or a restore hit
conflicts, 2 on usage or I/O errors.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FINGERPRINTS_PATH = Path(__file__).with_name("ecc_fingerprints.json")

CONFIRMED = "confirmed"
NAME_MATCH = "name-match"
MANUAL = "manual"
TIERS = (CONFIRMED, NAME_MATCH, MANUAL)

BEGIN_MARKER = "<!-- BEGIN ECC -->"
END_MARKER = "<!-- END ECC -->"
_BLOCK = re.compile(r"[ \t]*<!-- BEGIN ECC -->.*?<!-- END ECC -->[ \t]*\n?", re.DOTALL)
# Deliberately excludes a bare "ECC", which also means error-correcting code.
_ECC_TEXT = re.compile(r"everything-claude-code|Everything Claude Code|affaan-m/ecc\b|\becc@ecc\b", re.IGNORECASE)
_ECC_HEADING = re.compile(r"\A\s*#\s[^\n]*(Everything Claude Code|\(ECC\))")
_ECC_REPO = re.compile(r"affaan-m/(everything-claude-code|ecc)\b", re.IGNORECASE)
_ECC_LINK = re.compile(r"everything-claude-code|(^|[/\\])ecc([/\\@]|$)|ecc-homunculus", re.IGNORECASE)
_ORIGIN_ECC = re.compile(r"^\s*origin:\s*['\"]?ECC['\"]?\s*$", re.MULTILINE)
_AGENT_MARKER = "Prompt Defense Baseline"
_HOOK_SIGNATURE = re.compile(r"resolve-ecc-root|plugin-hook-bootstrap|everything-claude-code|\becc@ecc\b|\bECC_[A-Z_]+")
_HOOK_SCRIPT = re.compile(r"(?:CLAUDE_PLUGIN_ROOT\}?|\.claude)[/\\]scripts[/\\]hooks[/\\]([\w.-]+)")
_PLUGIN_ID = re.compile(r"^(ecc|everything-claude-code)@|@(ecc|everything-claude-code)$")
_MARKETPLACES = frozenset({"ecc", "everything-claude-code"})
_ENV_PREFIXES = ("ECC_", "CLV2_")
_TOML_TABLE = re.compile(r"^\s*\[+\s*([^\]]+?)\s*\]+\s*$", re.MULTILINE)
_SHELL_LINE = re.compile(r"\b(ECC_[A-Z_]+|CLV2_[A-Z_]+)\s*=|ecc-universal|ecc-install|everything-claude-code")

# Files ECC ships that are smaller than this are too generic to fingerprint.
_MIN_FINGERPRINT_BYTES = 64
_MAX_DIR_FILES = 2000

# Shared files other tools also write. ECC's entries in them are edited out by
# the dedicated scanners; the files are never moved as a whole.
_SHARED_FILES = frozenset(
    {
        "AGENTS.md",
        "AGENTS.override.md",
        "CLAUDE.md",
        "GEMINI.md",
        "settings.json",
        "settings.local.json",
        "hooks.json",
        "config.toml",
        ".claude.json",
        "mcp.json",
    }
)
_LABELS = {
    "skills": "skill",
    "agents": "agent",
    "commands": "command",
    "rules": "rule pack",
    "rule_files": "rule",
    "hooks": "hook file",
    "docs": "doc set",
}

# Directories a harness or the user owns. They are never moved or pruned as a
# whole; their entries are classified one by one instead.
_CONTAINERS = frozenset(
    {
        "",
        "skills",
        "agents",
        "commands",
        "rules",
        "hooks",
        "scripts",
        "scripts/hooks",
        "scripts/lib",
        "docs",
        "plugins",
        "projects",
        "prompts",
        "sessions",
    }
)
# Containers that stay even when quarantining leaves them empty.
_KEEP_WHEN_EMPTY = frozenset({"", "skills", "agents", "commands", "rules", "hooks", "plugins", "projects", "prompts"})

# Data ECC's hooks write under ~/.claude. Claude Code creates none of these.
_CLAUDE_DATA_CONFIRMED = (
    "ecc",
    "session-data",
    "session-aliases.json",
    "plan-canvas",
    "mcp-health-cache.json",
    "package-manager.json",
    "homunculus",
)
_CLAUDE_DATA_NAME_MATCH = ("claw", "orchestration", "metrics")

_SHELL_FILES = (".zshrc", ".zprofile", ".zshenv", ".bashrc", ".bash_profile", ".profile", ".config/fish/config.fish")
_PROJECT_ADAPTER_DIRS = (
    ".cursor",
    ".kimi-code",
    ".kimi",
    ".opencode",
    ".gemini",
    ".qwen",
    ".zed",
    ".codebuddy",
    ".joycode",
    ".adal",
    ".agents",
)


@dataclass(frozen=True)
class Fingerprints:
    """Names and content hashes of the files one ECC release ships."""

    version: str
    commit: str
    names: dict[str, frozenset[str]]
    hashes: frozenset[str]

    def name(self, kind: str) -> frozenset[str]:
        return self.names.get(kind, frozenset())


@dataclass
class Finding:
    """One ECC artifact, or one thing to review by hand."""

    tier: str
    path: Path
    reason: str
    action: str = "move"  # "move", "edit", or "none"
    new_text: str | None = None
    original_sha256: str | None = None
    hint: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "path": str(self.path),
            "action": self.action,
            "reason": self.reason,
            "hint": self.hint,
        }


def content_hash(path: Path) -> str:
    """Return a file's 16-hex-digit fingerprint, with CRLF normalised to LF."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def _fingerprint(path: Path) -> str | None:
    """Return ``content_hash`` for a readable, non-trivial file, else ``None``."""
    try:
        if path.stat().st_size < _MIN_FINGERPRINT_BYTES:
            return None
        return content_hash(path)
    except OSError:
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _read_text(path: Path, limit: int | None = None) -> str:
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            return handle.read(limit) if limit else handle.read()
    except OSError:
        return ""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _dump_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(os.path.abspath(path)).relative_to(os.path.abspath(root))
    except ValueError:
        return False
    return True


def _iter_files(root: Path) -> Iterator[Path]:
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in {".git", "node_modules"}]
        for filename in filenames:
            count += 1
            if count > _MAX_DIR_FILES:
                return
            yield Path(dirpath, filename)


def load_fingerprints(path: Path = FINGERPRINTS_PATH) -> Fingerprints:
    """Load the committed ECC fingerprint file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return Fingerprints(
        version=str(data["version"]),
        commit=str(data.get("commit", "")),
        names={kind: frozenset(values) for kind, values in data["names"].items()},
        hashes=frozenset(data["hashes"]),
    )


class Scanner:
    """Collect findings for one home directory and any extra project roots."""

    def __init__(self, fingerprints: Fingerprints, home: Path) -> None:
        self.fp = fingerprints
        self.home = home
        self.findings: list[Finding] = []
        self.locked_skills: set[str] = set()
        self._seen: set[str] = set()

    # -- bookkeeping ---------------------------------------------------------

    def add(
        self,
        tier: str,
        path: Path,
        reason: str,
        *,
        action: str = "move",
        new_text: str | None = None,
        hint: str | None = None,
    ) -> None:
        key = f"{os.path.abspath(path)}::{action}"
        if key in self._seen:
            return
        self._seen.add(key)
        original = _sha256(path) if action == "edit" else None
        self.findings.append(Finding(tier, path, reason, action, new_text, original, hint))

    def display(self, path: Path) -> str:
        text = os.path.abspath(path)
        home = os.path.abspath(self.home)
        return "~" + text[len(home) :] if text == home or text.startswith(home + os.sep) else text

    # -- classification ------------------------------------------------------

    def _hash_match(self, path: Path) -> bool:
        if path.is_file():
            return _fingerprint(path) in self.fp.hashes
        if path.is_dir():
            skill = path / "SKILL.md"
            if skill.is_file() and self._hash_match(skill):
                return True
            prints = [p for p in map(_fingerprint, _iter_files(path)) if p is not None]
            matched = sum(1 for p in prints if p in self.fp.hashes)
            return matched > 0 and matched * 2 >= len(prints)
        return False

    def classify(self, path: Path, name: str, kind: str, marker: Callable[[Path], str | None] | None) -> None:
        """Classify one entry of a skills/agents/commands-style container."""
        names = self.fp.name(kind)
        label = _LABELS.get(kind, kind)
        if path.is_symlink() and not path.exists():
            link = os.readlink(path)
            if _ECC_LINK.search(link):
                self.add(CONFIRMED, path, f"dangling symlink into an ECC install ({link})")
            elif name in names:
                self.add(NAME_MATCH, path, f"dangling symlink named like an ECC {label}")
            return
        target = path.resolve() if path.is_symlink() else path
        via = f" (symlink to {self.display(target)})" if path.is_symlink() else ""
        if kind == "skills" and name in self.locked_skills:
            self.add(CONFIRMED, path, f"installed from affaan-m/ECC per a skills lock file{via}")
        elif self._hash_match(target):
            self.add(CONFIRMED, path, f"byte-identical to ECC {self.fp.version}{via}")
        elif marker and (why := marker(target)):
            self.add(CONFIRMED, path, why + via)
        elif name in names:
            self.add(
                NAME_MATCH,
                path,
                f"named like an ECC {label} but no ECC marker (older ECC, or yours?){via}",
            )

    def scan_container(
        self,
        container: Path,
        kind: str,
        marker: Callable[[Path], str | None] | None,
        *,
        strip_suffix: bool = False,
        skip: frozenset[str] = frozenset(),
    ) -> None:
        if not container.is_dir():
            return
        for entry in sorted(container.iterdir()):
            if entry.name.startswith(".") or entry.name in skip:
                continue
            name = Path(entry.name).stem if strip_suffix and not entry.is_dir() else entry.name
            self.classify(entry, name, kind, marker)

    # -- markers -------------------------------------------------------------

    @staticmethod
    def skill_marker(path: Path) -> str | None:
        head = _read_text(path / "SKILL.md", 4096) if path.is_dir() else ""
        match = re.match(r"\A---\n(.*?)\n---", head, re.DOTALL)
        front = match.group(1) if match else ""
        if _ORIGIN_ECC.search(front):
            return "SKILL.md frontmatter says origin: ECC"
        if _ECC_TEXT.search(front):
            return "SKILL.md frontmatter references ECC"
        return None

    @staticmethod
    def agent_marker(path: Path) -> str | None:
        text = _read_text(path, 65536) if path.is_file() else ""
        if _AGENT_MARKER in text:
            return f"carries ECC's '{_AGENT_MARKER}' section"
        return Scanner.text_marker(path)

    @staticmethod
    def text_marker(path: Path) -> str | None:
        if path.is_file():
            text = _read_text(path, 65536)
        elif path.is_dir():
            text = "".join(_read_text(f, 16384) for f in list(_iter_files(path))[:20])
        else:
            text = ""
        return "content references ECC" if _ECC_TEXT.search(text) else None

    # -- install state and locks ---------------------------------------------

    def scan_install_state(self, state_file: Path, root: Path, *, move_path: Path | None = None) -> None:
        if not state_file.is_file():
            return
        self.add(CONFIRMED, move_path or state_file, "ECC install-state record")
        data = _load_json(state_file)
        operations = data.get("operations") if isinstance(data, dict) else None
        if not isinstance(operations, list):
            self.add(MANUAL, state_file, "ECC install-state file is unreadable; inspect it by hand", action="none")
            return
        containers = {os.path.abspath(root / c) for c in _CONTAINERS}
        for op in operations:
            if not isinstance(op, dict) or op.get("kind") == "update-claude-settings":
                continue
            raw = op.get("destinationPath")
            if not isinstance(raw, str):
                continue
            dest = Path(raw).expanduser()
            if (
                dest.name in _SHARED_FILES
                or not _inside(dest, root)
                or os.path.abspath(dest) in containers
                or not _exists(dest)
            ):
                continue
            sha = op.get("contentSha256")
            if isinstance(sha, str) and dest.is_file() and not dest.is_symlink() and _sha256(dest) != sha.lower():
                self.add(NAME_MATCH, dest, "recorded in ECC install-state but modified since install")
            else:
                self.add(CONFIRMED, dest, "recorded in ECC install-state")

    def scan_skill_lock(self, lock: Path) -> None:
        data = _load_json(lock)
        skills = data.get("skills") if isinstance(data, dict) else None
        if not isinstance(skills, dict):
            return
        ecc = sorted(name for name, entry in skills.items() if _ECC_REPO.search(json.dumps(entry)))
        if not ecc:
            return
        self.locked_skills.update(ecc)
        updated = dict(data)
        updated["skills"] = {name: entry for name, entry in skills.items() if name not in ecc}
        self.add(
            CONFIRMED,
            lock,
            f"drop {len(ecc)} skill(s) installed from affaan-m/ECC: {', '.join(ecc)}",
            action="edit",
            new_text=_dump_json(updated),
        )

    # -- shared files ECC merges into ---------------------------------------

    def _is_ecc_hook(self, hook: Any) -> bool:
        if not isinstance(hook, dict):
            return False
        command = hook.get("command")
        if not isinstance(command, str):
            return False
        if _HOOK_SIGNATURE.search(command):
            return True
        return any(script in self.fp.name("hook_scripts") for script in _HOOK_SCRIPT.findall(command))

    def strip_hooks(self, hooks: dict[str, Any]) -> int:
        """Remove ECC hook commands in place; return how many were removed."""
        removed = 0
        for event in list(hooks):
            groups = hooks[event]
            if not isinstance(groups, list):
                continue
            kept_groups: list[Any] = []
            event_removed = 0
            for group in groups:
                inner = group.get("hooks") if isinstance(group, dict) else None
                if not isinstance(inner, list):
                    kept_groups.append(group)
                    continue
                kept = [hook for hook in inner if not self._is_ecc_hook(hook)]
                event_removed += len(inner) - len(kept)
                if kept or not inner:
                    kept_groups.append({**group, "hooks": kept})
            removed += event_removed
            if event_removed:
                if kept_groups:
                    hooks[event] = kept_groups
                else:
                    del hooks[event]
        return removed

    def scan_settings(self, path: Path) -> None:
        data = _load_json(path)
        if not isinstance(data, dict):
            return
        updated = copy.deepcopy(data)
        changes: list[str] = []

        hooks = updated.get("hooks")
        if isinstance(hooks, dict) and (count := self.strip_hooks(hooks)):
            changes.append(f"{count} ECC hook command(s)")
            if not hooks:
                del updated["hooks"]

        def drop(section: str, is_ecc: Callable[[str, Any], bool]) -> None:
            entries = updated.get(section)
            if not isinstance(entries, dict):
                return
            keys = [key for key, value in entries.items() if is_ecc(key, value)]
            for key in keys:
                del entries[key]
            if keys:
                changes.append(f"{section} " + ", ".join(keys))
                if not entries:
                    del updated[section]

        drop("enabledPlugins", lambda key, _: bool(_PLUGIN_ID.search(key)))
        drop(
            "extraKnownMarketplaces",
            lambda key, value: key in _MARKETPLACES or bool(_ECC_REPO.search(json.dumps(value))),
        )
        drop("env", lambda key, _: key.startswith(_ENV_PREFIXES))

        status = updated.get("statusLine")
        if isinstance(status, dict) and _HOOK_SIGNATURE.search(json.dumps(status)):
            del updated["statusLine"]
            changes.append("statusLine")

        if changes:
            self.add(CONFIRMED, path, "remove " + "; ".join(changes), action="edit", new_text=_dump_json(updated))
        if data.get("includeCoAuthoredBy") is False:
            self.add(
                MANUAL,
                path,
                "includeCoAuthoredBy is false (ECC sets this). The key is deprecated; to keep attribution off, "
                'replace it with "attribution": {"commit": "", "pr": ""}',
                action="none",
            )

    def scan_hooks_json(self, path: Path) -> None:
        data = _load_json(path)
        hooks = data.get("hooks") if isinstance(data, dict) else None
        if not isinstance(hooks, dict):
            return
        updated = copy.deepcopy(data)
        if count := self.strip_hooks(updated["hooks"]):
            self.add(
                CONFIRMED, path, f"remove {count} ECC hook command(s)", action="edit", new_text=_dump_json(updated)
            )

    def scan_instructions(self, path: Path) -> None:
        if not path.is_file() or path.is_symlink():
            return
        text = _read_text(path)
        if BEGIN_MARKER in text and END_MARKER in text:
            remaining = _BLOCK.sub("", text)
            if not remaining.strip():
                self.add(CONFIRMED, path, "holds nothing but an ECC-managed block")
            else:
                self.add(
                    CONFIRMED,
                    path,
                    f"remove the {BEGIN_MARKER} ... {END_MARKER} block; keep the rest",
                    action="edit",
                    new_text=remaining.rstrip("\n") + "\n",
                )
        elif self._hash_match(path) or _ECC_HEADING.search(text):
            self.add(CONFIRMED, path, "ECC's own instruction file, copied here")
        elif _ECC_TEXT.search(text):
            self.add(MANUAL, path, "mentions ECC outside a managed block; review by hand", action="none")

    # -- roots ---------------------------------------------------------------

    def scan_claude_root(self, root: Path, *, home_level: bool) -> None:
        """Scan ``~/.claude`` or a project's ``.claude``."""
        if not root.is_dir():
            return
        self.scan_install_state(root / "ecc" / "install-state.json", root, move_path=root / "ecc")
        self.scan_container(root / "skills", "skills", self.skill_marker, skip=frozenset({"learned"}))
        if (root / "skills" / "learned").is_dir():
            self.add(CONFIRMED, root / "skills" / "learned", "ECC continuous-learning output")
        self.scan_container(root / "agents", "agents", self.agent_marker, strip_suffix=True)
        self.scan_container(root / "commands", "commands", self.text_marker, strip_suffix=True)
        self.scan_rules(root / "rules")
        self.scan_container(root / "hooks", "hooks", self.text_marker)
        self.scan_scripts(root / "scripts")
        self.scan_container(root / "docs", "docs", self.text_marker)
        for name in ("CLAUDE.md", "AGENTS.md"):
            self.scan_instructions(root / name)
        for name in ("settings.json", "settings.local.json"):
            self.scan_settings(root / name)
        for name in ("the-security-guide.md", "mcp-configs"):
            if _exists(root / name) and self._hash_match(root / name):
                self.add(CONFIRMED, root / name, f"byte-identical to ECC {self.fp.version}")
        plugin = _load_json(root / ".claude-plugin" / "plugin.json")
        if isinstance(plugin, dict) and plugin.get("name") in _MARKETPLACES:
            self.add(CONFIRMED, root / ".claude-plugin", "ECC plugin manifest copied into this directory")
        if home_level:
            for name in _CLAUDE_DATA_CONFIRMED:
                if _exists(root / name):
                    self.add(CONFIRMED, root / name, "data written by ECC hooks")
            for name in _CLAUDE_DATA_NAME_MATCH:
                if _exists(root / name):
                    self.add(NAME_MATCH, root / name, "named like data ECC writes; check before removing")

    def scan_rules(self, rules: Path) -> None:
        if not rules.is_dir():
            return
        if (rules / "ecc").is_dir():
            self.add(CONFIRMED, rules / "ecc", "ECC's namespaced rule pack")
        for entry in sorted(rules.iterdir()):
            if entry.name.startswith(".") or entry.name == "ecc":
                continue
            if entry.is_dir():
                self.classify(entry, entry.name, "rules", self.text_marker)
            elif entry.suffix == ".md":
                self.classify(entry, entry.stem, "rule_files", self.text_marker)

    def scan_scripts(self, scripts: Path) -> None:
        """Classify the hook runtime ECC copies into ``~/.claude/scripts``."""
        if not scripts.is_dir():
            return
        runtime = (scripts / "lib" / "resolve-ecc-root.js").is_file()
        known = self.fp.name("scripts")

        def is_ecc(path: Path) -> bool:
            rel = path.relative_to(scripts).as_posix()
            return _fingerprint(path) in self.fp.hashes or (runtime and rel in known)

        tier = CONFIRMED if runtime else NAME_MATCH
        for entry in sorted(scripts.iterdir()):
            if entry.name.startswith("."):
                continue
            if entry.is_dir() and not entry.is_symlink():
                files = list(_iter_files(entry))
                ecc_files = [f for f in files if is_ecc(f)]
                if files and len(ecc_files) == len(files):
                    self.add(tier, entry, "ECC hook runtime")
                else:
                    for file in ecc_files:
                        self.add(tier, file, "ECC hook runtime file")
            elif entry.is_file() and is_ecc(entry):
                self.add(tier, entry, "ECC hook runtime file")

    def scan_agents_root(self, root: Path) -> None:
        """Scan ``~/.agents`` or a project's ``.agents``."""
        if not root.is_dir():
            return
        self.scan_container(root / "skills", "skills", self.skill_marker)
        market = _load_json(root / "plugins" / "marketplace.json")
        if isinstance(market, dict) and market.get("name") in _MARKETPLACES:
            self.add(CONFIRMED, root / "plugins" / "marketplace.json", "ECC plugin marketplace definition")
        self.scan_instructions(root / "AGENTS.md")

    def scan_codex_root(self, root: Path) -> None:
        if not root.is_dir():
            return
        self.scan_install_state(root / "ecc-install-state.json", root)
        for name in ("AGENTS.md", "AGENTS.override.md"):
            self.scan_instructions(root / name)
        self.scan_container(root / "agents", "agents", self.agent_marker, strip_suffix=True)
        self.scan_container(root / "skills", "skills", self.skill_marker)
        self.scan_container(root / "prompts", "commands", self.text_marker, strip_suffix=True)
        self.scan_hooks_json(root / "hooks.json")
        config = root / "config.toml"
        if config.is_file():
            ecc_tables = self.fp.name("codex_tables")
            tables = []
            for header in _TOML_TABLE.findall(_read_text(config)):
                key = header.replace('"', "").replace(" ", "")
                if key in ecc_tables or any(_ECC_LINK.search(part) for part in key.split(".")):
                    tables.append(f"[{header}]")
            if tables:
                self.add(
                    MANUAL,
                    config,
                    "tables ECC's Codex config also defines: " + ", ".join(tables),
                    action="none",
                    hint="delete the ones you did not add yourself (ECC's [profiles.yolo] disables approvals)",
                )
        if any(_ECC_LINK.search(p.name) for p in (root / "plugins").glob("**/*") if p.is_dir()):
            self.add(
                MANUAL, root / "plugins", "ECC Codex plugin present", action="none", hint="codex plugin remove ecc@ecc"
            )
        for backup in sorted((root / "backups").glob("ecc-*")):
            self.add(
                MANUAL,
                backup,
                "ECC's backup of the Codex AGENTS.md/config.toml it replaced",
                action="none",
                hint="compare with the live files; restore anything of yours that ECC overwrote",
            )

    def scan_pi_root(self, root: Path) -> None:
        if not root.is_dir():
            return
        self.scan_instructions(root / "AGENTS.md")
        self.scan_container(root / "skills", "skills", self.skill_marker)
        self.scan_container(root / "prompts", "commands", self.text_marker, strip_suffix=True)

    def scan_cursor_root(self, root: Path) -> None:
        if not root.is_dir():
            return
        if (root / "ecc").is_dir():
            self.add(CONFIRMED, root / "ecc", "ECC memory-isolation data")
        for pattern in ("agents/ecc-*.md", "rules/ecc-*.mdc", "ecc-agent-data.json"):
            for path in sorted(root.glob(pattern)):
                self.add(CONFIRMED, path, "ECC-prefixed Cursor file")
        self.scan_container(root / "skills", "skills", self.skill_marker)

    def scan_plugins(self, claude: Path) -> None:
        plugins = claude / "plugins"
        found = [
            p for sub in ("marketplaces", "cache") for p in sorted((plugins / sub).glob("*")) if p.name in _MARKETPLACES
        ]
        for name in ("known_marketplaces.json", "installed_plugins.json"):
            text = _read_text(plugins / name)
            if re.search(r'"(ecc|everything-claude-code)(@[\w-]+)?"', text):
                found.append(plugins / name)
        if found:
            self.add(
                MANUAL,
                plugins,
                "ECC Claude Code plugin registered: " + ", ".join(self.display(p) for p in found),
                action="none",
                hint="claude plugin uninstall ecc@ecc; claude plugin marketplace remove ecc",
            )

    def scan_claude_json(self, path: Path) -> None:
        data = _load_json(path)
        if not isinstance(data, dict):
            return
        scopes: list[tuple[str, Any]] = [("user", data.get("mcpServers"))]
        projects = data.get("projects")
        if isinstance(projects, dict):
            scopes += [
                (f"project {key}", value.get("mcpServers"))
                for key, value in projects.items()
                if isinstance(value, dict)
            ]
        hits = [
            f"{name} ({scope})"
            for scope, servers in scopes
            if isinstance(servers, dict)
            for name, server in servers.items()
            if name.startswith("ecc") or _HOOK_SIGNATURE.search(json.dumps(server))
        ]
        if hits:
            self.add(
                MANUAL,
                path,
                "MCP servers wired to ECC: " + ", ".join(hits),
                action="none",
                hint="claude mcp remove <name> (add --scope project/local as listed)",
            )

    def scan_shell_files(self) -> None:
        for name in _SHELL_FILES:
            path = self.home / name
            lines = [
                f"{number}: {line.strip()}"
                for number, line in enumerate(_read_text(path).splitlines(), start=1)
                if _SHELL_LINE.search(line)
            ]
            if lines:
                self.add(
                    MANUAL, path, "ECC settings: " + " | ".join(lines), action="none", hint="delete these lines by hand"
                )

    def scan_home(self) -> None:
        home = self.home
        for lock in (home / ".agents" / ".skill-lock.json", home / ".agents" / "skills-lock.json"):
            self.scan_skill_lock(lock)
        self.scan_claude_root(home / ".claude", home_level=True)
        self.scan_plugins(home / ".claude")
        self.scan_claude_json(home / ".claude.json")
        self.scan_agents_root(home / ".agents")
        self.scan_codex_root(home / ".codex")
        self.scan_pi_root(home / ".pi" / "agent")
        self.scan_cursor_root(home / ".cursor")
        for name in ("AGENTS.md", "CLAUDE.md"):
            self.scan_instructions(home / name)
        self.scan_instructions(home / ".gemini" / "GEMINI.md")
        for parts in ((".config", "opencode"), (".qwen",), (".hermes",), (".openclaw",)):
            root = home.joinpath(*parts)
            self.scan_install_state(root / "ecc-install-state.json", root)
            self.scan_instructions(root / "AGENTS.md")
        homunculus = home / ".local" / "share" / "ecc-homunculus"
        if homunculus.is_dir():
            self.add(CONFIRMED, homunculus, "ECC continuous-learning data")
        self.scan_shell_files()

    def scan_project(self, project: Path) -> None:
        for lock in (project / "skills-lock.json", project / ".agents" / ".skill-lock.json"):
            self.scan_skill_lock(lock)
        self.scan_claude_root(project / ".claude", home_level=False)
        self.scan_agents_root(project / ".agents")
        self.scan_cursor_root(project / ".cursor")
        for name in ("AGENTS.md", "CLAUDE.md", "GEMINI.md", ".github/copilot-instructions.md", ".claude/CLAUDE.md"):
            self.scan_instructions(project / name)
        for name in _PROJECT_ADAPTER_DIRS:
            root = project / name
            self.scan_install_state(root / "ecc-install-state.json", root)


# -- apply and restore --------------------------------------------------------


def _selected(findings: list[Finding], include_name_matches: bool) -> list[Finding]:
    tiers = {CONFIRMED, NAME_MATCH} if include_name_matches else {CONFIRMED}
    chosen = [f for f in findings if f.tier in tiers and f.action in {"move", "edit"}]
    moved = sorted({os.path.abspath(f.path) for f in chosen if f.action == "move"}, key=len)
    kept: list[Finding] = []
    for finding in chosen:
        here = os.path.abspath(finding.path)
        if any(here != other and here.startswith(other + os.sep) for other in moved):
            continue  # an ancestor directory is already being moved
        if finding.action == "edit" and here in moved:
            continue
        kept.append(finding)
    return kept


def _quarantine_path(files_dir: Path, path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    return files_dir / absolute.relative_to(absolute.anchor)


def _write_manifest(qdir: Path, manifest: dict[str, Any]) -> None:
    tmp = qdir / "manifest.json.tmp"
    tmp.write_text(_dump_json(manifest), encoding="utf-8")
    os.replace(tmp, qdir / "manifest.json")


def _prune_empty_parents(path: Path, roots: list[Path]) -> None:
    parent = Path(os.path.abspath(path)).parent
    while True:
        root = next((r for r in roots if _inside(parent, r)), None)
        if root is None:
            return
        rel = parent.relative_to(os.path.abspath(root)).as_posix()
        if rel in {".", ""} or rel in _KEEP_WHEN_EMPTY:
            return
        try:
            parent.rmdir()
        except OSError:
            return
        parent = parent.parent


def apply(findings: list[Finding], quarantine_root: Path, roots: list[Path], include_name_matches: bool) -> Path:
    """Quarantine the selected findings and return the quarantine directory."""
    chosen = _selected(findings, include_name_matches)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    qdir = quarantine_root / f"ecc-{stamp}"
    suffix = 1
    while qdir.exists():
        suffix += 1
        qdir = quarantine_root / f"ecc-{stamp}-{suffix}"
    files_dir = qdir / "files"
    files_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {"created": stamp, "moves": [], "edits": [], "skipped": []}
    _write_manifest(qdir, manifest)

    for finding in (f for f in chosen if f.action == "move"):
        if not _exists(finding.path):
            continue
        dest = _quarantine_path(files_dir, finding.path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(os.path.abspath(finding.path), dest)
        manifest["moves"].append({"from": os.path.abspath(finding.path), "to": str(dest), "reason": finding.reason})
        _write_manifest(qdir, manifest)
        _prune_empty_parents(finding.path, roots)

    for finding in (f for f in chosen if f.action == "edit"):
        path = finding.path
        if not path.is_file() or _sha256(path) != finding.original_sha256 or finding.new_text is None:
            manifest["skipped"].append({"path": str(path), "reason": "changed since the audit; rerun"})
            _write_manifest(qdir, manifest)
            continue
        backup = _quarantine_path(files_dir, path)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        tmp = path.with_name(path.name + ".ecc-audit.tmp")
        tmp.write_text(finding.new_text, encoding="utf-8")
        shutil.copymode(path, tmp)
        os.replace(tmp, path)
        manifest["edits"].append(
            {"path": str(path), "backup": str(backup), "sha256_after": _sha256(path), "reason": finding.reason}
        )
        _write_manifest(qdir, manifest)
    return qdir


def restore(qdir: Path, out: Callable[[str], None] = print) -> int:
    """Undo one quarantine; return the number of conflicts."""
    manifest = json.loads((qdir / "manifest.json").read_text(encoding="utf-8"))
    conflicts = 0
    for edit in reversed(manifest.get("edits", [])):
        path, backup = Path(edit["path"]), Path(edit["backup"])
        if path.is_file() and _sha256(path) != edit["sha256_after"]:
            out(f"conflict: {path} changed after quarantine; its original is at {backup}")
            conflicts += 1
            continue
        shutil.copy2(backup, path)
        out(f"restored {path}")
    for move in reversed(manifest.get("moves", [])):
        src, dst = Path(move["to"]), Path(move["from"])
        if _exists(dst):
            out(f"conflict: {dst} exists again; left the quarantined copy at {src}")
            conflicts += 1
            continue
        if not _exists(src):
            out(f"missing: {src} is gone from the quarantine")
            conflicts += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), dst)
        out(f"restored {dst}")
    return conflicts


# -- fingerprint refresh ------------------------------------------------------


def build_fingerprints(checkout: Path) -> dict[str, Any]:
    """Derive the fingerprint file from an ECC source checkout."""
    modules = json.loads((checkout / "manifests" / "install-modules.json").read_text(encoding="utf-8"))["modules"]
    module_paths = [p for module in modules for p in module.get("paths", [])]

    def expand(rel: str) -> list[Path]:
        path = checkout / rel
        return sorted(f for f in _iter_files(path)) if path.is_dir() else ([path] if path.is_file() else [])

    script_files = [f for rel in module_paths if rel.startswith("scripts/") for f in expand(rel)]
    mcp = _load_json(checkout / "mcp-configs" / "mcp-servers.json") or {}
    codex_config = _read_text(checkout / ".codex" / "config.toml")
    names = {
        "skills": {p.name for base in ("skills", ".agents/skills") for p in (checkout / base).glob("*") if p.is_dir()},
        "agents": {p.stem for p in (checkout / "agents").glob("*.md")},
        "commands": {
            p.stem for base in ("commands", "legacy-command-shims/commands") for p in (checkout / base).glob("*.md")
        },
        "rules": {p.name for p in (checkout / "rules").iterdir() if p.is_dir()},
        "rule_files": {p.stem for p in (checkout / "rules").rglob("*.md")},
        "hooks": {
            p.name for p in (checkout / "hooks").iterdir() if p.name not in {"hooks.json", "hooks.metadata.json"}
        },
        "hook_scripts": {p.name for p in (checkout / "scripts" / "hooks").rglob("*") if p.is_file()},
        "scripts": {f.relative_to(checkout / "scripts").as_posix() for f in script_files},
        "docs": {Path(rel).name for rel in module_paths if rel.startswith("docs/")},
        "mcp_servers": set(mcp.get("mcpServers", {})),
        "codex_tables": {
            header.replace('"', "").replace(" ", "") for header in _TOML_TABLE.findall(codex_config) if "." in header
        },
    }
    hash_roots = [
        "agents",
        "commands",
        "rules",
        "skills",
        ".agents/skills",
        "hooks",
        "mcp-configs",
        "legacy-command-shims",
        ".claude-plugin",
        ".codex",
        "AGENTS.md",
        "CLAUDE.md",
        "the-security-guide.md",
    ]
    files = [f for rel in hash_roots for f in expand(rel)] + script_files
    hashes = {p for p in map(_fingerprint, files) if p is not None}
    try:
        commit = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = ""
    return {
        "source": "https://github.com/affaan-m/ECC",
        "version": (checkout / "VERSION").read_text(encoding="utf-8").strip(),
        "commit": commit,
        "names": {kind: sorted(values) for kind, values in names.items()},
        "hashes": sorted(hashes),
    }


# -- reporting ----------------------------------------------------------------

_TIER_HEADINGS = {
    CONFIRMED: "confirmed ECC artifacts (quarantined by --apply)",
    NAME_MATCH: "name matches only (quarantined with --apply --include-name-matches; review first)",
    MANUAL: "review by hand (never changed by this script)",
}


def render(scanner: Scanner, verbose: bool) -> str:
    lines = [f"ECC fingerprints: {scanner.fp.version} ({scanner.fp.commit or 'unknown commit'})", ""]
    if not scanner.findings:
        return "\n".join([*lines, "No ECC artifacts found."])
    wrap = textwrap.TextWrapper(width=100, initial_indent="      ", subsequent_indent="      ")
    for tier in TIERS:
        tier_findings = [f for f in scanner.findings if f.tier == tier]
        if not tier_findings:
            continue
        lines.append(f"{_TIER_HEADINGS[tier]}: {len(tier_findings)}")
        grouped: dict[tuple[str, str], list[Finding]] = {}
        for finding in tier_findings:
            if finding.action == "move" and not verbose:
                key = (scanner.display(finding.path.parent) + "/", finding.reason.split(" (symlink")[0])
            else:
                key = (scanner.display(finding.path), finding.path.name)
            grouped.setdefault(key, []).append(finding)
        for (where, _), group in grouped.items():
            first = group[0]
            if first.action == "move" and not verbose:
                lines.append(f"  {where}  ({len(group)}) {first.reason.split(' (symlink')[0]}")
                lines.extend(wrap.wrap(", ".join(f.path.name for f in group)))
            else:
                tag = {"move": "move", "edit": "edit", "none": "info"}[first.action]
                lines.append(f"  [{tag}] {where}: {first.reason}")
                if first.hint:
                    lines.append(f"      -> {first.hint}")
        lines.append("")
    return "\n".join(lines).rstrip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--home", type=Path, default=Path.home(), help="home directory to scan (default: yours)")
    parser.add_argument("--project", type=Path, action="append", default=[], help="also scan a repository checkout")
    parser.add_argument("--apply", action="store_true", help="quarantine confirmed artifacts")
    parser.add_argument("--include-name-matches", action="store_true", help="with --apply, quarantine name matches too")
    parser.add_argument("--quarantine-dir", type=Path, help="where quarantines go (default: <home>/.agent-quarantine)")
    parser.add_argument("--restore", type=Path, metavar="QUARANTINE", help="undo one quarantine")
    parser.add_argument("--json", action="store_true", help="print findings as JSON")
    parser.add_argument("--verbose", action="store_true", help="list every path instead of grouping")
    parser.add_argument("--fingerprints", type=Path, default=FINGERPRINTS_PATH, help=argparse.SUPPRESS)
    parser.add_argument(
        "--refresh-fingerprints", type=Path, metavar="ECC_CHECKOUT", help="rebuild the fingerprint file from ECC"
    )
    args = parser.parse_args(argv)

    try:
        if args.refresh_fingerprints:
            data = build_fingerprints(args.refresh_fingerprints)
            args.fingerprints.write_text(_dump_json(data), encoding="utf-8")
            print(f"wrote {args.fingerprints}: ECC {data['version']}, {len(data['hashes'])} file hashes")
            return 0
        if args.restore:
            return 1 if restore(args.restore) else 0

        home = args.home.expanduser()
        scanner = Scanner(load_fingerprints(args.fingerprints), home)
        scanner.scan_home()
        projects = [p.expanduser() for p in args.project]
        for project in projects:
            scanner.scan_project(project)

        if args.json:
            print(json.dumps([f.to_json() for f in scanner.findings], indent=2))
        else:
            print(render(scanner, args.verbose))

        if not args.apply:
            actionable = any(f.tier in {CONFIRMED, NAME_MATCH} for f in scanner.findings)
            if actionable and not args.json:
                print("\nDry run: nothing changed. Rerun with --apply to quarantine the confirmed items.")
            return 1 if scanner.findings else 0

        roots = [home / ".claude", home / ".agents", home / ".codex", home / ".pi" / "agent", home / ".cursor"]
        roots += [p / sub for p in projects for sub in (".claude", ".agents", ".cursor")]
        qdir = apply(
            scanner.findings, args.quarantine_dir or home / ".agent-quarantine", roots, args.include_name_matches
        )
        manifest = json.loads((qdir / "manifest.json").read_text(encoding="utf-8"))
        print(f"\nQuarantined {len(manifest['moves'])} path(s) and edited {len(manifest['edits'])} file(s) into {qdir}")
        for skipped in manifest["skipped"]:
            print(f"  skipped {skipped['path']}: {skipped['reason']}")
        print(f"Undo with: python3 {Path(__file__).as_posix()} --restore {qdir}")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
