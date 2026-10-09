"""Guard that the AWS benchmark tooling carries no account ids or credentials.

The benchmark infrastructure in `benchmarks/aws/` is committed to a public
repository. Terraform state holds the account id and resource ARNs and is
gitignored, which makes it easy for a future edit to reintroduce one of them
into a tracked file without noticing. This asserts the invariant instead of
trusting a review to spot it.

The account id used as a positive control is Amazon's own documented example
account (`123456789012`), never a real one, so this file cannot leak the very
value it exists to protect.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AWS_DIR = ROOT / "benchmarks" / "aws"
TERRAFORM_DIR = AWS_DIR / "terraform"

# Sample account id from the AWS documentation, safe to commit.
EXAMPLE_ACCOUNT_ID = "123456789012"

# The four credential shapes that matter: long-lived and temporary access keys,
# and a bare account id, which is enough to build resource ARNs. The account id
# pattern requires a non-alphanumeric delimiter on both sides, so a twelve-digit
# run inside a hex or base64 provider hash is not a false positive.
SECRET_PATTERNS = {
    "account id": re.compile(r"(?<![0-9A-Za-z])[0-9]{12}(?![0-9A-Za-z])"),
    "access key id": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "secret access key": re.compile(r"aws_secret_access_key"),
    "session token": re.compile(r"aws_session_token"),
}


def findings(text: str) -> list[str]:
    """Every credential-shaped value in ``text``, as ``pattern: match`` strings."""
    found = []
    for name, pattern in SECRET_PATTERNS.items():
        found.extend(f"{name}: {match.group(0)}" for match in pattern.finditer(text))
    return found


def tracked_files() -> list[Path]:
    """Files under benchmarks/aws that Git tracks.

    Deliberately not a filesystem walk: Terraform state, the provider plugin
    cache, and run logs all legitimately live under this directory and hold the
    account id, so a filesystem scan would report them as leaks.
    """
    listed = subprocess.run(  # nosec B603 # fixed argv, no shell, this repository
        ["git", "-C", str(ROOT), "ls-files", "benchmarks/aws"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return [ROOT / entry for entry in listed if entry]


def test_the_scanner_catches_what_it_claims_to() -> None:
    """A positive control, so a passing scan cannot just mean the patterns are broken."""
    assert findings(f"arn:aws:iam::{EXAMPLE_ACCOUNT_ID}:role/example") == [f"account id: {EXAMPLE_ACCOUNT_ID}"]
    assert findings(f'"account_id": "{EXAMPLE_ACCOUNT_ID}"') == [f"account id: {EXAMPLE_ACCOUNT_ID}"]
    assert findings("AKIAIOSFODNN7EXAMPLE") == ["access key id: AKIAIOSFODNN7EXAMPLE"]
    assert findings("aws_secret_access_key = abc") == ["secret access key: aws_secret_access_key"]
    assert findings("aws_session_token = abc") == ["session token: aws_session_token"]
    # negative controls: figures and hashes that only look like an account id
    assert findings("16384 MiB at 0.04048 per vCPU-hour") == []
    assert findings("h1:abc123456789012def456") == []


def test_aws_tooling_carries_no_account_id_or_credentials() -> None:
    """No tracked file under benchmarks/aws may name an account or hold a credential."""
    suspects = {}
    for path in tracked_files():
        found = findings(path.read_text(encoding="utf-8", errors="replace"))
        if found:
            suspects[str(path.relative_to(ROOT))] = found

    assert suspects == {}, f"account id or credential material in tracked files: {suspects}"


def test_terraform_state_is_not_tracked() -> None:
    """State holds the account id and ARNs, so Git must not track it, local or not."""
    tracked = subprocess.run(  # nosec B603 # fixed argv, no shell, this repository
        ["git", "-C", str(ROOT), "ls-files", "benchmarks/aws"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()

    assert not [entry for entry in tracked if "tfstate" in entry or ".terraform/" in entry]


def test_terraform_ignore_rules_cover_state() -> None:
    """The guarantee must not depend on the repository-root .gitignore staying intact."""
    rules = (TERRAFORM_DIR / ".gitignore").read_text(encoding="utf-8")

    assert "*.tfstate" in rules
    assert ".terraform/" in rules


@pytest.mark.parametrize("name", ["README.md", "bootstrap.sh", "run.sh", "spread.py"])
def test_tooling_files_exist(name: str) -> None:
    """A sanity check that the directory under guard is the one being scanned."""
    assert (AWS_DIR / name).is_file()
