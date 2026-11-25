"""Local data and source-code provenance, without a tracking SDK."""

import hashlib
import subprocess
from pathlib import Path


def data_fingerprint(raw_data):
    """Hash loaded values, headers, row order and index, rather than a stale sidecar."""
    payload = raw_data.to_csv(index=True, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_provenance(root=None):
    """Anchor Git to this checkout, independent of the process working directory."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()

    return {"git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain"))}
