"""One-command optional marimo setup and launch using the selected Python environment."""

import argparse
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements/visual-exploration.txt"


def dependency_issues():
    issues = []
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        package, expected = line.strip().split("==")
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            issues.append(f"{package}=={expected}: not installed")
        else:
            if installed != expected:
                issues.append(f"{package}=={expected}: installed version is {installed}")
    return issues


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--setup", action="store_true", help="Install pinned optional packages first"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check installation without starting a server"
    )
    parser.add_argument(
        "--sample", action="store_true", help="Start on committed sample data; no download"
    )
    args = parser.parse_args(argv)
    if sys.version_info[:2] != (3, 11):
        print("Use the project's Python 3.11 environment: conda activate asi-ml", file=sys.stderr)
        return 1
    if args.setup:
        commands = [
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(REQUIREMENTS),
                "-c",
                str(ROOT / "constraints.txt"),
            ],
            [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(ROOT)],
        ]
        for command in commands:
            result = subprocess.run(command, cwd=ROOT, check=False)
            if result.returncode:
                return result.returncode
    issues = dependency_issues()
    if issues:
        print("\n".join(issues), file=sys.stderr)
        print("Run in asi-ml with --setup to install this optional tool.", file=sys.stderr)
        return 1
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import marimo, plotly, anywidget, asi_project.exploration, asi_project.pca_widget",
        ],
        cwd=ROOT,
        check=False,
    )
    if check.returncode:
        print("Project imports failed. Rerun this command with --setup.", file=sys.stderr)
        return check.returncode
    print(f"Visual exploration ready: {sys.executable}", flush=True)
    if args.check:
        return 0
    env = os.environ.copy()
    if args.sample:
        env["ASI_DATA_MODE"] = "sample"
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "marimo",
            "edit",
            str(ROOT / "notebooks/03_visual_exploration.py"),
            "--no-sandbox",
            "--no-token",
            "--host",
            "127.0.0.1",
            "--skip-update-check",
        ],
        cwd=ROOT,
        env=env,
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
