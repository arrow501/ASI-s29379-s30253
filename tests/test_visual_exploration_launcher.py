"""Optional setup must be explicit and must fail before launching on install errors."""

import importlib.util
from types import SimpleNamespace


def load_launcher(repo_root):
    spec = importlib.util.spec_from_file_location(
        "visual_exploration_launcher", repo_root / "scripts/visual_exploration.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_optional_dependencies_do_not_install_implicitly(repo_root, monkeypatch, capsys):
    module = load_launcher(repo_root)
    monkeypatch.setattr(module, "dependency_issues", lambda: ["marimo: not installed"])
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    assert module.main(["--check"]) == 1
    assert calls == []
    assert "--setup" in capsys.readouterr().err


def test_failed_setup_does_not_launch_a_partial_install(repo_root, monkeypatch):
    module = load_launcher(repo_root)
    calls = []

    def fail_install(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr(module.subprocess, "run", fail_install)
    assert module.main(["--setup", "--check"]) == 17
    assert len(calls) == 1
    assert "-c" in calls[0] and str(repo_root / "constraints.txt") in calls[0]


def test_sample_launch_uses_current_python_and_root_from_any_cwd(repo_root, tmp_path, monkeypatch):
    module = load_launcher(repo_root)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "dependency_issues", lambda: [])
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    launches = []

    def capture_launch(command, **kwargs):
        launches.append((command, kwargs))
        return 0

    monkeypatch.setattr(module.subprocess, "call", capture_launch)
    assert module.main(["--sample"]) == 0
    command, kwargs = launches[0]
    assert command[0] == module.sys.executable
    assert kwargs["cwd"] == repo_root
    assert kwargs["env"]["ASI_DATA_MODE"] == "sample"
    assert command[command.index("--host") + 1] == "127.0.0.1"
