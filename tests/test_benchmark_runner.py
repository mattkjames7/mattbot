"""Tests for benchmark runner helpers."""

from tests.benchmark.runner import _eval_assertion, _snapshot_text_files


def test_snapshot_text_files(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_text("hello\n", encoding="utf-8")
    (root / "b.txt").write_text("world\n", encoding="utf-8")

    snapshot = _snapshot_text_files(root)

    assert snapshot["a.txt"] == "hello\n"
    assert snapshot["b.txt"] == "world\n"


def test_eval_assertion_contains_and_unchanged(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "file.txt"
    target.write_text("alpha beta\n", encoding="utf-8")

    before = {"file.txt": "alpha beta\n"}

    assert _eval_assertion({"type": "contains", "file": "file.txt", "text": "alpha"}, workspace, before) is None
    assert _eval_assertion({"type": "unchanged", "file": "file.txt"}, workspace, before) is None

    target.write_text("changed\n", encoding="utf-8")
    failure = _eval_assertion({"type": "unchanged", "file": "file.txt"}, workspace, before)
    assert failure is not None


def test_eval_assertion_python_syntax(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mod = workspace / "module.py"
    mod.write_text("def ok():\n    return 1\n", encoding="utf-8")

    before = {"module.py": mod.read_text(encoding="utf-8")}
    assert _eval_assertion({"type": "python_syntax", "file": "module.py"}, workspace, before) is None

    mod.write_text("def bad(:\n", encoding="utf-8")
    failure = _eval_assertion({"type": "python_syntax", "file": "module.py"}, workspace, before)
    assert failure is not None
