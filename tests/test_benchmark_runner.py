"""Tests for benchmark runner helpers."""

from pathlib import Path

from tests.benchmark.runner import (
    CaseSpec,
    CommandResult,
    RunEvaluation,
    _build_file_reports,
    _eval_assertion,
    _render_case_report_yaml,
    _snapshot_text_files,
)


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


def test_build_file_reports_includes_unified_diff():
    before = {
        "a.txt": "alpha\n",
        "b.txt": "unchanged\n",
    }
    after = {
        "a.txt": "beta\n",
        "b.txt": "unchanged\n",
    }

    reports = _build_file_reports(before, after)
    by_path = {report.path: report for report in reports}

    assert by_path["a.txt"].changed is True
    assert "-alpha" in by_path["a.txt"].diff
    assert "+beta" in by_path["a.txt"].diff
    assert by_path["b.txt"].changed is False
    assert by_path["b.txt"].diff == ""


def test_render_case_report_yaml_contains_before_after_and_diff():
    case = CaseSpec(
        case_path=Path("tests/benchmark/text_edits/sample_case"),
        name="sample_case",
        instruction="Update file",
        assertions=[],
        scope_assertions=[],
        expected_changed_files=None,
        allowed_changed_files=None,
        max_changed_files=None,
        timeout_seconds=10,
        require_exit_code_zero=True,
    )

    file_reports = _build_file_reports({"file.txt": "old\n"}, {"file.txt": "new\n"})
    evaluation = RunEvaluation(
        status="pass",
        intent_failures=[],
        scope_failures=[],
        changed_files=["file.txt"],
        file_reports=file_reports,
        command=CommandResult(exit_code=0, stdout="ok\n", stderr=""),
    )

    content = _render_case_report_yaml(case, [evaluation])

    assert "runs:" in content
    assert "before:" in content
    assert "after:" in content
    assert "diff:" in content
    assert "-old" in content
    assert "+new" in content
