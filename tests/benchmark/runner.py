#!/usr/bin/env python3
"""Run repeatable benchmark cases for agent file-editing tasks."""

from __future__ import annotations

import argparse
import ast
import json
import shlex
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str


@dataclass
class RunEvaluation:
    status: str  # pass | soft_fail | hard_fail
    intent_failures: list[str]
    scope_failures: list[str]
    changed_files: list[str]
    command: CommandResult


@dataclass
class CaseSpec:
    case_path: Path
    name: str
    instruction: str
    assertions: list[dict[str, Any]]
    scope_assertions: list[dict[str, Any]]
    expected_changed_files: list[str] | None
    allowed_changed_files: list[str] | None
    max_changed_files: int | None
    timeout_seconds: int
    require_exit_code_zero: bool


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _snapshot_text_files(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(root).as_posix()
        try:
            snapshot[rel] = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Skip non-text files in this benchmark harness.
            continue
    return snapshot


def _run_command(command: str, cwd: Path, timeout_seconds: int) -> CommandResult:
    completed = subprocess.run(  # noqa: S603
        command,
        shell=True,  # noqa: S602
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
    )
    return CommandResult(
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _eval_assertion(assertion: dict[str, Any], workspace: Path, before: dict[str, str]) -> str | None:
    kind = assertion["type"]

    if kind in {"contains", "not_contains", "count_equals", "exact_text", "regex"}:
        rel = assertion["file"]
        file_path = workspace / rel
        if not file_path.exists():
            return f"Missing file for assertion: {rel}"
        content = _read_text(file_path)

        if kind == "contains":
            text = assertion["text"]
            if text not in content:
                return f"Expected {rel} to contain text: {text!r}"
            return None

        if kind == "not_contains":
            text = assertion["text"]
            if text in content:
                return f"Expected {rel} to not contain text: {text!r}"
            return None

        if kind == "count_equals":
            text = assertion["text"]
            expected = assertion["equals"]
            actual = content.count(text)
            if actual != expected:
                return f"Expected {rel} count({text!r}) == {expected}, got {actual}"
            return None

        if kind == "exact_text":
            expected = assertion["text"]
            if content != expected:
                return f"Expected exact content mismatch in {rel}"
            return None

        if kind == "regex":
            import re

            pattern = assertion["pattern"]
            if not re.search(pattern, content, flags=re.MULTILINE):
                return f"Expected regex {pattern!r} to match in {rel}"
            return None

    if kind == "unchanged":
        rel = assertion["file"]
        file_path = workspace / rel
        if not file_path.exists():
            return f"Expected unchanged file missing: {rel}"
        if rel not in before:
            return f"Cannot verify unchanged file not in fixture snapshot: {rel}"
        current = _read_text(file_path)
        if current != before[rel]:
            return f"Expected {rel} to remain unchanged"
        return None

    if kind == "python_syntax":
        rel = assertion["file"]
        file_path = workspace / rel
        if not file_path.exists():
            return f"Missing file for syntax check: {rel}"
        try:
            ast.parse(_read_text(file_path))
            return None
        except SyntaxError as exc:
            return f"Python syntax invalid in {rel}: {exc.msg}"

    return f"Unknown assertion type: {kind}"


def _load_case(case_json_path: Path) -> CaseSpec:
    payload = json.loads(case_json_path.read_text(encoding="utf-8"))
    return CaseSpec(
        case_path=case_json_path.parent,
        name=payload.get("name", case_json_path.parent.name),
        instruction=payload["instruction"],
        assertions=payload.get("assertions", []),
        scope_assertions=payload.get("scope_assertions", []),
        expected_changed_files=payload.get("expected_changed_files"),
        allowed_changed_files=payload.get("allowed_changed_files"),
        max_changed_files=payload.get("max_changed_files"),
        timeout_seconds=payload.get("timeout_seconds", 120),
        require_exit_code_zero=payload.get("require_exit_code_zero", True),
    )


def _discover_cases(root: Path, feature: str | None) -> list[CaseSpec]:
    glob_expr = "**/case.json" if feature is None else f"{feature}/**/case.json"
    case_files = sorted(root.glob(glob_expr))
    return [_load_case(path) for path in case_files]


def _evaluate_case_run(
    case: CaseSpec,
    command_template: str,
    timeout_override_seconds: int | None = None,
) -> RunEvaluation:
    fixture_root = case.case_path / "fixture"
    if not fixture_root.exists():
        raise FileNotFoundError(f"Fixture directory is required: {fixture_root}")

    with tempfile.TemporaryDirectory(prefix="mattbot-bench-") as tmp_dir:
        workspace = Path(tmp_dir) / "workspace"
        shutil.copytree(fixture_root, workspace)

        before = _snapshot_text_files(workspace)
        instruction_file = workspace / ".benchmark_instruction.txt"
        instruction_file.write_text(case.instruction, encoding="utf-8")

        repo_root = Path(__file__).resolve().parents[2]

        command = command_template.format(
            workspace=shlex.quote(str(workspace)),
            instruction=shlex.quote(case.instruction),
            instruction_file=shlex.quote(str(instruction_file)),
            case_dir=shlex.quote(str(case.case_path)),
            repo_root=shlex.quote(str(repo_root)),
        )

        effective_timeout = timeout_override_seconds or case.timeout_seconds

        try:
            cmd_result = _run_command(command, workspace, effective_timeout)
        except subprocess.TimeoutExpired:
            return RunEvaluation(
                status="hard_fail",
                intent_failures=[f"Command timed out after {effective_timeout}s"],
                scope_failures=[],
                changed_files=[],
                command=CommandResult(exit_code=124, stdout="", stderr="timeout"),
            )

        after = _snapshot_text_files(workspace)
        changed_files = sorted(
            rel
            for rel in set(before) | set(after)
            if before.get(rel, "") != after.get(rel, "")
            and rel != ".benchmark_instruction.txt"
        )

        intent_failures: list[str] = []
        scope_failures: list[str] = []

        if case.require_exit_code_zero and cmd_result.exit_code != 0:
            intent_failures.append(
                f"Command exit code was {cmd_result.exit_code}, expected 0"
            )

        for assertion in case.assertions:
            failure = _eval_assertion(assertion, workspace, before)
            if failure:
                intent_failures.append(failure)

        for assertion in case.scope_assertions:
            failure = _eval_assertion(assertion, workspace, before)
            if failure:
                scope_failures.append(failure)

        if case.expected_changed_files is not None:
            expected = sorted(case.expected_changed_files)
            if changed_files != expected:
                scope_failures.append(
                    f"Changed files mismatch. expected={expected}, actual={changed_files}"
                )

        if case.allowed_changed_files is not None:
            disallowed = [f for f in changed_files if f not in case.allowed_changed_files]
            if disallowed:
                scope_failures.append(
                    f"Unexpected changed files outside allowed set: {disallowed}"
                )

        if case.max_changed_files is not None and len(changed_files) > case.max_changed_files:
            scope_failures.append(
                f"Too many files changed. max={case.max_changed_files}, actual={len(changed_files)}"
            )

        if intent_failures:
            status = "hard_fail"
        elif scope_failures:
            status = "soft_fail"
        else:
            status = "pass"

        return RunEvaluation(
            status=status,
            intent_failures=intent_failures,
            scope_failures=scope_failures,
            changed_files=changed_files,
            command=cmd_result,
        )


def _print_case_summary(case_name: str, results: list[RunEvaluation]) -> None:
    counts = Counter(r.status for r in results)
    total = len(results)
    pass_rate = (counts.get("pass", 0) / total) * 100
    strict_rate = pass_rate
    print(
        f"[{case_name}] total={total} "
        f"pass={counts.get('pass', 0)} soft_fail={counts.get('soft_fail', 0)} hard_fail={counts.get('hard_fail', 0)} "
        f"pass_rate={pass_rate:.1f}% strict_rate={strict_rate:.1f}%"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run agent editing benchmark cases")
    parser.add_argument(
        "--benchmark-root",
        default="tests/benchmark",
        help="Root directory containing feature/test/case.json files",
    )
    parser.add_argument(
        "--feature",
        default=None,
        help="Optional feature folder to run (e.g. text_edits)",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per case")
    parser.add_argument(
        "--agent-command",
        required=True,
        help=(
            "Shell command template used to run your agent. "
            "Supports placeholders: {workspace}, {instruction}, {instruction_file}, {case_dir}, {repo_root}."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at first hard fail",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=int,
        default=120,
        help=(
            "Extra timeout buffer added only to the very first run, "
            "useful for initial model loading (default: 120)."
        ),
    )

    args = parser.parse_args()
    benchmark_root = Path(args.benchmark_root)

    if not benchmark_root.exists():
        raise SystemExit(f"Benchmark root not found: {benchmark_root}")

    cases = _discover_cases(benchmark_root, args.feature)
    if not cases:
        raise SystemExit("No benchmark cases discovered.")

    print(f"Discovered {len(cases)} case(s). Running {args.runs} run(s) per case.\n")

    overall: Counter[str] = Counter()
    per_case: dict[str, list[RunEvaluation]] = {}

    is_first_run = True

    for case in cases:
        case_results: list[RunEvaluation] = []
        print(f"Running case: {case.name}")

        for run_index in range(1, args.runs + 1):
            timeout_override = None
            if is_first_run and args.warmup_seconds > 0:
                timeout_override = case.timeout_seconds + args.warmup_seconds
                print(
                    f"  applying first-run warmup buffer: +{args.warmup_seconds}s "
                    f"(timeout={timeout_override}s)"
                )

            evaluation = _evaluate_case_run(
                case,
                args.agent_command,
                timeout_override_seconds=timeout_override,
            )
            is_first_run = False
            case_results.append(evaluation)
            overall[evaluation.status] += 1

            if evaluation.status == "pass":
                print(f"  run {run_index}: pass")
            else:
                print(f"  run {run_index}: {evaluation.status}")
                for item in evaluation.intent_failures:
                    print(f"    intent: {item}")
                for item in evaluation.scope_failures:
                    print(f"    scope: {item}")

            if args.fail_fast and evaluation.status == "hard_fail":
                break

        per_case[case.name] = case_results
        _print_case_summary(case.name, case_results)
        print()

        if args.fail_fast and case_results and case_results[-1].status == "hard_fail":
            break

    total_runs = sum(len(v) for v in per_case.values())
    print("Overall summary")
    print(
        f"total_runs={total_runs} pass={overall.get('pass', 0)} "
        f"soft_fail={overall.get('soft_fail', 0)} hard_fail={overall.get('hard_fail', 0)}"
    )

    return 0 if overall.get("hard_fail", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
