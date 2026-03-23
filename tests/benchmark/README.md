# Agent Editing Benchmarks

This folder contains lightweight benchmark cases for evaluating file-editing reliability.

## Layout

Each case lives under:

- `tests/benchmark/<feature>/<test_name>/case.json`
- `tests/benchmark/<feature>/<test_name>/fixture/*`

## Running

From the repository root:

```bash
python tests/benchmark/runner.py \
  --feature text_edits \
  --runs 10 \
  --warmup-seconds 120 \
  --report-dir tests/benchmark/reports \
  --agent-command "python {repo_root}/your_agent_driver.py --workspace {workspace} --instruction-file {instruction_file}"
```

Supported placeholders in `--agent-command`:

- `{workspace}`: temporary working copy of fixture files
- `{instruction}`: raw instruction text (shell-escaped)
- `{instruction_file}`: path to a text file containing the instruction
- `{case_dir}`: path to the benchmark case directory
- `{repo_root}`: absolute path to the repository root

## Notes

- The runner scores each run as `pass`, `soft_fail`, or `hard_fail`.
- `soft_fail` means intent assertions passed, but scope/minimal-diff checks failed.
- `hard_fail` means intended behavior failed (or command failed).
- Results include per-case rates over repeated runs for statistical benchmarking.
- The first run gets an extra timeout buffer (`--warmup-seconds`, default `120`) to absorb initial model load latency.
- A human-readable YAML report is written per case (default folder: `tests/benchmark/reports`).
- Each report includes per-run `before`, `after`, and unified `diff` for fixture files.
