#!/usr/bin/env python3
"""A tiny deterministic editor for smoke-testing benchmark cases.

This is not an LLM agent. It only supports the included sample cases.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--instruction-file", required=True)
    args = parser.parse_args()

    workspace = Path(args.workspace)
    instruction = Path(args.instruction_file).read_text(encoding="utf-8")

    if "Replace the word 'colour' with 'color'" in instruction:
        path = workspace / "document.txt"
        _write(path, _read(path).replace("colour", "color"))
        return 0

    if "remove the deprecated block" in instruction:
        path = workspace / "config.ini"
        text = _read(path)
        old = "# BEGIN_DEPRECATED\nlegacy_mode=true\nlegacy_timeout=30\n# END_DEPRECATED\n\n"
        _write(path, text.replace(old, ""))
        return 0

    if "replace the DATABASE_URL line" in instruction:
        path = workspace / "app.env"
        lines = []
        for line in _read(path).splitlines():
            if line.startswith("DATABASE_URL="):
                lines.append("DATABASE_URL=postgresql://localhost/newdb")
            elif line.startswith("CACHE_TTL="):
                lines.append("CACHE_TTL=900")
            else:
                lines.append(line)
        _write(path, "\n".join(lines) + "\n")
        return 0

    if "immediately before '# END CONFIG'" in instruction:
        path = workspace / "config.ini"
        text = _read(path)
        marker = "# END CONFIG"
        if "enabled=true" not in text:
            text = text.replace(marker, "enabled=true\n" + marker)
            _write(path, text)
        return 0

    if "Fix spelling mistakes in comments only" in instruction:
        path = workspace / "module.py"
        text = _read(path)
        text = text.replace(
            "# This funtion calcualtes a responce",
            "# This function calculates a response",
        )
        _write(path, text)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
