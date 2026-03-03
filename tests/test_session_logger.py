"""Tests for session logging."""

import json

from mattbot.session_logger import SessionLogger


def test_session_logger_disabled(tmp_path):
    """Disabled logger should not create files."""
    logger = SessionLogger(enabled=False, log_dir=str(tmp_path))
    logger.log_message("user", "hello")
    logger.finalize()

    assert logger.file_path is None
    assert list(tmp_path.glob("*.jsonl")) == []


def test_session_logger_writes_events_and_summary(tmp_path):
    """Enabled logger should write a single JSONL session log."""
    logger = SessionLogger(enabled=True, log_dir=str(tmp_path), session_metadata={"model": "test"})

    logger.log_message("user", "hello")
    logger.log_tool_call("write_file", {"file_path": "demo.txt"}, tool_call_id="1")
    logger.log_tool_result("write_file", {"success": True, "file_path": "demo.txt"}, tool_call_id="1")
    logger.finalize(reason="done")

    assert logger.file_path is not None
    assert logger.file_path.exists()

    lines = logger.file_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 4

    events = [json.loads(line) for line in lines]
    assert events[0]["event"] == "session_start"
    assert events[-1]["event"] == "session_end"
    assert events[-1]["reason"] == "done"
    assert events[-1]["changed_files"] == ["demo.txt"]
