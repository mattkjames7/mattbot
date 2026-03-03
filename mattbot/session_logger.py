"""Session logging for MattBot conversations and tool activity."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SessionLogger:
    """Writes one JSONL log file per MattBot session."""

    def __init__(
        self,
        enabled: bool = False,
        log_dir: str = "~/.mattbot/logs",
        session_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.enabled = enabled
        self.log_dir = Path(log_dir).expanduser()
        self.session_metadata = session_metadata or {}
        self.changed_files: set[str] = set()
        self.file_path: Path | None = None
        self._closed = False

        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.file_path = self.log_dir / f"session-{timestamp}-{os.getpid()}.jsonl"
            self.log_event("session_start", metadata=self.session_metadata)

    @staticmethod
    def _iso_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def log_event(self, event_type: str, **payload: Any) -> None:
        """Write a single event line to the session log."""
        if not self.enabled or not self.file_path or self._closed:
            return

        entry = {
            "timestamp": self._iso_now(),
            "event": event_type,
            **payload,
        }

        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_message(self, role: str, content: str, **extra: Any) -> None:
        """Log a user/assistant/tool message event."""
        self.log_event(
            "message",
            role=role,
            content=content,
            **extra,
        )

    def log_tool_call(self, tool_name: str, arguments: Any, tool_call_id: str | None = None) -> None:
        """Log invocation of a tool."""
        self.log_event(
            "tool_call",
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
        )

    def log_tool_result(self, tool_name: str, result: dict[str, Any], tool_call_id: str | None = None) -> None:
        """Log tool result and track changed files when detectable."""
        self.log_event(
            "tool_result",
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            result=result,
        )

        if tool_name in {"write_file", "edit_file"} and result.get("success") and result.get("file_path"):
            self.changed_files.add(str(result["file_path"]))

    def finalize(self, reason: str = "session_end") -> None:
        """Finalize session log with summary metadata."""
        if not self.enabled or self._closed:
            return

        self.log_event(
            "session_end",
            reason=reason,
            changed_files=sorted(self.changed_files),
        )
        self._closed = True
