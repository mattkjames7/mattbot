"""Persistent artifact storage for large tool outputs."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class ArtifactStore:
    """Store large tool outputs on disk and return lightweight references."""

    def __init__(
        self,
        enabled: bool = True,
        artifact_dir: str = "~/.mattbot/artifacts",
        ttl_days: int = 7,
        max_sessions: int = 20,
    ) -> None:
        self.enabled = enabled
        self.artifact_dir = Path(artifact_dir).expanduser()
        self.ttl_days = max(0, int(ttl_days))
        self.max_sessions = max(1, int(max_sessions))
        self.session_id: str | None = None
        self.session_dir: Path | None = None

        if self.enabled:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            self.cleanup()
            self.start_new_session()

    def start_new_session(self) -> str | None:
        """Start a new artifact session directory."""
        if not self.enabled:
            return None

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.session_id = f"session-{timestamp}-{os.getpid()}"
        self.session_dir = self.artifact_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        return self.session_id

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def put(self, tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
        """Persist one tool result and return artifact metadata."""
        if not self.enabled:
            return {"stored": False, "reason": "disabled"}

        if not self.session_dir:
            self.start_new_session()

        payload = json.dumps(result, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        stamp = datetime.now(timezone.utc).strftime("%H%M%S%f")
        artifact_id = f"{tool_name}_{stamp}_{digest}"

        artifact_record = {
            "artifact_id": artifact_id,
            "tool_name": tool_name,
            "created_at": self._now_iso(),
            "result": result,
        }

        artifact_path = self.session_dir / f"{artifact_id}.json"
        artifact_path.write_text(json.dumps(artifact_record, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "stored": True,
            "artifact_id": artifact_id,
            "session_id": self.session_id,
            "path": str(artifact_path),
            "size_chars": len(payload),
            "size_bytes": artifact_path.stat().st_size,
        }

    def cleanup(self) -> None:
        """Delete old artifact sessions using TTL and max session policies."""
        if not self.enabled or not self.artifact_dir.exists():
            return

        now = datetime.now(timezone.utc)
        session_dirs = [p for p in self.artifact_dir.iterdir() if p.is_dir()]

        # TTL pruning first
        if self.ttl_days > 0:
            cutoff = now - timedelta(days=self.ttl_days)
            for session in session_dirs:
                mtime = datetime.fromtimestamp(session.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff:
                    self._remove_dir(session)

        # Max-session pruning (keep newest)
        session_dirs = [p for p in self.artifact_dir.iterdir() if p.is_dir()]
        session_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for session in session_dirs[self.max_sessions :]:
            self._remove_dir(session)

    def _remove_dir(self, path: Path) -> None:
        for child in path.rglob("*"):
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()
        path.rmdir()
