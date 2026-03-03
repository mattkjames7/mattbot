"""Tests for filesystem-backed artifact store."""

from pathlib import Path

from mattbot.artifact_store import ArtifactStore


def test_artifact_store_put_persists_json(tmp_path):
    store = ArtifactStore(enabled=True, artifact_dir=str(tmp_path), ttl_days=7, max_sessions=20)

    saved = store.put("run_bash_command", {"success": True, "stdout": "ok"})

    assert saved["stored"] is True
    assert saved["artifact_id"]
    artifact_path = Path(saved["path"])
    assert artifact_path.exists()
    content = artifact_path.read_text(encoding="utf-8")
    assert "run_bash_command" in content
    assert "stdout" in content


def test_artifact_store_disabled_returns_not_stored(tmp_path):
    store = ArtifactStore(enabled=False, artifact_dir=str(tmp_path))

    saved = store.put("read_file", {"success": True, "content": "demo"})

    assert saved["stored"] is False
    assert saved["reason"] == "disabled"
