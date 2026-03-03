"""Tests for config management."""

from argparse import Namespace

from mattbot.config import AgentConfig, read_config, resolve_config, write_config


def test_read_missing_config_returns_defaults(tmp_path):
    """Reading a missing config should return defaults."""
    config_path = tmp_path / "missing.toml"

    config = read_config(config_path)

    assert config.model == "gpt-oss:latest"
    assert config.ollama_url == "http://localhost:11434"
    assert config.max_context_tokens == 8192
    assert config.temperature == 0.7
    assert config.embedding_model == "all-minilm:l6-v2"
    assert config.history_length == 1000
    assert config.logging_enabled is False
    assert config.log_dir == "~/.mattbot/logs"
    assert config.artifact_store_enabled is True
    assert config.artifact_dir == "~/.mattbot/artifacts"
    assert config.artifact_ttl_days == 7
    assert config.artifact_max_sessions == 20
    assert config.artifact_inline_char_limit == 8000


def test_write_and_read_round_trip(tmp_path):
    """Config written to disk should read back unchanged."""
    config_path = tmp_path / "config.toml"
    expected = AgentConfig(
        model="llama3.2",
        ollama_url="http://localhost:11434",
        max_context_tokens=16384,
        temperature=0.9,
        embedding_model="nomic-embed-text",
        history_length=250,
        logging_enabled=True,
        log_dir="~/custom-mattbot-logs",
        artifact_store_enabled=True,
        artifact_dir="~/custom-artifacts",
        artifact_ttl_days=14,
        artifact_max_sessions=33,
        artifact_inline_char_limit=9000,
    )

    write_config(expected, config_path)
    actual = read_config(config_path)

    assert actual == expected


def test_resolve_config_cli_overrides_file_and_env(tmp_path, monkeypatch):
    """CLI args should have highest precedence."""
    config_path = tmp_path / "config.toml"
    write_config(
        AgentConfig(
            model="from-file",
            ollama_url="http://file:11434",
            max_context_tokens=4096,
            temperature=0.5,
            embedding_model="model-file",
            history_length=100,
            logging_enabled=False,
            log_dir="~/file-logs",
            artifact_store_enabled=False,
            artifact_dir="~/file-artifacts",
            artifact_ttl_days=3,
            artifact_max_sessions=9,
            artifact_inline_char_limit=1111,
        ),
        config_path,
    )

    monkeypatch.setenv("MATTBOT_MODEL", "from-env")
    monkeypatch.setenv("MATTBOT_OLLAMA_URL", "http://env:11434")
    monkeypatch.setenv("MATTBOT_MAX_CONTEXT_TOKENS", "6000")
    monkeypatch.setenv("MATTBOT_TEMPERATURE", "0.6")
    monkeypatch.setenv("MATTBOT_EMBEDDING_MODEL", "model-env")
    monkeypatch.setenv("MATTBOT_HISTORY_LENGTH", "400")
    monkeypatch.setenv("MATTBOT_LOGGING", "true")
    monkeypatch.setenv("MATTBOT_LOG_DIR", "~/env-logs")
    monkeypatch.setenv("MATTBOT_ARTIFACT_STORE_ENABLED", "true")
    monkeypatch.setenv("MATTBOT_ARTIFACT_DIR", "~/env-artifacts")
    monkeypatch.setenv("MATTBOT_ARTIFACT_TTL_DAYS", "8")
    monkeypatch.setenv("MATTBOT_ARTIFACT_MAX_SESSIONS", "44")
    monkeypatch.setenv("MATTBOT_ARTIFACT_INLINE_CHAR_LIMIT", "4321")

    args = Namespace(
        model="from-cli",
        ollama_url="http://cli:11434",
        max_context_tokens=8000,
        temperature=0.8,
        embedding_model="model-cli",
        history_length=300,
        logging_enabled=False,
        log_dir="~/cli-logs",
        artifact_store_enabled=False,
        artifact_dir="~/cli-artifacts",
        artifact_ttl_days=12,
        artifact_max_sessions=99,
        artifact_inline_char_limit=2222,
    )
    resolved = resolve_config(args, config_path=config_path, create_if_missing=False)

    assert resolved.model == "from-cli"
    assert resolved.ollama_url == "http://cli:11434"
    assert resolved.max_context_tokens == 8000
    assert resolved.temperature == 0.8
    assert resolved.embedding_model == "model-cli"
    assert resolved.history_length == 300
    assert resolved.logging_enabled is False
    assert resolved.log_dir == "~/cli-logs"
    assert resolved.artifact_store_enabled is False
    assert resolved.artifact_dir == "~/cli-artifacts"
    assert resolved.artifact_ttl_days == 12
    assert resolved.artifact_max_sessions == 99
    assert resolved.artifact_inline_char_limit == 2222


def test_resolve_config_uses_env_when_cli_missing(tmp_path, monkeypatch):
    """Env vars should override file when CLI args are None."""
    config_path = tmp_path / "config.toml"
    write_config(
        AgentConfig(
            model="from-file",
            ollama_url="http://file:11434",
            max_context_tokens=4096,
            temperature=0.5,
            embedding_model="model-file",
            history_length=100,
            logging_enabled=False,
            log_dir="~/file-logs",
            artifact_store_enabled=False,
            artifact_dir="~/file-artifacts",
            artifact_ttl_days=3,
            artifact_max_sessions=9,
            artifact_inline_char_limit=1111,
        ),
        config_path,
    )

    monkeypatch.setenv("MATTBOT_MODEL", "from-env")
    monkeypatch.setenv("MATTBOT_MAX_CONTEXT_TOKENS", "6000")
    monkeypatch.setenv("MATTBOT_HISTORY_LENGTH", "400")
    monkeypatch.setenv("MATTBOT_LOGGING", "yes")
    monkeypatch.setenv("MATTBOT_LOG_DIR", "~/env-logs")
    monkeypatch.setenv("MATTBOT_ARTIFACT_STORE_ENABLED", "yes")
    monkeypatch.setenv("MATTBOT_ARTIFACT_DIR", "~/env-artifacts")
    monkeypatch.setenv("MATTBOT_ARTIFACT_TTL_DAYS", "15")
    monkeypatch.setenv("MATTBOT_ARTIFACT_MAX_SESSIONS", "77")
    monkeypatch.setenv("MATTBOT_ARTIFACT_INLINE_CHAR_LIMIT", "3333")

    args = Namespace(
        model=None,
        ollama_url=None,
        max_context_tokens=None,
        temperature=None,
        embedding_model=None,
        history_length=None,
        logging_enabled=None,
        log_dir=None,
        artifact_store_enabled=None,
        artifact_dir=None,
        artifact_ttl_days=None,
        artifact_max_sessions=None,
        artifact_inline_char_limit=None,
    )
    resolved = resolve_config(args, config_path=config_path, create_if_missing=False)

    assert resolved.model == "from-env"
    assert resolved.ollama_url == "http://file:11434"
    assert resolved.max_context_tokens == 6000
    assert resolved.temperature == 0.5
    assert resolved.embedding_model == "model-file"
    assert resolved.history_length == 400
    assert resolved.logging_enabled is True
    assert resolved.log_dir == "~/env-logs"
    assert resolved.artifact_store_enabled is True
    assert resolved.artifact_dir == "~/env-artifacts"
    assert resolved.artifact_ttl_days == 15
    assert resolved.artifact_max_sessions == 77
    assert resolved.artifact_inline_char_limit == 3333
