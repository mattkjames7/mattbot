"""Configuration management for MattBot."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib


@dataclass(slots=True)
class AgentConfig:
    """Resolved MattBot configuration values."""

    model: str = "gpt-oss:latest"
    ollama_url: str = "http://localhost:11434"
    max_context_tokens: int = 8192
    temperature: float = 0.7
    embedding_model: str = "all-minilm:l6-v2"
    history_length: int = 1000
    logging_enabled: bool = False
    log_dir: str = "~/.mattbot/logs"
    artifact_store_enabled: bool = True
    artifact_dir: str = "~/.mattbot/artifacts"
    artifact_ttl_days: int = 7
    artifact_max_sessions: int = 20
    artifact_inline_char_limit: int = 8000


def _parse_bool(value: Any, default: bool = False) -> bool:
    """Parse bool-like values from TOML/env/CLI inputs."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def get_config_path(config_path: str | os.PathLike[str] | None = None) -> Path:
    """Return the path to the MattBot config file."""
    if config_path:
        return Path(config_path).expanduser()

    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    base_dir = Path(xdg_config_home).expanduser() if xdg_config_home else Path.home() / ".config"
    return base_dir / "mattbot" / "config.toml"


def _escape_toml_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def write_config(config: AgentConfig, config_path: str | os.PathLike[str] | None = None) -> Path:
    """Write config values to disk in TOML format."""
    path = get_config_path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    content = (
        "# MattBot configuration\n"
        "# Values can also be overridden with env vars: MATTBOT_MODEL, MATTBOT_OLLAMA_URL,\n"
        "# MATTBOT_MAX_CONTEXT_TOKENS, MATTBOT_TEMPERATURE, MATTBOT_EMBEDDING_MODEL,\n"
        "# MATTBOT_HISTORY_LENGTH, MATTBOT_LOGGING, MATTBOT_LOG_DIR,\n"
        "# MATTBOT_ARTIFACT_STORE_ENABLED, MATTBOT_ARTIFACT_DIR, MATTBOT_ARTIFACT_TTL_DAYS,\n"
        "# MATTBOT_ARTIFACT_MAX_SESSIONS, MATTBOT_ARTIFACT_INLINE_CHAR_LIMIT\n"
        "\n"
        "[agent]\n"
        f'model = "{_escape_toml_string(config.model)}"\n'
        f'ollama_url = "{_escape_toml_string(config.ollama_url)}"\n'
        f'max_context_tokens = {config.max_context_tokens}\n'
        f'temperature = {config.temperature}\n'
        f'embedding_model = "{_escape_toml_string(config.embedding_model)}"\n'
        f'history_length = {config.history_length}\n'
        f'logging_enabled = {str(config.logging_enabled).lower()}\n'
        f'log_dir = "{_escape_toml_string(config.log_dir)}"\n'
        f'artifact_store_enabled = {str(config.artifact_store_enabled).lower()}\n'
        f'artifact_dir = "{_escape_toml_string(config.artifact_dir)}"\n'
        f'artifact_ttl_days = {config.artifact_ttl_days}\n'
        f'artifact_max_sessions = {config.artifact_max_sessions}\n'
        f'artifact_inline_char_limit = {config.artifact_inline_char_limit}\n'
    )

    path.write_text(content, encoding="utf-8")
    return path


def ensure_config_file(config_path: str | os.PathLike[str] | None = None) -> Path:
    """Create a default config file if it does not exist."""
    path = get_config_path(config_path)
    if not path.exists():
        write_config(AgentConfig(), path)
    return path


def read_config(config_path: str | os.PathLike[str] | None = None) -> AgentConfig:
    """Read config from disk, falling back to defaults when missing/invalid."""
    defaults = AgentConfig()
    path = get_config_path(config_path)

    if not path.exists():
        return defaults

    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return defaults

    agent = data.get("agent", {}) if isinstance(data, dict) else {}

    model = agent.get("model", defaults.model) if isinstance(agent, dict) else defaults.model
    ollama_url = agent.get("ollama_url", defaults.ollama_url) if isinstance(agent, dict) else defaults.ollama_url
    max_context_tokens = agent.get("max_context_tokens", defaults.max_context_tokens) if isinstance(agent, dict) else defaults.max_context_tokens
    temperature = agent.get("temperature", defaults.temperature) if isinstance(agent, dict) else defaults.temperature
    embedding_model = agent.get("embedding_model", defaults.embedding_model) if isinstance(agent, dict) else defaults.embedding_model
    history_length = agent.get("history_length", defaults.history_length) if isinstance(agent, dict) else defaults.history_length
    logging_enabled = agent.get("logging_enabled", defaults.logging_enabled) if isinstance(agent, dict) else defaults.logging_enabled
    log_dir = agent.get("log_dir", defaults.log_dir) if isinstance(agent, dict) else defaults.log_dir
    artifact_store_enabled = agent.get("artifact_store_enabled", defaults.artifact_store_enabled) if isinstance(agent, dict) else defaults.artifact_store_enabled
    artifact_dir = agent.get("artifact_dir", defaults.artifact_dir) if isinstance(agent, dict) else defaults.artifact_dir
    artifact_ttl_days = agent.get("artifact_ttl_days", defaults.artifact_ttl_days) if isinstance(agent, dict) else defaults.artifact_ttl_days
    artifact_max_sessions = agent.get("artifact_max_sessions", defaults.artifact_max_sessions) if isinstance(agent, dict) else defaults.artifact_max_sessions
    artifact_inline_char_limit = agent.get("artifact_inline_char_limit", defaults.artifact_inline_char_limit) if isinstance(agent, dict) else defaults.artifact_inline_char_limit

    return AgentConfig(
        model=str(model),
        ollama_url=str(ollama_url),
        max_context_tokens=int(max_context_tokens),
        temperature=float(temperature),
        embedding_model=str(embedding_model),
        history_length=int(history_length),
        logging_enabled=_parse_bool(logging_enabled, default=defaults.logging_enabled),
        log_dir=str(log_dir),
        artifact_store_enabled=_parse_bool(artifact_store_enabled, default=defaults.artifact_store_enabled),
        artifact_dir=str(artifact_dir),
        artifact_ttl_days=int(artifact_ttl_days),
        artifact_max_sessions=int(artifact_max_sessions),
        artifact_inline_char_limit=int(artifact_inline_char_limit),
    )


def resolve_config(
    cli_args: Any | None = None,
    config_path: str | os.PathLike[str] | None = None,
    create_if_missing: bool = True,
) -> AgentConfig:
    """Resolve final config with precedence: CLI > env > file > defaults."""
    if create_if_missing:
        ensure_config_file(config_path)

    resolved = read_config(config_path)

    env_model = os.environ.get("MATTBOT_MODEL")
    if env_model:
        resolved.model = env_model

    env_ollama_url = os.environ.get("MATTBOT_OLLAMA_URL")
    if env_ollama_url:
        resolved.ollama_url = env_ollama_url

    env_max_context = os.environ.get("MATTBOT_MAX_CONTEXT_TOKENS")
    if env_max_context:
        resolved.max_context_tokens = int(env_max_context)

    env_temperature = os.environ.get("MATTBOT_TEMPERATURE")
    if env_temperature:
        resolved.temperature = float(env_temperature)

    env_embedding_model = os.environ.get("MATTBOT_EMBEDDING_MODEL")
    if env_embedding_model:
        resolved.embedding_model = env_embedding_model

    env_history_length = os.environ.get("MATTBOT_HISTORY_LENGTH")
    if env_history_length:
        resolved.history_length = int(env_history_length)

    env_logging_enabled = os.environ.get("MATTBOT_LOGGING")
    if env_logging_enabled is not None:
        resolved.logging_enabled = _parse_bool(env_logging_enabled, default=resolved.logging_enabled)

    env_log_dir = os.environ.get("MATTBOT_LOG_DIR")
    if env_log_dir:
        resolved.log_dir = env_log_dir

    env_artifact_store_enabled = os.environ.get("MATTBOT_ARTIFACT_STORE_ENABLED")
    if env_artifact_store_enabled is not None:
        resolved.artifact_store_enabled = _parse_bool(
            env_artifact_store_enabled,
            default=resolved.artifact_store_enabled,
        )

    env_artifact_dir = os.environ.get("MATTBOT_ARTIFACT_DIR")
    if env_artifact_dir:
        resolved.artifact_dir = env_artifact_dir

    env_artifact_ttl_days = os.environ.get("MATTBOT_ARTIFACT_TTL_DAYS")
    if env_artifact_ttl_days:
        resolved.artifact_ttl_days = int(env_artifact_ttl_days)

    env_artifact_max_sessions = os.environ.get("MATTBOT_ARTIFACT_MAX_SESSIONS")
    if env_artifact_max_sessions:
        resolved.artifact_max_sessions = int(env_artifact_max_sessions)

    env_artifact_inline_char_limit = os.environ.get("MATTBOT_ARTIFACT_INLINE_CHAR_LIMIT")
    if env_artifact_inline_char_limit:
        resolved.artifact_inline_char_limit = int(env_artifact_inline_char_limit)

    if cli_args is not None:
        args_dict = vars(cli_args) if hasattr(cli_args, "__dict__") else dict(cli_args)

        cli_model = args_dict.get("model")
        if cli_model:
            resolved.model = str(cli_model)

        cli_ollama_url = args_dict.get("ollama_url")
        if cli_ollama_url:
            resolved.ollama_url = str(cli_ollama_url)

        cli_max_context = args_dict.get("max_context_tokens")
        if cli_max_context:
            resolved.max_context_tokens = int(cli_max_context)

        cli_temperature = args_dict.get("temperature")
        if cli_temperature:
            resolved.temperature = float(cli_temperature)

        cli_embedding_model = args_dict.get("embedding_model")
        if cli_embedding_model:
            resolved.embedding_model = str(cli_embedding_model)

        cli_history_length = args_dict.get("history_length")
        if cli_history_length:
            resolved.history_length = int(cli_history_length)

        if "logging_enabled" in args_dict and args_dict.get("logging_enabled") is not None:
            resolved.logging_enabled = _parse_bool(args_dict.get("logging_enabled"), default=resolved.logging_enabled)

        cli_log_dir = args_dict.get("log_dir")
        if cli_log_dir:
            resolved.log_dir = str(cli_log_dir)

        if "artifact_store_enabled" in args_dict and args_dict.get("artifact_store_enabled") is not None:
            resolved.artifact_store_enabled = _parse_bool(
                args_dict.get("artifact_store_enabled"),
                default=resolved.artifact_store_enabled,
            )

        cli_artifact_dir = args_dict.get("artifact_dir")
        if cli_artifact_dir:
            resolved.artifact_dir = str(cli_artifact_dir)

        cli_artifact_ttl_days = args_dict.get("artifact_ttl_days")
        if cli_artifact_ttl_days:
            resolved.artifact_ttl_days = int(cli_artifact_ttl_days)

        cli_artifact_max_sessions = args_dict.get("artifact_max_sessions")
        if cli_artifact_max_sessions:
            resolved.artifact_max_sessions = int(cli_artifact_max_sessions)

        cli_artifact_inline_char_limit = args_dict.get("artifact_inline_char_limit")
        if cli_artifact_inline_char_limit:
            resolved.artifact_inline_char_limit = int(cli_artifact_inline_char_limit)

    return resolved
