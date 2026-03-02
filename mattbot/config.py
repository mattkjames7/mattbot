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
        "# MATTBOT_MAX_CONTEXT_TOKENS, MATTBOT_TEMPERATURE, MATTBOT_EMBEDDING_MODEL\n"
        "\n"
        "[agent]\n"
        f'model = "{_escape_toml_string(config.model)}"\n'
        f'ollama_url = "{_escape_toml_string(config.ollama_url)}"\n'
        f'max_context_tokens = {config.max_context_tokens}\n'
        f'temperature = {config.temperature}\n'
        f'embedding_model = "{_escape_toml_string(config.embedding_model)}"\n'
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

    return AgentConfig(
        model=str(model),
        ollama_url=str(ollama_url),
        max_context_tokens=int(max_context_tokens),
        temperature=float(temperature),
        embedding_model=str(embedding_model),
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

    return resolved
