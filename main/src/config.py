"""Configuration helpers for nested YAML and command-line overrides."""
import argparse
from pathlib import Path
from typing import Any, Optional

import yaml


def load_config(config_path: Optional[Path] = None) -> tuple[dict[str, Any], Path]:
    """Load the YAML config and return it with the main package directory."""
    main_dir = Path(__file__).resolve().parent.parent
    if config_path is None:
        config_path = main_dir / "configs" / "default.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return (config or {}), main_dir


def get_section(config: dict[str, Any], section_name: str) -> dict[str, Any]:
    """Return a required nested config section."""
    section = config.get(section_name)
    if not isinstance(section, dict):
        raise KeyError(f"Missing required config section: {section_name}")
    return section


def convert_section_paths(
    config: dict[str, Any],
    section_name: str,
    path_keys: list[str],
    base_dir: Path,
) -> dict[str, Any]:
    """Resolve configured paths inside one nested section against base_dir."""
    section = get_section(config, section_name)
    for key in path_keys:
        if key in section and section[key] is not None:
            path_value = Path(section[key])
            if not path_value.is_absolute():
                section[key] = (base_dir / path_value).resolve()
            else:
                section[key] = path_value.resolve()
    return config


def merge_config_with_args(
    config: dict[str, Any],
    args: argparse.Namespace,
    arg_targets: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    """Merge non-None command-line arguments into nested config targets."""
    for key, value in vars(args).items():
        if value is not None:
            section_name, section_key = arg_targets[key]
            get_section(config, section_name)[section_key] = value
    return config
