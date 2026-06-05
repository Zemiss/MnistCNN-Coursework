"""配置管理模块 - 支持 YAML 配置文件和命令行参数覆盖"""
import argparse
from pathlib import Path
from typing import Any, Optional

import yaml


def load_config(config_path: Optional[Path] = None) -> tuple[dict[str, Any], Path]:
    """从 YAML 文件加载配置，返回 (配置字典, main_dir)"""
    if config_path is None:
        # 默认在 main/configs/default.yaml
        main_dir = Path(__file__).resolve().parent.parent
        config_path = main_dir / "configs" / "default.yaml"
    else:
        main_dir = Path(__file__).resolve().parent.parent
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return (config or {}), main_dir


def convert_paths(config: dict[str, Any], path_keys: list[str], base_dir: Path) -> dict[str, Any]:
    """
    将指定的配置键转换为相对于 base_dir 的绝对路径，并规范化
    
    Args:
        config: 配置字典
        path_keys: 需要转换的路径键列表
        base_dir: 基础目录（通常是 main_dir）
    """
    for key in path_keys:
        if key in config and config[key] is not None:
            path_value = Path(config[key])
            # 如果是相对路径，则相对于 base_dir；如果是绝对路径，保持不变
            if not path_value.is_absolute():
                config[key] = (base_dir / path_value).resolve()
            else:
                config[key] = path_value.resolve()
    return config


def merge_config_with_args(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """将命令行参数合并到配置中，命令行参数优先"""
    for key, value in vars(args).items():
        if value is not None:
            # 只覆盖命令行中明确指定的参数
            config[key] = value
    return config
