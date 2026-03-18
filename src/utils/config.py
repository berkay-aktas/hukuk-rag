"""Project configuration loader."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load YAML config and resolve paths for current environment.

    Args:
        path: Path to config YAML. Defaults to configs/config.yaml.

    Returns:
        Parsed config dictionary.
    """
    config_path = path or _DEFAULT_CONFIG
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["_is_colab"] = _is_colab()
    config["_project_root"] = str(_PROJECT_ROOT)

    if config["_is_colab"]:
        drive_root = Path(config["paths"]["drive_root"])
        for key, val in config["paths"].items():
            if key != "drive_root" and not val.startswith("/"):
                config["paths"][key] = str(drive_root / val)
    else:
        for key, val in config["paths"].items():
            if key != "drive_root" and not val.startswith("/"):
                config["paths"][key] = str(_PROJECT_ROOT / val)

    return config


def _is_colab() -> bool:
    """Detect if running in Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def set_seeds(config: dict[str, Any]) -> None:
    """Set all random seeds for reproducibility.

    Args:
        config: Config dict containing seeds section.
    """
    import random
    import numpy as np
    import torch

    seeds = config.get("seeds", {})
    random.seed(seeds.get("random", 42))
    np.random.seed(seeds.get("numpy", 42))
    torch.manual_seed(seeds.get("torch", 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeds.get("torch", 42))


def get_device() -> "torch.device":
    """Get the best available device.

    Returns:
        torch.device for cuda or cpu.
    """
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def free_gpu_memory() -> None:
    """Free GPU memory between pipeline stages."""
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
