import json
import logging
import sys
from pathlib import Path


def check_version(major: int = 3, minor: int = 8):
    """
    Check if the Python version meets the specified minimum requirements.

    Args:
        major (int, optional): The major Python release. Defaults to 3.
        minor (int, optional): The minor Python release. Defaults to 8.

    Returns:
        bool: Whether the Python version meets the specified minimum requirements.
    """
    python_version = sys.version_info
    return int(python_version.major) == major and int(python_version.minor) >= minor


def format_shapes(shape: tuple):
    """
    Format tuples for logging.

    Args:
        shape (tuple): Shape tuple.

    Returns:
        str: Shape tuple formatted like common resolution representation.
    """
    # (h, w)
    temp_shape = shape[::-1]  # (w, h)
    return "x".join([str(i) for i in temp_shape])


def read_config(config_file: Path, base_path: Path, logger: logging.Logger):
    """
    Read config json file.

    Args:
        config_file (Path): Config file path.
        base_path (Path): Base directory path.
        logger (logging.Logger): Logger to use.

    Returns:
        dict: Configurations as dictionary.
    """
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        config["model_path"] = base_path / config["model_path"]
        config["roi_tiles_path"] = base_path / config["roi_tiles_path"]
        config["preds_path"] = base_path / config["preds_path"]
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file '{config_file}' not found.")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in configuration file '{config_file}': {e}")
        return None
