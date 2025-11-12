import json
import logging
import re
import sys
import time
from functools import wraps
from pathlib import Path


def setup_logger(
    log_file: str | Path, name: str = "inference", debug: bool = False
) -> logging.Logger:
    logger = logging.getLogger(name)
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def load_config(config_file: str = "config.json") -> dict:
    script_dir = Path(__file__).parent.parent
    absolute_config_path = script_dir / config_file
    with open(absolute_config_path, "r") as f:
        return json.load(f)


def timer(func):
    logger = logging.getLogger(func.__module__)

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            run_time = end_time - start_time
            logger.debug(f"Finished '{func.__name__}' in {run_time:.4f} secs.")

    return wrapper


def parse_properties(filename: str) -> dict:
    match = re.search(r"\[([^\]]+)\]", filename)
    if not match:
        return {}
    properties_str = match.group(1)
    properties = {}
    for item in properties_str.split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            if k == "d":
                v = float(v)
            else:
                v = int(v)
            properties[k] = v
    return properties
