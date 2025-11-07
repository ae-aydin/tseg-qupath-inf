import json
import logging
import re
import sys
import time
from functools import wraps
from pathlib import Path

logger = logging.getLogger("inference")


def setup_logger(name: str = "inference", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_config(config_file: str = "config.json") -> dict:
    script_dir = Path(__file__).parent.parent
    absolute_config_path = script_dir / config_file
    with open(absolute_config_path, "r") as f:
        return json.load(f)


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            run_time = end_time - start_time
            logger.info(f"Finished '{func.__name__}' in {run_time:.4f} secs")
            for handler in logger.handlers:
                handler.flush()
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
            if k == 'd':
                v = float(v)
            else:
                v = int(v)
            properties[k] = v
    return properties
