import argparse
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from inference.geo import extract_and_save_polygons, stitch_tiles
from inference.utils import load_config, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roi-x", type=int, required=True)
    parser.add_argument("--roi-y", type=int, required=True)
    parser.add_argument("--roi-width", type=int, required=True)
    parser.add_argument("--roi-height", type=int, required=True)
    parser.add_argument("--downsample-rate", type=float, required=True)
    parser.add_argument("--tile-size", type=int, required=True)
    parser.add_argument("--confidence", type=float, required=True)
    return parser.parse_args()


def run_inference(args: argparse.Namespace, infer_size: int) -> NDArray[np.uint8]:
    # ratio between input size and model input/output size
    infer_scale = args.tile_size // infer_size
    # total scaling constant
    total_scale = args.downsample_rate * infer_scale

    # canvas shape
    canvas_width = int(args.roi_width / total_scale)
    canvas_height = int(args.roi_height / total_scale)
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    canvas_weights = np.zeros_like(canvas)

    return stitch_tiles(
        args.model_path,
        infer_size,
        args.tile_dir,
        canvas,
        args.roi_x,
        args.roi_y,
        canvas_weights,
        total_scale,
        args.confidence,
    )


def main():
    start_time = time.perf_counter()
    args = parse_args()
    logger = setup_logger()
    config = load_config()

    infer_size = config["model"]["infer_size"]
    gj_props = config["geojson_properties"]

    logger.info("Inference started")

    canvas = run_inference(args, infer_size)
    extract_and_save_polygons(canvas, args.output_dir, args.roi_x, args.roi_y, gj_props)
    run_time = time.perf_counter() - start_time
    logger.info(f"Finished in {run_time:.4f} secs")


if __name__ == "__main__":
    main()
