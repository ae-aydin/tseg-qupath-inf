import argparse
import json
import sys
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
    parser.add_argument("--log-file", type=Path, required=True)
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
        infer_scale,
        total_scale,
        args.confidence,
    )


def main():
    args = parse_args()
    logger = None

    try:
        logger = setup_logger(args.log_file, debug=True)
        logger.info(f"Arguments: {vars(args)}")

        start_time = time.perf_counter()

        logger.info("Inference process started.")
        config = load_config()

        infer_size = config["model"]["infer_size"]
        gj_props = config["geojson_properties"]

        logger.info("Running inference...")
        canvas = run_inference(args, infer_size)

        logger.info("Extracting and saving polygons...")
        n_polygons = extract_and_save_polygons(
            canvas, args.output_dir, args.roi_x, args.roi_y, gj_props
        )

        run_time = time.perf_counter() - start_time
        logger.info(f"Finished successfully in {run_time:.4f} secs")

        success_message = {
            "status": "success",
            "message": "Processing completed successfully.",
            "runtime_sec": round(run_time, 4),
            "n_polygons": n_polygons,
            "log_file": args.log_file,
        }

        print(json.dumps(success_message))
        sys.exit(0)

    except Exception as e:
        if logger:
            logger.exception("A fatal error was caught by the main handler")

        error_message = {
            "status": "error",
            "message": str(e),
            "log_file": args.log_file,
        }

        print(json.dumps(error_message))
        sys.exit(1)


if __name__ == "__main__":
    main()
