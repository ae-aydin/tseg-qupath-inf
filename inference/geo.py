import logging
from pathlib import Path

import cv2
import geojson
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from .image import gaussian_weight_map, post_process, sigmoid
from .model import infer, load
from .utils import parse_properties, timer

logger = logging.getLogger("inference")


@timer
def stitch_tiles(
    model_path: Path,
    infer_size: int,
    tile_dir: Path,
    canvas: np.ndarray,
    canvas_weight: np.ndarray,
    total_scale: float,
    confidence: float,
) -> NDArray[np.uint8]:
    model = load(model_path)
    gaussian_weight = gaussian_weight_map(infer_size)
    for tile_path in tqdm(list(tile_dir.iterdir()), ncols=64, desc="Stitching tiles"):
        tile_properties = parse_properties(tile_path.stem)
        pred = infer(tile_path, model, infer_size)
        y = int(tile_properties["y"] / total_scale)
        x = int(tile_properties["x"] / total_scale)
        canvas[y : y + infer_size, x : x + infer_size] += pred * gaussian_weight
        canvas_weight[y : y + infer_size, x : x + infer_size] += gaussian_weight

    logger.info("Averaging the stitched logits")
    min_logit = np.min(canvas[canvas_weight > 0]) if np.any(canvas_weight > 0) else 0
    out_canvas = np.full_like(canvas, fill_value=min_logit, dtype=np.float32)
    r_canvas = np.divide(canvas, canvas_weight, out=out_canvas, where=canvas_weight > 0)
    r_canvas = sigmoid(r_canvas)

    logger.info("Post processing the ROI")
    r_canvas = post_process(r_canvas, confidence)

    canvas_full_size = (
        int(r_canvas.shape[1] * total_scale),
        int(r_canvas.shape[0] * total_scale),
    )
    logger.info(f"Resizing the ROI to full size -> {canvas_full_size}")
    r_canvas = cv2.resize(r_canvas, canvas_full_size, interpolation=cv2.INTER_NEAREST)

    return r_canvas.astype(np.uint8)


def add_offsets(contour: list, roi_x: float, roi_y: float) -> list:
    np_contour = np.array(contour)
    np_contour[:, 0] += roi_x
    np_contour[:, 1] += roi_y
    return np_contour.tolist()


@timer
def extract_and_save_polygons(
    canvas: np.ndarray,
    output_dir: Path,
    roi_x: int,
    roi_y: int,
    properties: dict,
    min_area: int = 32,
    filename: str = "polygons.geojson",
) -> None:
    contours, hierarchy = cv2.findContours(
        canvas, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    features = []

    if hierarchy is None:
        logger.warning("No contours were found in the provided canvas")
        return

    hierarchy = hierarchy[0]
    for i, contour in enumerate(tqdm(contours, desc="Processing contours", ncols=64)):
        if cv2.contourArea(contour) < min_area:
            continue

        if hierarchy[i][3] == -1:
            external = contour.squeeze().tolist()
            if len(external) < 3:
                continue

            if external[0] != external[-1]:
                external.append(external[0])

            external = add_offsets(external, roi_x, roi_y)
            holes = []
            child_idx = hierarchy[i][2]

            while child_idx != -1:
                hole_contour = contours[child_idx]
                if cv2.contourArea(hole_contour) >= min_area:
                    hole = hole_contour.squeeze().tolist()
                    if len(hole) >= 3:
                        if hole[0] != hole[-1]:
                            hole.append(hole[0])

                        hole = add_offsets(hole, roi_x, roi_y)
                        holes.append(hole)

                child_idx = hierarchy[child_idx][0]

            geometry = geojson.Polygon([external] + holes)
            features.append(geojson.Feature(geometry=geometry, properties=properties))

    logger.info(f"Extracted {len(features)} polygon(s)")
    feature_collection = geojson.FeatureCollection(features)
    output_path = output_dir / filename
    output_path.write_text(geojson.dumps(feature_collection))
