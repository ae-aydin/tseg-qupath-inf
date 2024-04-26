import argparse
import logging
import os
import sys
import time

import cv2
import geojson
import numpy as np
import torch
import yaml
from ultralytics import YOLO

os.environ["YOLO_VERBOSE"] = "False"

logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


PROPERTIES = {
    "objectType": "annotation",
    "classification": {"name": "Tumor", "color": [0, 0, 200]},
}


def check_version(major: int = 3, minor: int = 6):
    python_version = sys.version_info
    return int(python_version.major) == major and int(python_version.minor) >= minor


def format_shapes(shape: tuple, ds: float = 1.0):
    """Format tuples for logging

    Args:
        shape (tuple): tuple
        ds (float, optional): downsample. Defaults to 1.0.

    Returns:
        str: tuple elements formatted
    """
    format_arr = np.array(shape) * int(ds)
    return "x".join(format_arr.astype("str").tolist())


def add_offsets(contour: list, offset_x: float, offset_y: float):
    """Add region offsets to coordinates

    Args:
        contour (list): coordinates
        offset_x (float): x offset in px
        offset_y (float): y offset in px

    Returns:
        _type_: _description_
    """
    np_contour = np.array(contour)
    np_contour[:, 0] = np_contour[:, 0] + offset_x
    np_contour[:, 1] = np_contour[:, 1] + offset_y
    return np_contour.tolist()


def mask_to_geojson(mask: np.ndarray, output_path: str, args: argparse.Namespace):
    """Convert given mask to geojson (FeatureCollection)

    Args:
        mask (np.ndarray): binary mask
        output_path (str): where geojson file will be saved
        args (argparse.Namespace): args
    """
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS
    )
    features = list()
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            external = contour.squeeze().tolist()
            if external[0] != external[-1]:
                external.append(external[0])
                external = add_offsets(external, args.roi_x, args.roi_y)
            holes = []
            if hierarchy[0][i][2] != -1:
                child_idx = hierarchy[0][i][2]
                while child_idx != -1:
                    hole = contours[child_idx].squeeze().tolist()
                    if hole[0] != hole[-1]:
                        hole.append(hole[0])
                    holes.append(hole)
                    child_idx = hierarchy[0][child_idx][0]
            geometry = geojson.Polygon([external] + holes)
            feature = geojson.Feature(geometry=geometry, properties=PROPERTIES)
            features.append(feature)
    feature_collection = geojson.FeatureCollection(features)
    logger.info(f"Found {len(features)} objects.")
    with open(output_path, "w") as f:
        f.write(geojson.dumps(feature_collection, indent=2))


def predict(paths: dict, args: argparse.Namespace, imgsz: int = 640):
    """Get predictions for given image

    Args:
        paths (dict): file paths
        args (argparse.Namespace): args
        imgsz (int, optional): inference image size. Defaults to 640.

    Returns:
        mask: binary prediction mask
    """
    model = YOLO(paths["model_path"])
    roi_shape = cv2.imread(paths["roi_path"]).shape[:-1]
    logger.info(f"Selected region size is {format_shapes(roi_shape, ds= args.ds)}")
    logger.info(f"QuPath export size is {format_shapes(roi_shape)}")
    logger.info(
        f"Model arguments - confidence: {args.conf}, iou: {args.iou}, imgsz: {imgsz}"
    )
    preds = model.predict(
        paths["roi_path"], conf=args.conf, iou=args.iou, imgsz=imgsz, verbose=False
    )
    try:
        stacked_preds = torch.stack([mask.data for p in preds for mask in p.masks])
    except Exception as e:
        return None
    mask_size = stacked_preds.shape[2:]
    mask = (
        torch.any(stacked_preds, dim=0).numpy().reshape(*mask_size, 1).astype(np.uint8)
    )
    logger.info(f"Prediction mask size is {format_shapes(mask.shape[:-1])}")
    mask = cv2.resize(mask, roi_shape[::-1], interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, None, fx=args.ds, fy=args.ds, interpolation=cv2.INTER_CUBIC)
    return mask * 255


def load_arguments():
    """Load arguments

    Returns:
        args: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("base_path", type=str)
    parser.add_argument("roi_x", type=float)
    parser.add_argument("roi_y", type=float)
    parser.add_argument("ds", type=float)
    parser.add_argument("conf", type=float)
    parser.add_argument("iou", type=float)
    args = parser.parse_args()
    return args


def load_paths(args: argparse.Namespace):
    """Prepare and load file paths

    Args:
        args (argparse.Namespace): args

    Returns:
        dict: path dict
    """
    with open(os.path.join(args.base_path, "settings.yaml"), "r") as f:
        settings = yaml.safe_load(f)
    paths = dict()
    paths["model_path"] = os.path.join(
        args.base_path, settings["models_path"], settings["model_name"]
    )
    paths["roi_path"] = os.path.join(
        args.base_path, settings["rois_path"], settings["roi_name"]
    )
    paths["output_path"] = os.path.join(
        args.base_path, settings["preds_path"], settings["output_name"]
    )
    return paths


def main():
    if not check_version():
        logger.warning("PYTHON VERSION SHOULD BE AT LEAST 3.6!")
        return
    logger.info("Process started.")
    start_time = time.time()
    args = load_arguments()
    paths = load_paths(args)
    mask = predict(paths, args)
    if mask is not None:
        mask_to_geojson(mask, paths["output_path"], args)
    else:
        logger.warning("NO OBJECTS FOUND.")
    elapsed_time_ms = (time.time() - start_time) * 1000
    logger.info(f"Process finished in {round(elapsed_time_ms, 2)} ms.")


if __name__ == "__main__":
    main()
