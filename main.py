import argparse
import logging
import os
import re
import time
from pathlib import Path

import cv2
import geojson
import numpy as np
from ultralytics import YOLO

from utils import check_version, format_shapes, read_config

os.environ["YOLO_VERBOSE"] = "False"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


def add_offsets(contour: list, offset_x: float, offset_y: float):
    """
    Add ROI offset values to coordinates.

    Args:
        contour (list): Found polygons and coordinates.
        offset_x (float): Offset value in X-axis.
        offset_y (float): Offset value in Y-axis.

    Returns:
        list: Offset added coordinates.
    """
    np_contour = np.array(contour)
    np_contour[:, 0] = np_contour[:, 0] + offset_x
    np_contour[:, 1] = np_contour[:, 1] + offset_y
    return np_contour.tolist()


def mask_to_geojson(
    mask: np.ndarray, output_path: str, args: argparse.Namespace, properties: dict
):
    """
    Extract polygons from given mask and save to given output path.

    Args:
        mask (np.ndarray): Segmentation mask.
        output_path (str): Output path.
        args (argparse.Namespace): Parsed command-line arguments.
        properties (dict): QuPath-specific geoJson properties.

    Returns:
        None
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
            feature = geojson.Feature(geometry=geometry, properties=properties)
            features.append(feature)
    feature_collection = geojson.FeatureCollection(features)
    logging.info(f"Found {len(features)} objects.")
    with open(output_path, "w") as f:
        f.write(geojson.dumps(feature_collection, indent=2))


def get_boundaries(tile_info: dict, args: argparse.Namespace, mask_shape: tuple):
    """
    Get slice boundaries of to-be-assigned mask to main mask.

    Args:
        tile_info (dict): Tile properties.
        args (argparse.Namespace): Parsed command-line arguments.
        mask_shape (tuple): Exact sizes of the mask.

    Returns:
        tuple: A tuple containing tile properties.
            - lower_y (int): The lower bound of the y-coordinate.
            - upper_y (int): The upper bound of the y-coordinate.
            - lower_x (int): The lower bound of the x-coordinate.
            - upper_x (int): The upper bound of the x-coordinate.
    """
    lower_x = round((tile_info["x"] - args.roi_x) / args.ds)
    diff_x = abs(lower_x) if lower_x < 0 else 0

    lower_x = 0 if lower_x < 0 else lower_x
    upper_x = lower_x + mask_shape[1] + diff_x

    lower_y = round((tile_info["y"] - args.roi_y) / args.ds)
    diff_y = abs(lower_y) if lower_y < 0 else 0

    lower_y = 0 if lower_y < 0 else lower_y
    upper_y = lower_y + mask_shape[0] + diff_y

    assert upper_x - lower_x == mask_shape[1]
    assert upper_y - lower_y == mask_shape[0]
    return (lower_y, upper_y, lower_x, upper_x)


def extract_tile_info(tile_name: str):
    """
    Extract tile information from tile file name.

    Args:
        tile_name (str): Name of the tile file.

    Returns:
        dict: Tile properties.
    """
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, tile_name)
    if matches:
        matched_string = matches[0]
        split_string = matched_string.split(",")
        extracted_dict = {}
        extracted_dict["file_name"] = tile_name
        for item in split_string:
            key, value = item.split("=")
            extracted_dict[key] = float(value) if key == "ds" else int(value)
        return extracted_dict
    else:
        return None


def clean_mask(mask: np.ndarray, operation: str, kernel_size: int):
    """
    Apply a morphological operation to given mask.

    Args:
        mask (np.ndarray): The input mask.
        operation (str): The type of morphological operation to apply.
            Supported values: 'erosion', 'dilation', 'opening', 'closing'.
        kernel_size (int): The size of the kernel used for morphological operations.

    Raises:
        ValueError: If the specified operation is not supported.

    Returns:
        np.ndarray: Morpohology applied mask.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == "close":
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    elif operation == "open":
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == "erode":
        cleaned_mask = cv2.erode(mask, kernel)
    elif operation == "dilate":
        cleaned_mask = cv2.dilate(mask, kernel)
    else:
        raise ValueError("Operation should be either 'close' or 'open'")
    return cleaned_mask


def predict(config: dict, args: argparse.Namespace, operation: str, k_size: int):
    """
    Get predictions on all tiles and reconstruct a mask for main ROI.

    Args:
        config (dict): Dictionary containing configurations such as I/O paths.
        args (argparse.Namespace): Parsed command-line arguments.
        operation (str): Morphological operation to use.
        kernel_size (int): The size of the kernel used for morphological operations.

    Returns:
        np.ndarray: Reconstructed segmentation mask.
    """
    full_shape = (int(args.roi_height), int(args.roi_width))
    w = int(np.ceil(args.roi_width / args.ds))
    h = int(np.ceil(args.roi_height / args.ds))
    logging.info(f"ROI size: {format_shapes(full_shape)} -> {format_shapes((h, w))}")

    main_mask = np.zeros((h, w), dtype=np.uint8)
    model = YOLO(config["model_path"])
    logging.info(
        f"Model arguments - confidence: {args.conf}, iou: {args.iou}, imgsz: {args.imgsz}"
    )
    tile_count = len(os.listdir(config["roi_tiles_path"]))
    logging.info(f"{tile_count} tiles found.")

    infer_start = time.time()
    preds = model.predict(
        config["roi_tiles_path"],
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        stream=True,
        verbose=False,
    )
    logging.info(f"Inference completed in {time.time() - infer_start:.2f} seconds.")

    logging.info("Arranging predictions to create full mask...")
    mask_start = time.time()
    empty_tile_list = list()
    for pred in preds:
        tile_info = extract_tile_info(Path(pred.path).stem)
        if tile_info is None:
            logging.error("PROBLEM WITH ROI TILE NAMES!")
            return

        mask = pred.masks
        if mask is not None:
            merged_mask = np.any(mask.numpy().data, axis=0).astype(np.uint8)
            empty_tile_list.append(True)
        else:
            empty_tile_list.append(False)
            continue

        merged_mask = cv2.resize(
            merged_mask, pred.orig_shape[::-1], interpolation=cv2.INTER_CUBIC
        )
        bound = get_boundaries(tile_info, args, merged_mask.shape)
        main_mask[bound[0] : bound[1], bound[2] : bound[3]] |= merged_mask
    logging.info(f"Full mask prepared in {time.time() - mask_start:.2f} seconds.")

    if not np.any(empty_tile_list):
        logging.warning("NO OBJECTS FOUND.")
        return

    main_mask = clean_mask(main_mask, operation, k_size)
    main_mask = cv2.resize(
        main_mask, None, fx=args.ds, fy=args.ds, interpolation=cv2.INTER_CUBIC
    )
    return main_mask * 255


def load_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("base_path", type=Path)
    parser.add_argument("roi_x", type=float)
    parser.add_argument("roi_y", type=float)
    parser.add_argument("roi_width", type=float)
    parser.add_argument("roi_height", type=float)
    parser.add_argument("imgsz", type=int)
    parser.add_argument("ds", type=float)
    parser.add_argument("conf", type=float)
    parser.add_argument("iou", type=float)
    args = parser.parse_args()
    return args


def main():
    if not check_version():
        logging.error("PYTHON VERSION SHOULD BE AT LEAST 3.8!")
        return
    logging.info("Process started.")
    start_time = time.time()
    args = load_arguments()
    config_path = args.base_path / "config.json"
    config = read_config(config_path, args.base_path, logging.getLogger(__name__))
    mask = predict(
        config,
        args,
        config["morphology"]["operation"],
        config["morphology"]["kernel_size"],
    )
    if mask is not None:
        mask_to_geojson(mask, config["preds_path"], args, config["geojson_properties"])
    logging.info(f"Process finished in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
