import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from .image import read_image, scale

logger = logging.getLogger("inference")


def load_torch_model(model_path: Path):
    raise NotImplementedError


def infer_torch(image: NDArray[np.uint8], model) -> NDArray[np.float32]:
    raise NotImplementedError


def load_onnx_model(model_path: Path) -> ort.InferenceSession:
    device = ort.get_device()
    logger.info(f"Using onnx device - {device}")
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "GPU"
        else ["CPUExecutionProvider"]
    )
    model = ort.InferenceSession(str(model_path), providers=providers)
    logger.info(f"ONNX model loaded: {model_path.name}")
    return model


def infer_onnx(
    image: NDArray[np.uint8], model: ort.InferenceSession
) -> NDArray[np.float32]:
    image = scale(image)
    image = image.transpose(2, 0, 1)[None, ...]
    input_name = model.get_inputs()[0].name
    pred = model.run(None, {input_name: image})[0]
    return pred.squeeze()


def load(model_path: Path) -> ort.InferenceSession:
    ext = model_path.suffix.lower()
    if ext == ".onnx":
        return load_onnx_model(model_path)
    else:
        raise ValueError("Unsupported model extension")


def infer(
    image_path: Path,
    model: ort.InferenceSession,
    infer_size: int,
) -> NDArray[np.float32]:
    image = read_image(image_path)
    image = cv2.resize(image, (infer_size, infer_size))

    if isinstance(model, ort.InferenceSession):
        return infer_onnx(image, model)
    else:
        raise TypeError("Unsupported model type")
