import yaml
import os
from ultralytics import YOLO


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_model(config_path: str) -> YOLO:
    config = load_config(config_path)
    path_to_weights = config.get('path_to_weights')
    is_tpu_model = config.get('TPU_optimization')
    if is_tpu_model:
        path_to_weights = path_to_weights.replace('.', '_full_integer_quant_edgetpu.tflite')

    model = YOLO(path_to_weights)
    return model
