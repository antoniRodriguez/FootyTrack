import cv2
import numpy as np
import os
from ultralytics import YOLO
from footballtracker.io import ConfigManager
from typing import List


class ObjectDetector:
    def __init__(self, config_manager: ConfigManager, device: str = 'mps'):
        self.is_tpu_optimized = config_manager.get('TPU_optimization')
        self.model = self.load_model(config_manager.get('object_detector.weights_path'))
        self.class_names_dict = config_manager.get('object_detector.classes')
        self.device = device

    def load_model(self, path_to_weights: str) -> YOLO:
        if self.is_tpu_optimized:
            fname_weights = os.path.basename(path_to_weights)
            fname_quantized_weights = fname_weights.replace('.pt', '_full_integer_quant_edgetpu.tflite')
            path_to_weights = os.path.join(os.path.dirname(path_to_weights), fname_quantized_weights)
        return YOLO(path_to_weights)

    def run_inference(self, frame: np.ndarray) -> List:
        if self.is_tpu_optimized:
            results = self.model(frame)[0]
        else:
            results = self.model(frame, device=self.device)[0]
        return results

    def get_class_names_dict(self):
        return self.class_names_dict

    def draw_detections(self, frame: np.ndarray, detections) -> np.ndarray:
        """
        Args:
            frame: np.ndarray
            detections: ultralytics.yolo.engine.results.Results
        """
        bboxes = detections.xyxy
        classif_indices = detections.class_id
        confidences = detections.confidence
        for det_idx, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            class_name = self.class_names_dict[classif_indices[det_idx]]
            conf = confidences[det_idx]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
