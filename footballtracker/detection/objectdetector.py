import cv2
import numpy as np
from ultralytics import YOLO
from footballtracker.io import ConfigManager
from typing import List


class ObjectDetector:
    def __init__(self, config_manager: ConfigManager, device: str = 'mps'):
        self.model = self.load_model(config_manager.get('object_detector.weights_path'))
        self.class_names_dict = config_manager.get('object_detector.classes')
        self.device = device

    @staticmethod
    def load_model(path_to_weights: str) -> YOLO:
        return YOLO(path_to_weights)

    def run_inference(self, frame: np.ndarray) -> List:
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
