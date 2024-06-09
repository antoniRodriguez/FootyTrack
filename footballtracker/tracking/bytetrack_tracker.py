from .base_tracker import BaseTracker
import numpy as np
from typing import List, Any
from supervision import ByteTrack, Detections


class ByteTracker(BaseTracker):
    def __init__(self, fps: int, confidence_threshold: float = 0.1, iou_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        # self.tracker = ByteTrack(fps, self.confidence_threshold)  # expects int but works with float?
        self.tracker = ByteTrack(frame_rate=fps)

    def update(self, detections: Detections) -> Detections:
        return self.tracker.update_with_detections(detections=detections)

    def reset(self):
        self.tracker.reset()

    def track(self, frame: np.ndarray, detections: List[Any]) -> Detections:
        detections = Detections.from_ultralytics(detections)
        detections = detections[detections.confidence > self.confidence_threshold]
        detections = detections.with_nms(threshold=self.iou_threshold)
        return detections
