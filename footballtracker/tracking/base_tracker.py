from abc import ABC, abstractmethod
import numpy as np
from typing import List, Any

class BaseTracker(ABC):
    @abstractmethod
    def update(self, detections: List[Any]) -> List[Any]:
        """Update the tracker with the latest detections and return the tracked objects"""
        pass

    @abstractmethod
    def reset(self):
        """Reset the tracker to its initial state"""
        pass

    @abstractmethod
    def track(self, frame: np.ndarray, detections: List[Any]) -> List[Any]:
        """Track objects in the given frame"""
        pass
