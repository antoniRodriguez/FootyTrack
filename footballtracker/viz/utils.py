import numpy as np
import supervision as sv


class Visualizer:
    def __init__(self, resolution_wh):
        self.text_thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
        self.text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)
        self.bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=self.text_thickness)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=self.text_scale / 2,
            text_thickness=self.text_thickness // 2,
            text_position=sv.Position.BOTTOM_CENTER,
        )

    def draw_detections(self, frame: np.ndarray, detections, class_names_dict) -> np.ndarray:
        """
                Args:
                    detections: ultralytics.yolo.engine.results.Results
                """
        tracker_ids = detections.tracker_id
        confidences = detections.confidence

        labels = [
            f"{class_names_dict.get(int(class_id), 'Unknown')}[{tracker_id}]_{confidence:.2f}"
            for tracker_id, class_id, confidence in zip(tracker_ids, detections.class_id, confidences)
        ]
        annotated_frame = frame.copy()
        annotated_frame = self.bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        return annotated_frame

    def draw_tracked_detections(self, frame: np.ndarray, detections, class_names_dict) -> np.ndarray:
        tracker_ids = detections.tracker_id
        confidences = detections.confidence
        labels = [
            f"ID {tracker_id} {class_names_dict.get(int(class_id), 'Unknown')} {confidence:.2f}"
            for tracker_id, class_id, confidence in zip(tracker_ids, detections.class_id, confidences)
        ]
        annotated_frame = self.bounding_box_annotator.annotate(
            scene=frame, detections=detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        return annotated_frame
