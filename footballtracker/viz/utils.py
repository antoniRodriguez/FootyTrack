import cv2
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

    def add_text(self, frame: np.ndarray, text: str):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.text_scale
        font_thickness = 2
        text_color = (0, 255, 0)  # Green
        background_color = (0, 0, 0)  # Black
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

        text_x = 10  # 10 pixels from the left edge
        text_y = frame.shape[0] - 10  # 10 pixels from the bottom edge

        # Define the rectangle background for the text
        rect_x1 = text_x - 5
        rect_y1 = text_y + baseline
        rect_x2 = text_x + text_width + 5
        rect_y2 = text_y - text_height - 5

        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), background_color, thickness=cv2.FILLED)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

        return frame
