import cv2
import supervision as sv

from footballtracker import ByteTracker
from footballtracker import ObjectDetector
from footballtracker.io import ConfigManager
from footballtracker.viz import Visualizer


class FootballTracker:
    def __init__(self, config_manager: ConfigManager):
        self.object_detector = ObjectDetector(config_manager)
        self.object_classes_dict = self.object_detector.get_class_names_dict()
        self.input_video_path = config_manager.get('input_video_path')
        self.video_info = sv.VideoInfo.from_video_path(video_path=self.input_video_path)
        self.show_live = config_manager.get('show_live')
        self.tracker = ByteTracker(
            fps=int(self.video_info.fps),
            confidence_threshold=0.1,
            iou_threshold=0.7
        )
        self.visualizer = Visualizer(self.video_info.resolution_wh)

    def run(self):
        cap = cv2.VideoCapture(self.input_video_path)
        frames = []
        ret, frame = cap.read()
        while ret:
            # DETECT
            detections = self.object_detector.run_inference(frame)

            # TRACK
            detections = self.tracker.track(frame, detections)
            self.tracker.update(detections)

            # frame = self.visualizer.draw_tracked_detections(frame, detections, self.object_classes_dict)
            frame = self.visualizer.draw_detections(frame, detections, self.object_classes_dict)
            frames.append(frame)

            if self.show_live:
                cv2.imshow('Live Detections', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            ret, frame = cap.read()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    cfg_path = '/Users/antonirodriguezvillegas/PycharmProjects/FootyTrack/configs/config.yaml'
    cfg_manager = ConfigManager(cfg_path)
    footy_tracker = FootballTracker(cfg_manager)
    footy_tracker.run()
