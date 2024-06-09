import cv2

from footballtracker import ByteTracker
from footballtracker import ObjectDetector


def process_video(config_path: str, input_video_path: str, show_live: bool = False):
    cap = cv2.VideoCapture(input_video_path)

    object_detector = ObjectDetector(config_path)
    tracker = ByteTracker(
        fps=int(cap.get(cv2.CAP_PROP_FPS)),
        confidence_threshold=0.1,
        iou_threshold=0.7
    )

    frames = []
    ret, frame = cap.read()
    while ret:
        # DETECT
        detections = object_detector.run_inference(frame)

        # TRACK
        detections = tracker.track(frame, detections)
        tracker.update(detections)

        frame = object_detector.draw_detections(frame, detections)
        frames.append(frame)

        if show_live:
            cv2.imshow('Live Detections', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    process_video('/Users/antonirodriguezvillegas/PycharmProjects/FootyTrack/configs/config.yaml',
                  '/Users/antonirodriguezvillegas/PycharmProjects/FootyTrack/data/0b1495d3_0.mp4',
                  'output/output_video.mp4', show_live=True)
