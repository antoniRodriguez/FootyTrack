from ultralytics import YOLO
import cv2

model_float32 = YOLO("/Users/antonirodriguezvillegas/PycharmProjects/FootyTrack/footballtracker/detection/weights/yolov8n.pt")  # Load an official model or custom model
model_TPU = YOLO("/Users/antonirodriguezvillegas/PycharmProjects/FootyTrack/footballtracker/detection/weights/yolov8n_full_integer_quant_edgetpu.tflite")  # Load an official model or custom model

# Read image
image = cv2.imread('/Users/antonirodriguezvillegas/PycharmProjects/FootyTrack/TPU_optimization/test/img.jpg')

model_float32.predict(image)
print('Predicted using the float32 standard model!')
model_TPU.predict(image)
print('Predicted using the full integer-quantized model!')
