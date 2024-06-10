# FootyTrack ⚽🏃‍♂️

Welcome to FootyTrack, a cutting-edge tool for tracking football players in videos. This project uses the YOLOv8 model for object detection and supports TPU acceleration for enhanced performance.

##  Features 🚀
- **Real-time Detection**: Detects and tracks football players, referees, balls, and more.
- **TPU Acceleration**: Supports TPU for faster inference times (Python 3.3-3.9).
- **CLI Support**: Easy-to-use command-line interface.
- **Model**: Utilizes YOLOv8 for accurate and efficient detection.

## Installation 🛠️

To install FootyTrack, simply clone the repository and use pip to install it as a package:

```sh
git clone https://github.com/antoniRodriguez/FootyTrack.git
cd FootyTrack
pip install .
```

## TPU acceleration ⚡
For TPU acceleration, make sure your Python version is between 3.3 and 3.9 due to Google Coral limitations. Follow the [Google Coral installation instructions](https://coral.ai/docs/accelerator/get-started/) to set up TPU support.
A set of scripts to test of helper tools for TPU installation and model conversion can be found under `TPU_optimization/`

## Usage 🖥️
```sh
footytracker --input 'video.mp4' --config 'config.yaml' --tpu --show_live
```
* --input (mandatory): Path to the input video file. 
* --config (mandatory): Path to the configuration file. 
* --tpu (optional): Enable TPU acceleration. 
* --show_live (optional): Display the live detection results.
`

## Object detection model
The current detection model is yolov8s.pt. The TPU model is a tflite version of the float32 model with post-training full integer quantization. 
`Note that 
> ⚠️🚧 **Warning**: The TPU model is under development, and its performance may not be on par with the float32 model (Post-training full integer quantisation has been used for the current TPU model, a quantisation-aware training version is on the works)

![confusion_matrix.png](..%2F..%2FDownloads%2Fcontent%203%2Fruns%2Fdetect%2Ftrain7%2Fconfusion_matrix.png)