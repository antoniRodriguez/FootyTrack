import json
import subprocess

import yaml


class ConfigManager:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)

    @staticmethod
    def load_config(config_path: str) -> dict:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def get(self, key: str, default=None):
        keys = key.split('.')  # account for nested dictionaries
        value = self.config
        for k in keys:
            value = value.get(k, default)
            if value is default:
                break
        return value


def get_video_fps(video_path: str) -> float:
    """
    Extract the frames per second (FPS) of a video using ffprobe.

    Parameters:
    video_path (str): Path to the video file.

    Returns:
    float: video frame rate (FPS)
    """
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "json",
        video_path
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr}")

    info = json.loads(result.stdout)
    frame_rate = info['streams'][0]['r_frame_rate']
    num, denom = map(int, frame_rate.split('/'))

    return num / denom
