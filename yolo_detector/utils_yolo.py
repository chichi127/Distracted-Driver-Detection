# yolo_detector/utils_yolo.py
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "driver"
DATA_ROOT = Path(os.environ.get("DDD_DATA_ROOT", str(DEFAULT_DATA_ROOT)))

CONFIG_DIR = PROJECT_ROOT / "configs"
YOLO_PROJECT_DIR = PROJECT_ROOT / "runs" / "yolo"   # 깃허브엔 여기 저장

def get_data_yaml_path() -> Path:
    return CONFIG_DIR / "data.yaml"

def get_project_dir() -> Path:
    YOLO_PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    return YOLO_PROJECT_DIR

def get_model_list():
    return [
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov8m.pt",
        "yolo11n.pt",
        "yolo11s.pt",
        "yolo11m.pt",
    ]
