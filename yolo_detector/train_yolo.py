# yolo_detector/train_yolo.py
from ultralytics import YOLO
from .utils_yolo import get_data_yaml_path, get_project_dir, get_model_list

EPOCHS    = 50
PATIENCE  = 5
IMGSZ     = 640
BATCH     = 32
DEVICE    = 0


def train_one(model_name: str):
    project = get_project_dir()
    data_yaml = get_data_yaml_path()
    exp_name = f"{model_name.replace('.pt','')}_early5_fast"

    print("\n======================================================================")
    print(f"   {model_name} 학습 시작 (50ep, earlystop=5)")
    print(f"   data={data_yaml}")
    print(f"   project={project / exp_name}")
    print("======================================================================")

    model = YOLO(model_name)
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=4,
        device=DEVICE,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,
        deterministic=True,
        cache=True,
        project=str(project),
        name=exp_name,
        exist_ok=True,
        verbose=True,
    )


def main():
    for m in get_model_list():
        train_one(m)
    print("\n 모든 YOLO 모델 학습 완료")


if __name__ == "__main__":
    main()
