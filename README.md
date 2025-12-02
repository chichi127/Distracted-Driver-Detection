# Driver Distraction Detection  
YOLO Detection + EfficientNet Classification Two-Stage Pipeline
---

![파이프라인](figures/pipeline.png)</br>
운전자의 주의 분산 행동(c0~c7)을 자동 탐지하기 위한 Two-Stage Deep Learning 파이프라인입니다.

- **Stage 1 – YOLO (Detection)**  
  운전자 동작 위치를 Bounding Box로 탐지

- **Stage 2 – EfficientNet (Classification)**  
  Crop 이미지 기반 행동 클래스 분류

- **Fusion Output**  
  YOLO + CLF 결합 최종 행동 예측

---

## Project Structure

```
DriverDistractionDetection/
│
├── README.md
├── requirements.txt
│
├── configs/
│   ├── data.yaml
│   ├── clf_config.json
│
├── yolo_detector/
│   ├── train_yolo.py
│   ├── test_yolo.py
│   ├── detect.py
│   └── utils_yolo.py
│
├── classifier/
│   ├── train_clf.py
│   ├── test_clf.py
│   ├── models_clf.py
│   └── utils_clf.py
│
├── fusion/
│   ├── run_fusion.py
│   └── crop_utils.py
│
├── dataset_tools/
│   ├── merge_c2c3_labels.py
│   ├── make_crops.py
│   └── split_dataset.py
│
├── figures/
│   ├── pipeline.png
│   └── examples/
│
└── examples/
    ├── input.jpg
    └── output.jpg
```

---

---

## Usage Flow

1. **데이터 라벨 통합 (c2/c3 → c2 통합)**  
   `dataset_tools/merge_c2c3_labels.py`

2. **YOLO 학습에 필요한 data.yaml 설정**  
   `configs/data.yaml`

3. **YOLO 탐지 모델 학습**  
   `yolo_detector/train_yolo.py`

4. **YOLO bbox 기반 Crop 생성**  
   `dataset_tools/make_crops.py`

5. **EfficientNet 분류 모델 학습**  
   `classifier/train_clf.py`

6. **YOLO + EfficientNet 통합 추론**  
   `fusion/run_fusion.py`

---

## End-to-End Pipeline (Example)

아래는 한 번에 전체 파이프라인을 실행하는 예시입니다.  
(경로와 옵션은 사용자의 데이터 구조에 맞게 수정해서 사용하세요.)

```bash
# 0) 의존성 설치
pip install -r requirements.txt

# 1) c2(폰 왼손) / c3(폰 오른손) 라벨을 하나의 c2 클래스로 통합
python dataset_tools/merge_c2c3_labels.py \
  --data-root data/driver

# 2) train / val / test 분리 (필요 시)
python dataset_tools/split_dataset.py \
  --data-root data/driver \
  --val-ratio 0.2 --test-ratio 0.1

# 3) YOLO 탐지 모델 학습
python yolo_detector/train_yolo.py \
  --data-config configs/data.yaml \
  --weights yolov8s.pt \
  --epochs 50

# 4) 학습된 YOLO bbox를 이용해 Crop 생성 (분류기 학습용)
python dataset_tools/make_crops.py \
  --data-root data/driver \
  --yolo-weights runs_yolo/exp/weights/best.pt \
  --out-dir data/driver/cls_crops

# 5) EfficientNet 분류기 학습
python classifier/train_clf.py \
  --config configs/clf_config.json \
  --data-root data/driver/cls_crops

# 6) YOLO + EfficientNet 통합 추론 (test set)
python fusion/run_fusion.py \
  --data-root data/driver \
  --yolo-weights runs_yolo/exp/weights/best.pt \
  --clf-weights cls_runs/efficientnet_b3_best.pth \
  --output-csv results/fusion_test_results.csv
---
```

## Notes

- 모든 코드에서 데이터 경로는 `configs/data.yaml`을 기준으로 변경 가능  
- Custom dataset에도 동일한 파이프라인 적용 가능  
