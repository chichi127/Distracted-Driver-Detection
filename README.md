# Driver Distraction Detection  
YOLO Detection + EfficientNet Classification Two-Stage Pipeline

운전자의 주의 분산 행동(c0~c7)을 자동 탐지하기 위한 Two-Stage Deep Learning 파이프라인입니다.

- **Stage 1 – YOLO (Detection)**  
  운전자 동작 위치를 Bounding Box로 탐지

- **Stage 2 – EfficientNet (Classification)**  
  Crop 이미지 기반 행동 클래스 분류

- **Fusion Output**  
  YOLO + CLF 결합 최종 행동 예측

---

## 📁 Project Structure

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

## 🚀 Usage Flow

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

## 🔗 References

- YOLO 기반 운전자 행동 탐지 연구  
- EfficientNet 기반 인간 행동 분류 연구  
- Detection → Classification Two-Stage 방법론

---

## 📝 Notes

- 모든 코드에서 데이터 경로는 `configs/data.yaml`을 기준으로 변경 가능  
- Custom dataset에도 동일한 파이프라인 적용 가능  
- 그림(`figures/pipeline.png`)을 README.md에 삽입 가능  
