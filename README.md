# Distracted-Driver-Detection
🚗 Driver Distraction Detection

YOLO Detection + EfficientNet Classification Two-Stage Pipeline

본 프로젝트는 운전자의 주의 분산 행동을 자동 탐지하기 위한 Two-Stage Deep Learning Pipeline입니다.
	•	Stage 1 — YOLO (Detection)
운전자 동작 위치를 Bounding Box로 탐지
	•	Stage 2 — EfficientNet (Classification)
Crop된 이미지로 행동 종류(c0~c7)를 정밀 분류
	•	Fusion Output
YOLO + CLF 결합 최종 행동 예측
