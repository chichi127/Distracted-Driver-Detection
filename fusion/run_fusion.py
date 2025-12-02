# fusion/run_fusion.py

"""
YOLO + EfficientNet-B3 통합 추론 스크립트.

기본 동작:
1) YOLO detector로 전체 이미지에서 운전자/행동 영역 bbox 탐지
2) crop_utils를 이용해 bbox 영역 crop
3) EfficientNet-B3 classifier로 최종 행동(c0~c7) 분류
4) (선택) test_labels_gt.csv가 있으면 accuracy 계산
5) 예시 이미지를 bbox + predicted label로 저장

데이터 루트:
- env DDD_DATA_ROOT 사용 가능
- 없으면 <repo_root>/data/driver
"""

import os
from pathlib import Path
import argparse

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

from ultralytics import YOLO
import pandas as pd

from crop_utils import crop_box_xyxy, draw_bbox_with_label


def get_data_root() -> Path:
    env = os.getenv("DDD_DATA_ROOT")
    if env is not None:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "data" / "driver"


def build_classifier(num_classes: int, ckpt_path: Path, device: str = "cuda"):
    """
    EfficientNet-B3 분류기 로드.
    - num_classes: 8 (c0~c7)
    - ckpt_path: fine-tuned weight (.pth)
    """
    weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
    model = models.efficientnet_b3(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_clf_transform():
    """EfficientNet 학습 시 사용했던 전형적인 변환."""
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO + EfficientNet fusion inference"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Dataset root (default: env DDD_DATA_ROOT or <repo>/data/driver)",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default=None,
        help="Path to YOLO weights (best.pt)",
    )
    parser.add_argument(
        "--clf-weights",
        type=str,
        default=None,
        help="Path to EfficientNet-B3 classifier weights (.pth)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of example images to save with bbox+label",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save example images and CSV (default: <data_root>/fusion_results)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ===== 0. paths =====
    data_root = Path(args.data_root) if args.data_root else get_data_root()
    test_img_dir = data_root / "test" / "images"
    gt_csv_path = data_root / "test_labels_gt.csv"

    # default YOLO & CLF weights 위치 (README에서 안내했던 구조 기준)
    default_yolo = (
        Path(__file__).resolve().parents[1]
        / "runs"
        / "yolo"
        / "yolov8s_early5_fast"
        / "weights"
        / "best.pt"
    )
    default_clf = (
        data_root / "cls_runs_v2_efficientnet" / "efficientnet_b3_best.pth"
    )

    yolo_weights = Path(args.yolo_weights) if args.yolo_weights else default_yolo
    clf_weights = Path(args.clf_weights) if args.clf_weights else default_clf

    save_dir = Path(args.save_dir) if args.save_dir else (data_root / "fusion_results")
    save_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] DATA_ROOT      = {data_root}")
    print(f"[INFO] test images    = {test_img_dir}")
    print(f"[INFO] YOLO weights   = {yolo_weights}")
    print(f"[INFO] CLF weights    = {clf_weights}")
    print(f"[INFO] SAVE_DIR       = {save_dir}")
    print(f"[INFO] DEVICE         = {device}")

    # ===== 1. load models =====
    print("\n[INFO] Loading YOLO detector...")
    yolo_model = YOLO(str(yolo_weights))

    print("[INFO] Loading EfficientNet-B3 classifier...")
    num_classes = 8  # c0 ~ c7
    clf_model = build_classifier(num_classes, clf_weights, device=device)
    clf_tf = get_clf_transform()
    class_names = [f"c{i}" for i in range(num_classes)]

    # ===== 2. load images =====
    img_paths = sorted(list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png")))
    print(f"\n[INFO] Found test images: {len(img_paths)}")

    results = []
    example_count = 0

    for idx, img_path in enumerate(img_paths, start=1):
        print(f"[{idx}/{len(img_paths)}] {img_path.name}")

        # --- YOLO inference ---
        yolo_out = yolo_model(str(img_path))[0]
        boxes = yolo_out.boxes

        if boxes is None or len(boxes) == 0:
            print("  [WARN] No detection, skip")
            results.append(
                {
                    "image": str(img_path),
                    "yolo_cls": None,
                    "clf_cls": None,
                }
            )
            continue

        # 가장 conf 높은 박스 하나만 사용
        confs = boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confs))
        box_xyxy = boxes.xyxy[best_idx].cpu().numpy().tolist()
        yolo_cls_id = int(boxes.cls[best_idx].item())

        # --- crop 영역 추출 ---
        # 원본 BGR 이미지
        img_bgr = cv2.imread(str(img_path))
        crop_bgr = crop_box_xyxy(img_bgr, box_xyxy, pad_ratio=0.05)

        # 분류기 입력용 (PIL + transform)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        crop_tensor = clf_tf(crop_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = clf_model(crop_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_cls_id = int(torch.argmax(probs, dim=1).item())
            pred_cls_name = class_names[pred_cls_id]
            # prob = float(probs[0, pred_cls_id].item())  # 필요하면 사용

        results.append(
            {
                "image": str(img_path),
                "yolo_cls": yolo_cls_id,
                "clf_cls": pred_cls_id,
                "clf_cls_name": pred_cls_name,
            }
        )

        # --- 예시 이미지 저장 (bbox + label) ---
        if example_count < args.num_examples:
            label_text = f"{pred_cls_name}"
            vis = draw_bbox_with_label(
                img_bgr,
                box_xyxy,
                label=label_text,
                color=(0, 255, 0),
            )
            out_name = f"example_{example_count+1:02d}.jpg"
            cv2.imwrite(str(save_dir / out_name), vis)
            example_count += 1

    # ===== 3. 결과 CSV 저장 =====
    df_pred = pd.DataFrame(results)
    csv_path = save_dir / "fusion_predictions.csv"
    df_pred.to_csv(csv_path, index=False)
    print(f"\n Saved fusion predictions to: {csv_path}")

    # ===== 4. GT가 있으면 accuracy 계산 =====
    if gt_csv_path.exists():
        print(f"[INFO] Found GT CSV: {gt_csv_path}")
        df_gt = pd.read_csv(gt_csv_path)

        # 이미지 파일명 기준으로 merge
        df_pred["img_name"] = df_pred["image"].apply(lambda x: Path(x).name)
        df_gt["img_name"] = df_gt["image"].apply(lambda x: Path(x).name)

        df_merged = pd.merge(df_gt, df_pred, on="img_name", how="inner")

        if len(df_merged) == 0:
            print("[WARN] No matched images between GT and prediction.")
        else:
            acc = (df_merged["gt_cls_id"] == df_merged["clf_cls"]).mean()
            print(f" Fusion classifier accuracy (on test): {acc:.4f} (N={len(df_merged)})")

            merged_csv_path = save_dir / "fusion_with_gt.csv"
            df_merged.to_csv(merged_csv_path, index=False)
            print(f" Saved fusion+GT CSV to: {merged_csv_path}")
    else:
        print("[INFO] GT CSV not found, skip accuracy. (expected: test_labels_gt.csv)")

    print("\n===== Done: run_fusion =====")


if __name__ == "__main__":
    main()
