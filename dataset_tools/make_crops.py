# dataset_tools/make_crops.py
from pathlib import Path
import os
import cv2
from glob import glob

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "driver"
DATA_ROOT = Path(os.environ.get("DDD_DATA_ROOT", str(DEFAULT_DATA_ROOT)))

IMAGE_DIRS = [
    DATA_ROOT / "train" / "images",
    DATA_ROOT / "valid" / "images",
]
LABEL_DIRS = [
    DATA_ROOT / "train" / "labels",
    DATA_ROOT / "valid" / "labels",
]

CROPS_ROOT = DATA_ROOT / "cls_crops_v2"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def process_split(img_dir: Path, lbl_dir: Path, split_name: str):
    out_root = CROPS_ROOT / split_name
    ensure_dir(out_root)

    img_paths = sorted(glob(str(img_dir / "*.jpg")))
    print(f"[INFO] {split_name}: {len(img_paths)} images")

    for img_path in img_paths:
        img_path = Path(img_path)
        stem = img_path.stem
        label_path = lbl_dir / f"{stem}.txt"
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        lines = label_path.read_text().strip().splitlines()
        if not lines:
            continue

        # 객체 1개라고 가정
        parts = lines[0].split()
        cid = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:5])

        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        cls_name = f"c{cid}"
        out_dir = out_root / cls_name
        ensure_dir(out_dir)

        out_path = out_dir / f"{stem}.jpg"
        cv2.imwrite(str(out_path), crop)


def main():
    process_split(IMAGE_DIRS[0], LABEL_DIRS[0], "train")
    process_split(IMAGE_DIRS[1], LABEL_DIRS[1], "valid")
    print("cls_crops_v2 생성 완료")


if __name__ == "__main__":
    main()
