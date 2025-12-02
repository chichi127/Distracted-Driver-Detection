# dataset_tools/merge_c2c3_labels.py
from pathlib import Path
import os
from glob import glob

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "driver"
DATA_ROOT = Path(os.environ.get("DDD_DATA_ROOT", str(DEFAULT_DATA_ROOT)))

LABEL_DIRS = [
    DATA_ROOT / "train" / "labels",
    DATA_ROOT / "valid" / "labels",
    DATA_ROOT / "test" / "labels",   # test 라벨도 있다면
]

MERGE_TARGET = 2   # 최종 전화 클래스 (c2)
LEFT_ID  = 3       # c3: Right call (예전)
RIGHT_ID = 2       # c2: Left call (예전)


def merge_one_file(path: Path):
    text = path.read_text().strip().splitlines()
    new_lines = []
    changed = False

    for line in text:
        if not line.strip():
            continue
        parts = line.split()
        cid = int(parts[0])
        if cid in [LEFT_ID, RIGHT_ID]:
            parts[0] = str(MERGE_TARGET)
            changed = True
        new_lines.append(" ".join(parts))

    if changed:
        path.write_text("\n".join(new_lines) + "\n")


def main():
    print("===== c2/c3 → c2 라벨 통합 시작 =====")
    for d in LABEL_DIRS:
        if not d.exists():
            print(f"[WARN] 라벨 디렉토리 없음: {d}")
            continue
        files = sorted(glob(str(d / "*.txt")))
        print(f"[INFO] {d} : {len(files)}개")
        for p in files:
            merge_one_file(Path(p))
    print("완료")


if __name__ == "__main__":
    main()
