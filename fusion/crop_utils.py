# fusion/crop_utils.py

"""
Fusion 단계에서 사용할 이미지/박스 유틸 함수들.
- bbox crop
- bbox 그리기
"""

from typing import Tuple
import numpy as np
import cv2


def crop_box_xyxy(
    image: np.ndarray,
    box_xyxy: Tuple[float, float, float, float],
    pad_ratio: float = 0.05,
):
    """
    xyxy 형식 박스를 기준으로 이미지 crop.
    - image: H x W x C (BGR)
    - box_xyxy: (x1, y1, x2, y2) in pixels
    - pad_ratio: bbox 크기의 비율로 패딩

    return: cropped image (numpy array, BGR)
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box_xyxy

    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * pad_ratio
    pad_y = bh * pad_ratio

    x1 = int(max(0, x1 - pad_x))
    y1 = int(max(0, y1 - pad_y))
    x2 = int(min(w - 1, x2 + pad_x))
    y2 = int(min(h - 1, y2 + pad_y))

    if x2 <= x1 or y2 <= y1:
        return image.copy()  # fallback

    return image[y1:y2, x1:x2]


def draw_bbox_with_label(
    image: np.ndarray,
    box_xyxy: Tuple[float, float, float, float],
    label: str,
    color=(0, 255, 0),
):
    """
    이미지에 bbox와 텍스트(label)를 그려서 반환.
    - image: H x W x C (BGR)
    - box_xyxy: (x1,y1,x2,y2)
    - label: 텍스트 (예: 'c1 - Texting')
    """
    x1, y1, x2, y2 = map(int, box_xyxy)
    out = image.copy()

    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    # 텍스트 박스
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

    text_bg_x2 = x1 + tw + 6
    text_bg_y2 = y1 - th - 6
    if text_bg_y2 < 0:
        text_bg_y2 = y1 + th + 6

    cv2.rectangle(
        out,
        (x1, y1),
        (text_bg_x2, text_bg_y2),
        color,
        -1,
    )

    text_org_y = y1 - 4 if text_bg_y2 != (y1 + th + 6) else y1 + th + 2
    cv2.putText(
        out,
        label,
        (x1 + 3, text_org_y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        lineType=cv2.LINE_AA,
    )

    return out
