import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


def visualize_defect_results(image_path, defect_box, best_mask):
    """
    DINOv2 박스와 SAM3 마스크를 이미지 위에 시각화합니다.
    """
    # 1. 원본 이미지 로드
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.title("Step 1: DINOv2 Detection (Box)")

    # 2. DINOv2 Bounding Box 표시 (빨간색)
    x, y, w, h = [int(v) for v in defect_box]
    box_img = image.copy()
    cv2.rectangle(box_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
    plt.imshow(box_img)
    plt.axis('off')

    # 3. SAM3 Segmentation Mask 표시 (파란색 오버레이)
    plt.subplot(1, 2, 2)
    plt.title("Step 2: SAM3 Segmentation (Mask)")

    mask_overlay = image.copy()
    if best_mask is not None and best_mask.any():
        # 마스크 영역을 파란색으로 칠함
        mask_overlay[best_mask] = [0, 0, 255]
        # 원본과 합성 (불투명도 0.5)
        combined = cv2.addWeighted(image, 0.7, mask_overlay, 0.3, 0)
        plt.imshow(combined)
    else:
        plt.imshow(image)
        plt.text(10, 30, "Mask Not Found", color='red', fontsize=15, backgroundcolor='white')

    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Inference.py에서 분석이 끝난 후 이 함수를 호출하도록 연결하세요.