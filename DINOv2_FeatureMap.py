# DINOv2의 feature map에서 결함 부위 찾아내어 SAM3용 box 좌표로 변환하는 함수

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from scipy.ndimage import label  # 보완점: 객체 분리 로직용


def get_multiple_defect_boxes(image_path, model_dinov2, device="cuda", threshold_q=0.97):
    # 1. 이미지 전처리
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    transform = T.Compose([
        T.Resize((448, 448)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 2. 특징 맵 추출
    with torch.no_grad():
        features = model_dinov2.get_intermediate_layers(img_tensor, n=1)[0]
        # 특징 맵 활성도 계산 (32x32 패치)
        attn = torch.norm(features, dim=-1).reshape(32, 32)

    # 3. 이진화 및 개별 객체 분리 (Connected Component Labeling)
    threshold = torch.quantile(attn, threshold_q)
    mask = (attn > threshold).cpu().numpy().astype(int)

    # 인접한 픽셀끼리 그룹화 (라벨링)
    labeled_array, num_features = label(mask)

    defect_boxes = []
    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled_array == i)
        if len(coords) < 2: continue  # 너무 작은 노이즈 제거

        y1, x1 = coords.min(axis=0) * 14
        y2, x2 = coords.max(axis=0) * 14

        # 원본 이미지 크기로 스케일링
        scale_x, scale_y = orig_w / 448, orig_h / 448
        box = [
            x1 * scale_x,
            y1 * scale_y,
            (x2 - x1 + 14) * scale_x,
            (y2 - y1 + 14) * scale_y
        ]
        defect_boxes.append(box)

    return defect_boxes  # 여러 개의 [x, y, w, h] 리스트 반환