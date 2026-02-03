# ===================================================================
# 2단계: 실시간 검사 - 신규 이미지 → SAM3 Mask → DINOv2 → Top-5 검색
# ===================================================================

import torch
import os
import sys
from PIL import Image
import numpy as np
import torchvision.transforms as T

from connection import initialize_project

initialize_project()

from sam3 import build_sam3_image_model
from DINOv2_FeatureMap import get_multiple_defect_boxes

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 모델 로드
print("=" * 70)
print("모델 로딩 중...")
print("=" * 70)

# DINOv2 모델
model_dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
model_dinov2.eval()
print("✓ DINOv2 모델 로드 완료!")

# SAM3 모델
sam3_model = build_sam3_image_model(
    checkpoint_path=r"C:\Users\hjchung\Desktop\sam3\checkpoints\sam3.pt"
).to(device)
sam3_model.eval()
print("✓ SAM3 모델 로드 완료!")


# 2. 전체 파이프라인 함수
def process_new_defect_image(image_path, dinov2_model, sam3_model, device):
    """
    신규 결함 이미지 처리 파이프라인
    1) DINOv2로 결함 위치(box) 찾기
    2) SAM3로 정밀한 Mask 생성
    3) Mask 영역에서 DINOv2 특징 추출
    4) DB에서 Top-5 유사 이미지 검색
    """

    print(f"\n{'=' * 70}")
    print(f"📸 신규 이미지 분석: {os.path.basename(image_path)}")
    print(f"{'=' * 70}")

    # Step 1: DINOv2로 결함 박스 찾기 (Tiling)
    print("\n[Step 1] DINOv2로 결함 위치 찾기...")
    defect_boxes = get_multiple_defect_boxes(image_path, dinov2_model, device)

    if not defect_boxes:
        print("✗ 결함이 탐지되지 않았습니다.")
        return None

    print(f"✓ {len(defect_boxes)}개의 결함 영역 탐지!")

    # 첫 번째 박스 사용 (또는 가장 큰 박스 선택 가능)
    defect_box = defect_boxes[0]
    print(f"  - 선택된 박스 [x, y, w, h]: {[f'{x:.1f}' for x in defect_box]}")

    # Step 2: SAM3로 정밀한 Mask 생성
    print("\n[Step 2] SAM3로 정밀 Mask 생성...")

    img = Image.open(image_path).convert("RGB")
    img_width, img_height = img.size

    # SAM3 입력 형식으로 변환
    img_array = np.array(img)

    # Box 좌표를 SAM3 형식으로 변환 [x1, y1, x2, y2]
    x, y, w, h = defect_box
    box_sam = np.array([x, y, x + w, y + h])

    with torch.no_grad():
        # SAM3에 이미지와 박스 전달
        sam3_model.set_image(img_array)

        masks, scores, _ = sam3_model.predict(
            point_coords=None,
            point_labels=None,
            box=box_sam[None, :],  # [1, 4] 형태
            multimask_output=False
        )

    # 가장 좋은 마스크 선택
    best_mask = masks[0]  # [H, W]
    print(f"✓ Mask 생성 완료! (크기: {best_mask.shape})")

    # Step 3: Mask 영역에서 DINOv2 특징 추출
    print("\n[Step 3] Mask 영역에서 특징 추출...")

    # Mask를 적용한 이미지 생성
    masked_img = img_array.copy()
    masked_img[~best_mask] = 0  # Mask 외부는 검은색

    masked_pil = Image.fromarray(masked_img)

    # DINOv2로 특징 추출
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(masked_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        features = dinov2_model(img_tensor)
        query_vector = features.cpu().numpy().flatten()

    print(f"✓ 특징 벡터 추출 완료! (차원: {len(query_vector)})")

    # Step 4: PostgreSQL에서 Top-5 유사 이미지 검색
    print("\n[Step 4] DB에서 유사 이미지 Top-5 검색...")

    try:
        import psycopg2

        conn = psycopg2.connect(
            host="localhost",
            database="defect_db",
            user="postgres",
            password="your_password"
        )
        cur = conn.cursor()

        # 코사인 유사도로 Top-5 검색
        cur.execute("""
            SELECT 
                id,
                defect_type,
                filename,
                1 - (feature_vector <=> %s::vector) AS similarity
            FROM defect_images
            ORDER BY feature_vector <=> %s::vector
            LIMIT 5;
        """, (query_vector.tolist(), query_vector.tolist()))

        top5_results = cur.fetchall()

        cur.close()
        conn.close()

        print(f"\n{'=' * 70}")
        print("🏆 Top-5 유사 이미지 검색 결과")
        print(f"{'=' * 70}")
        print(f"{'순위':<6} {'Defect 타입':<15} {'파일명':<30} {'유사도':<10}")
        print("-" * 70)

        predicted_defect = None
        for rank, (img_id, defect_type, filename, similarity) in enumerate(top5_results, 1):
            print(f"{rank:<6} {defect_type:<15} {filename:<30} {similarity:.4f}")
            if rank == 1:
                predicted_defect = defect_type

        print(f"{'=' * 70}")
        print(f"\n✅ 최종 판정: [{predicted_defect}] 타입으로 분류됨")
        print(f"{'=' * 70}\n")

        return {
            "image_path": image_path,
            "defect_box": defect_box,
            "mask": best_mask,
            "feature_vector": query_vector,
            "predicted_defect": predicted_defect,
            "top5_results": top5_results
        }

    except ImportError:
        print("⚠️ PostgreSQL 연결 실패. 로컬 데이터베이스 사용...")

        import pickle
        with open('defect_database.pkl', 'rb') as f:
            defect_db = pickle.load(f)

        # 코사인 유사도 계산
        from scipy.spatial.distance import cosine

        similarities = []
        for item in defect_db:
            sim = 1 - cosine(query_vector, item['feature_vector'])
            similarities.append({
                'defect_type': item['defect_type'],
                'filename': item['filename'],
                'similarity': sim
            })

        # Top-5 정렬
        top5 = sorted(similarities, key=lambda x: x['similarity'], reverse=True)[:5]

        print(f"\n{'=' * 70}")
        print("🏆 Top-5 유사 이미지 검색 결과 (로컬 DB)")
        print(f"{'=' * 70}")
        print(f"{'순위':<6} {'Defect 타입':<15} {'파일명':<30} {'유사도':<10}")
        print("-" * 70)

        for rank, item in enumerate(top5, 1):
            print(f"{rank:<6} {item['defect_type']:<15} {item['filename']:<30} {item['similarity']:.4f}")

        predicted_defect = top5[0]['defect_type']

        print(f"{'=' * 70}")
        print(f"\n✅ 최종 판정: [{predicted_defect}] 타입으로 분류됨")
        print(f"{'=' * 70}\n")

        return {
            "image_path": image_path,
            "defect_box": defect_box,
            "mask": best_mask,
            "feature_vector": query_vector,
            "predicted_defect": predicted_defect,
            "top5_results": top5
        }

    except Exception as e:
        print(f"✗ 검색 실패: {e}")
        return None


# 3. 테스트 실행
if __name__ == "__main__":
    # 테스트할 신규 이미지 경로 (실제 경로로 변경)
    test_image = r"C:\Users\hjchung\Desktop\test_defect.jpg"

    if os.path.exists(test_image):
        result = process_new_defect_image(test_image, model_dinov2, sam3_model, device)

        if result:
            print("\n✓ 파이프라인 실행 완료!")

            # (선택) 결과 저장
            # import pickle
            # with open('inference_result.pkl', 'wb') as f:
            #     pickle.dump(result, f)
    else:
        print(f"⚠️ 테스트 이미지를 찾을 수 없습니다: {test_image}")
        print("test_image 변수를 실제 이미지 경로로 수정하세요.")