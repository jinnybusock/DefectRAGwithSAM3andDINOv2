import os
import sys
from connection import initialize_project

initialize_project()

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as T
from sam3 import build_sam3_image_model
from DINOv2_FeatureMap import get_multiple_defect_boxes
from collections import defaultdict

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 모델 로드
print("=" * 70)
print("모델 로딩 중...")
print("=" * 70)

# DINOv2 모델
model_dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
model_dinov2.eval()
print("✓ DINOv2 모델 로드 완료!")

# SAM3 모델 로드
sam3_checkpoint = r"C:\Users\hjchung\Desktop\sam3\checkpoints\sam3.pt"
sam3_model = build_sam3_image_model(checkpoint_path=sam3_checkpoint).to(device)
sam3_model.eval()
print("✓ SAM3 모델 로드 완료!")

# 2. DINOv2 특징 추출 함수
def extract_dinov2_features_from_mask(image_np, mask, dinov2_model, device):
    """Mask 영역에서 DINOv2 특징 추출"""

    # Mask 적용한 이미지 생성
    masked_img = image_np.copy()
    masked_img[~mask] = 0  # Mask 외부는 검은색

    # PIL Image로 변환
    masked_pil = Image.fromarray(masked_img.astype('uint8'))

    # DINOv2 전처리
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(masked_pil).unsqueeze(0).to(device)

    # 특징 벡터 추출
    with torch.no_grad():
        features = dinov2_model(img_tensor)
        feature_vector = features.cpu().numpy().flatten()

    return feature_vector

# 3. 전체 파이프라인 함수
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

    # 함수 시작 시 변수 초기화
    best_mask= None
    defect_box= None

    # Step 1: DINOv2로 결함 박스 찾기
    print("\n[Step 1] DINOv2로 결함 위치 찾기...")
    defect_boxes = get_multiple_defect_boxes(image_path, dinov2_model, device)
    defect_box= defect_boxes[0]

    if not defect_boxes:
        print("✗ 결함이 탐지되지 않았습니다.")
        return None

    print(f"✓ {len(defect_boxes)}개의 결함 영역 탐지!")

    # Step 2: SAM3로 정밀 Mask 생성 (시각화/확인용으로 유지)
    print("\n[Step 2] SAM3로 정밀 Mask 생성...")

    try:
        # PIL Image 전역에서 가져와 사용
        raw_image = Image.open(image_path).convert('RGB')
        image_np = np.array(raw_image)

        # SAM3 입력 형식: [x1, y1, x2, y2]
        x, y, w, h = defect_box
        # SAM3 입력 형식에 맞게 [x1, y1, x2, y2]로 변환하여 텐서화
        box_tensor = torch.tensor([x, y, x + w, y + h], device=device).unsqueeze(0)

        with torch.no_grad():
            # Sam3Image 모델은 set_image 대신 이미지를 직접 입력받습니다.
            # 모델의 forward 또는 predict_masks 메소드를 사용해야 합니다.
            # 여기서는 가장 일반적인 모델 직접 호출 방식을 적용합니다.
            masks, scores = sam3_model.predict(
                image=image_np,
                boxes=box_tensor,
                multimask_output=False
            )
            best_mask = masks[0]
            print(f"✓ Mask 생성 완료! (점수: {scores[0]:.3f})")

    except Exception as e:
        print(f"⚠️ SAM3 마스크 생성 실패, 박스 영역을 사용합니다.")
        # 대안으로 박스 영역을 마스크로 만듦
        best_mask = np.zeros(image_np.shape[:2], dtype=bool)

        # [Step 3] 특징 추출 - 마스크 씌우지 않고 원본 이미지에서 바로 특징 추출
        print("\n[Step 3] 원본 이미지에서 특징 추출 (DB 정합성 유지)...")
        try:
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            img_tensor = transform(raw_image).unsqueeze(0).to(device)

            with torch.no_grad():
                # [수정] features 변수를 먼저 정의해야 합니다.
                features = dinov2_model(img_tensor)
                query_vector = features.cpu().numpy().flatten()

            print(f"✓ 특징 벡터 추출 완료! (차원: {len(query_vector)})")
        except Exception as e:
            print(f"✗ 특징 추출 중 오류 발생: {e}")
            return None

        # Step 4: PostgreSQL에서 Top-5 유사 이미지 검색
        print("\n[Step 4] DB에서 유사 이미지 Top-5 검색...")

        try:
            import psycopg2
            from collections import defaultdict

            conn = psycopg2.connect(
                host="localhost", port="5432", dbname="postgres",
                user="postgres", password="3510"
            )
            cur = conn.cursor()

            cur.execute("""
                SELECT id, defect_type, image_name, 1 - (feature_vector <=> %s::vector) AS similarity
                FROM semiconductor_defects
                ORDER BY feature_vector <=> %s::vector LIMIT 5;
            """, (query_vector.tolist(), query_vector.tolist()))

            top5_results = cur.fetchall()
            cur.close()
            conn.close()

            if not top5_results:
                print("⚠️ DB에서 유사한 이미지를 찾을 수 없습니다.")
                return None

            # --- [수정 포인트] 합산 로직을 먼저 수행한 후 max()를 호출합니다 ---
            defect_scores = defaultdict(float)
            defect_counts = defaultdict(int)

            for row in top5_results:
                d_type = row[1]  # defect_type
                similarity = row[3]  # similarity
                defect_scores[d_type] += similarity
                defect_counts[d_type] += 1

            # 이제 데이터가 채워졌으므로 max() 에러가 발생하지 않습니다.
            predicted_defect = max(defect_scores.items(), key=lambda x: x[1])[0]

            # 결과 출력
            print(f"\n{'=' * 70}")
            print("🏆 Top-5 유사 이미지 검색 결과 (RAG 시스템)")
            print(f"{'=' * 70}")
            print(f"{'순위':<6} {'Defect 타입':<15} {'파일명':<35} {'유사도':<10}")
            print("-" * 70)

            for rank, row in enumerate(top5_results, 1):
                print(f"{rank:<6} {row[1]:<15} {row[2]:<35} {row[3]:.4f}")

            print(f"{'=' * 70}")
            print(f"✅ 최종 판정: [{predicted_defect}] 타입 (유사도 합산 결과)")
            print(f"{'=' * 70}")

            # 타입별 합산 점수 상세 출력 (👑 표시)
            for d_type, total_score in sorted(defect_scores.items(), key=lambda x: x[1], reverse=True):
                count = defect_counts[d_type]
                marker = "👑" if d_type == predicted_defect else "  "
                print(f"{marker} {d_type:<12} : 합산={total_score:.4f} (건수={count})")

            return {
                "image_path": image_path,
                "defect_box": defect_box,
                "mask": best_mask if (best_mask is not None and best_mask.any()) else np.array([]),
                "feature_vector": query_vector,
                "predicted_defect": predicted_defect,
                "top5_results": top5_results
            }

        except Exception as e:
            print(f"✗ 분석/검색 중 오류 발생: {e}")
            return None

# 4. 테스트 실행
if __name__ == "__main__":
    test_folder = r"C:\Users\hjchung\Desktop\RAG Test"

    if not os.path.exists(test_folder):
        print(f"✗ 폴더를 찾을 수 없습니다: {test_folder}")
        sys.exit(1)

    while True:
        # 폴더 내 이미지 리스트
        images = [f for f in os.listdir(test_folder)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        if not images:
            print(f"⚠️ {test_folder} 폴더에 이미지가 없습니다!")
            break

        print("\n" + "=" * 70)
        print(f"📂 테스트 폴더: {test_folder}")
        print(f"📊 이미지 수: {len(images)}개")
        print("=" * 70)

        print("\n💡 종료하려면 'q' 또는 'exit' 입력")

        # 사용자 입력
        user_input = input("\n👉 테스트할 파일명 또는 번호: ").strip()

        # 종료 조건
        if user_input.lower() in ['q', 'quit', 'exit', '종료']:
            print("\n👋 프로그램을 종료합니다.")
            break

        # 파일 선택
        target_file = None

        try:
            # 번호 입력
            if user_input.isdigit():
                idx = int(user_input)
                if 0 <= idx < len(images):
                    target_file = images[idx]
            # 파일명 입력
            else:
                if user_input in images:
                    target_file = user_input
                else:
                    # 확장자 없이 입력한 경우
                    for img in images:
                        if os.path.splitext(img)[0] == user_input:
                            target_file = img
                            break

            if target_file:
                full_path = os.path.join(test_folder, target_file)
                print(f"\n🔍 분석 시작: {target_file}")
                print("-" * 70)

                # 파이프라인 실행
                result = process_new_defect_image(
                    full_path, model_dinov2, sam3_model, device
                )

                if result:
                    print("\n✅ 분석 완료!")
                else:
                    print("\n⚠️ 분석 중 문제가 발생했습니다.")
            else:
                print(f"\n⚠️ '{user_input}'와 일치하는 이미지를 찾을 수 없습니다.")

        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback

            traceback.print_exc()

        print("\n" + "-" * 70)