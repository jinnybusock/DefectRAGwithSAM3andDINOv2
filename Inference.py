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
from sam3 import build_sam3_image_model
from show_image import visualize_defect_results

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 모델 로드
print("=" * 70)
print("모델 로딩 중...")
print("=" * 70)

# DINOv2 모델
device= "cuda" if torch.cuda.is_available() else "cpu"
model_dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)

# 학습된 대조 학습 가중치 로드
trained_weights = "dinov2_semicon_contrastive.pt"
if os.path.exists(trained_weights):
    model_dinov2.load_state_dict(torch.load(trained_weights, map_location=device))
    print(f"🔥 대조 학습 가중치({trained_weights}) 로드 완료!")
else:
    print("⚠️ 학습된 가중치가 없어 기본 DINOv2 모델을 사용합니다.")

model_dinov2.eval()
print("✓ DINOv2 모델 로드 완료!")

# SAM3 모델 로드
sam3_checkpoint = r"C:\Users\hjchung\Desktop\sam3\checkpoints\sam3.pt"
sam3_model = build_sam3_image_model(checkpoint_path=sam3_checkpoint).to(device)
print("✓ SAM3 모델 로드 완료!")

# 3. 전체 파이프라인 함수
def process_new_defect_image(image_path, dinov2_model, predictor_sam,device, threshold):
    """
    신규 결함 이미지 처리 파이프라인 (최종 수정본)
    """
    # 1. 초기 변수 설정
    best_mask = None
    defect_box = None

    print(f"\n{'=' * 70}")
    print(f"📸 분석 시작: {os.path.basename(image_path)} (임계값: {threshold})")
    print(f"{'=' * 70}")

    # [Step 1] DINOv2로 결함 위치(Box) 찾기
    print("\n[Step 1] DINOv2로 결함 위치 탐지 중...")
    found_boxes = get_multiple_defect_boxes(image_path, dinov2_model, device)

    if not found_boxes:
        print("✗ 결함이 탐지되지 않았습니다.")
        return None

    num_found= len(found_boxes)
    # 첫 번째 결함 영역 선택
    defect_box = found_boxes[0]
    print(f"✓ {num_found}개의 영역 중 대표 영역 선택: {defect_box}")

    # [Step 2] SAM3로 정밀 Mask 생성
    print("\n[Step 2] SAM3로 정밀 Mask 생성 중...")
    raw_image = Image.open(image_path).convert('RGB')
    image_np = np.array(raw_image)

    try:
        x, y, w, h = defect_box
        # SAM3 입력 형식 [x1, y1, x2, y2]
        box_xyxy = torch.tensor([x, y, x + w, y + h], device=device).unsqueeze(0) # 텐서화

        with torch.no_grad():
            # [수정] Predictor 대신 모델의 predict 메소드를 직접 호출
            masks, scores = sam3_model.predict(
                image=image_np,
                boxes=box_xyxy,
                multimask_output=False
            )
            best_mask = masks[0]
            print(f"✓ SAM3 Mask 생성 성공! (신뢰도: {scores[0]:.3f})")

    except Exception as e:
        print(f"⚠️ SAM3 실패 ({e}), 박스 영역을 마스크로 대체합니다.")
        best_mask = np.zeros(image_np.shape[:2], dtype=bool)

    # [Step 3] 특징 추출 (들여쓰기 수정: SAM3 성공/실패 여부와 상관없이 실행)
    print("\n[Step 3] DINOv2 특징 추출 (DB 정합성 유지)...")
    try:
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(raw_image).unsqueeze(0).to(device)

        with torch.no_grad():
            # features 변수 정의 및 1024차원 추출
            features = dinov2_model(img_tensor)
            query_vector = features.cpu().numpy().flatten()
        print(f"✓ 특징 추출 완료! (차원: {len(query_vector)})")
    except Exception as e:
        print(f"✗ 특징 추출 실패: {e}")
        return None

    # [Step 4] PostgreSQL DB 검색 및 최종 판정
    print("\n[Step 4] DB 유사 이미지 검색 및 RAG 판정...")
    try:
        import psycopg2
        from collections import defaultdict

        conn = psycopg2.connect(
            host="localhost", port="5432", dbname="postgres",
            user="postgres", password="3510"
        )
        cur = conn.cursor()

        # SQL 문법 교정 및 Threshold 적용
        cur.execute("""
            SELECT * FROM (
                SELECT DISTINCT ON (image_name) id, defect_type, image_name, 1 - (feature_vector <=> %s::vector) AS similarity
                FROM semiconductor_defects
                ORDER BY image_name, 1- (feature_vector <=> %s::vector) DESC
            ) sub
            WHERE similarity >= %s
            ORDER BY similarity DESC
            LIMIT 5;
        """, (query_vector.tolist(), query_vector.tolist(), threshold))

        top5_results = cur.fetchall()
        cur.close()
        conn.close()

        if not top5_results:
            print(f"⚠️ 임계값({threshold}) 이상의 유사 사례가 DB에 없습니다.")
            return None

        # 유사도 및 건수 합산 로직
        defect_scores = defaultdict(float)
        defect_counts = defaultdict(int)

        for row in top5_results:
            d_type = row[1]  # defect_type
            similarity = row[3]  # similarity
            defect_scores[d_type] += similarity
            defect_counts[d_type] += 1

        # 점수 기반 최종 승자 결정
        predicted_defect = max(defect_scores.items(), key=lambda x: x[1])[0]

        # 결과 리포트 출력
        print(f"\n{'=' * 70}")
        print(f"🏆 최종 판정: [{predicted_defect}] 타입")
        print(f"{'=' * 70}")
        for rank, row in enumerate(top5_results, 1):
            print(f"{rank}위: {row[1]:<12} | {row[2]:<35} | 유사도: {row[3]:.4f}")
        print("-" * 70)
        for d_type, total_score in sorted(defect_scores.items(), key=lambda x: x[1], reverse=True):
            count = defect_counts[d_type]
            marker = "👑" if d_type == predicted_defect else "  "
            print(f"{marker} {d_type:<12} : 합산 유사도 {total_score:.4f} ({count}건)")

        try:
            visualize_defect_results(image_path, defect_box, best_mask)
        except ImportError:
            print("⚠️ show_image.py 파일을 찾을 수 없어 시각화를 건너뜁니다.")

        return {"predicted_defect": predicted_defect}

    except Exception as e:
        print(f"✗ DB 검색 오류: {e}")
        return None

# 4. 테스트 실행
if __name__ == "__main__":
    # 사용자로부터 유사도 threshold 값 받기
    try:
        user_threshold= float(input("\n⚙️ 검색 유사도 임계값을 설정하세요 (0.0 ~ 1.0, 권장 0.7): ").strip())
    except ValueError:
        user_threshold = 0.7 # 입력 오류 시 기본 값
        print(f"⚠️ 올바른 숫자가 아닙니다. 기본값 {user_threshold}로 설정합니다.")

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

                # predictor와 threshold 인자를 명시적으로 전달
                result = process_new_defect_image(
                    full_path,
                    model_dinov2,
                    sam3_model,
                    device,
                    user_threshold  # threshold 값 전달
                )

                if result:
                    print(f"✅ [{target_file}] 분석 및 시각화 완료!")
                else:
                    print("\n⚠️ 분석 중 문제가 발생했습니다.")
            else:
                print(f"\n⚠️ '{user_input}'와 일치하는 이미지를 찾을 수 없습니다.")

        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback

            traceback.print_exc()

        print("\n" + "-" * 70)