# ===================================================================
# 1단계: DB 구축 - 과거 결함 이미지를 DINOv2로 특징 추출 후 PostgreSQL 저장
# ===================================================================

import torch
import os
import sys
from PIL import Image
import numpy as np
from collections import defaultdict
import glob
import psycopg2
from pgvector.psycopg2 import register_vector

from connection import initialize_project

initialize_project()

from DINOv2_FeatureMap import get_multiple_defect_boxes

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. DINOv2 모델 로드
print("DINOv2 모델 로딩 중...")
model_dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
model_dinov2.eval()
print("✓ DINOv2 모델 로드 완료!")


# 2. 이미지에서 DINOv2 특징 벡터 추출 함수
def extract_dinov2_features(image_path, model, device):
    """이미지 전체에 대한 DINOv2 특징 벡터 추출 (384차원)"""
    import torchvision.transforms as T

    img = Image.open(image_path).convert("RGB")

    transform = T.Compose([
        T.Resize((224, 224)),  # DINOv2 표준 입력 크기
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # CLS token 사용 (전역 특징)
        features = model(img_tensor)
        feature_vector = features.cpu().numpy().flatten()

    return feature_vector


# 3. DB에 저장할 데이터 수집
base_folder = r"C:\Users\hjchung\Desktop\RAG Train"
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']

defect_database = []

for defect_type in os.listdir(base_folder):
    defect_path = os.path.join(base_folder, defect_type)

    if not os.path.isdir(defect_path):
        continue

    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(defect_path, ext)))
        image_files.extend(glob.glob(os.path.join(defect_path, ext.upper())))

    print(f"\n📁 [{defect_type}] 처리 중... ({len(image_files)}개)")

    for idx, image_path in enumerate(image_files, 1):
        try:
            # DINOv2 특징 추출
            feature_vector = extract_dinov2_features(image_path, model_dinov2, device)

            defect_database.append({
                "id": len(defect_database) + 1,
                "defect_type": defect_type,
                "image_path": image_path,
                "filename": os.path.basename(image_path),
                "feature_vector": feature_vector,  # 384차원 벡터
                "vector_dim": len(feature_vector)
            })

            if idx % 10 == 0:
                print(f"  [{idx}/{len(image_files)}] 처리 완료")

        except Exception as e:
            print(f"  ✗ {os.path.basename(image_path)}: {e}")
            continue

print(f"\n✓ 총 {len(defect_database)}개의 이미지 특징 추출 완료!")


# [중요] 1. 함수 정의를 먼저 합니다.
def initialize_db():
    conn_params = {
        "host": "localhost",
        "database": "postgres",
        "user": "postgres",
        "password": "3510",
        "port": 5432
    }
    try:
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()

        # 1. 벡터 확장 기능 활성화
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # 2. [수정 포인트] 기존 테이블이 있으면 삭제 (차원 변경 및 중복 방지)
        print("기존 테이블을 삭제하고 1024차원용으로 새로 생성합니다...")
        cur.execute("DROP TABLE IF EXISTS semiconductor_defects;")

        # 3. 1024차원(ViT-L/14) 벡터 테이블 생성
        cur.execute("""
            CREATE TABLE semiconductor_defects (
                id serial PRIMARY KEY,
                image_name text,
                defect_type text,
                feature_vector vector(1024), 
                mask_path text
            );
        """)
        conn.commit()
        return conn, cur
    except Exception as e:
        print(f"✗ DB 초기화 실패: {e}")
        return None, None


if __name__ == "__main__":
    # 1. DB 먼저 연결
    conn, cur = initialize_db()
    if conn:
        register_vector(conn)

        # 2. 이미지 파일 리스트 확보
        base_dir = r"C:\Users\hjchung\Desktop\RAG Train"
        extensions = ('*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.bmp', '*.BMP')
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(base_dir, "**", ext), recursive=True))

        print(f"🚀 총 {len(image_files)}개 이미지 발견 (예상: 700개)")
        print(f"시작 전 기존 데이터를 삭제합니다...")
        cur.execute("TRUNCATE TABLE semiconductor_defects;")  # 중복 방지용 초기화
        conn.commit()

        for idx, image_path in enumerate(image_files):
            img_name = os.path.basename(image_path)
            try:
                # 1. 특징 추출
                feature_vector = extract_dinov2_features(image_path, model_dinov2, device)
                defect_type = os.path.basename(os.path.dirname(image_path))

                # 2. DB에 INSERT
                cur.execute("""
                            INSERT INTO semiconductor_defects (image_name, defect_type, feature_vector, mask_path)
                            VALUES (%s, %s, %s, %s);
                        """, (img_name, defect_type, feature_vector.tolist(), "none"))

                if idx % 50 == 0:
                    conn.commit()
                    print(f"  [{idx}/{len(image_files)}] 저장 중...")

            except Exception as e:
                # [핵심] 에러 발생 시 현재 트랜잭션을 롤백하여 다음 데이터가 들어갈 수 있게 함
                conn.rollback()
                print(f"  ✗ {img_name} 처리 실패 및 롤백 수행: {e}")

        conn.commit()  # 최종 저장
        cur.close()
        conn.close()
        print("✅ 모든 데이터가 PostgreSQL에 저장되었습니다! DBeaver를 확인하세요.")