# SAM3의 학습 데이터(Training Sample)로 구성
# Domain Adaptation 위해 (이미지, DINOv2가 찾은 박스, 실제 결함 마스크) 이 세 가지가 한 쌍이 되어야 함
# Defect 타입별 폴더 구조 지원 (crack, fabDefect, good, ink, mapout, particle, unknown)

import torch
import os
import sys
from PIL import Image
import glob
from collections import defaultdict

# 1. 경로 연결 (기존에 만든 connection.py 활용)
from connection import initialize_project

initialize_project()

# 필요한 함수 및 모델 빌더 import
from sam3 import build_sam3_image_model

# 보완된 다중 객체 추출 함수 가져오기
try:
    from DINOv2_FeatureMap import get_multiple_defect_boxes

    print("DINOv2 특징 추출 함수 임포트 성공!")
except ImportError:
    from DINOv2_FeatureMap import get_defect_box_from_dinov2

    get_multiple_defect_boxes = get_defect_box_from_dinov2

device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. DINOv2 모델 로드
print("DINOv2 모델 로딩 중...")
model_dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
model_dinov2.eval()
print("DINOv2 모델 로드 완료!")

# 3. SAM3 모델 로드
print("SAM3 모델 로딩 중...")
sam3_model = build_sam3_image_model(
    checkpoint_path=r"C:\Users\hjchung\Desktop\sam3\checkpoints\sam3.pt"
).to(device)
print("SAM3 모델 로드 완료!")

# 4. 서브폴더별 이미지 파일 수집
base_folder = r"C:\Users\hjchung\Desktop\RAG Train"
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']

# Defect 타입별로 이미지 경로 저장
defect_images = defaultdict(list)

# 각 서브폴더 탐색
for defect_type in os.listdir(base_folder):
    defect_path = os.path.join(base_folder, defect_type)

    # 폴더인지 확인
    if not os.path.isdir(defect_path):
        continue

    # 각 확장자별로 이미지 찾기
    for ext in image_extensions:
        images = glob.glob(os.path.join(defect_path, ext))
        images.extend(glob.glob(os.path.join(defect_path, ext.upper())))
        defect_images[defect_type].extend(images)

# 이미지가 없는 폴더 확인
if not defect_images:
    print(f"⚠️ 경고: {base_folder} 폴더에서 이미지를 찾을 수 없습니다!")
    sys.exit(1)

# 전체 통계 출력
total_images = sum(len(imgs) for imgs in defect_images.values())
print(f"\n{'=' * 70}")
print(f"총 {len(defect_images)}개의 Defect 타입에서 {total_images}개의 이미지를 찾았습니다.")
print(f"{'=' * 70}")

for defect_type, images in sorted(defect_images.items()):
    print(f"  • {defect_type:15s}: {len(images):3d}개 이미지")

print(f"{'=' * 70}\n")

# 5. Defect 타입별로 이미지 처리
all_training_samples = []
defect_statistics = defaultdict(lambda: {"processed": 0, "detected": 0, "failed": 0})

for defect_type, image_list in sorted(defect_images.items()):
    print(f"\n{'=' * 70}")
    print(f"📁 [{defect_type}] 폴더 처리 중... ({len(image_list)}개 이미지)")
    print(f"{'=' * 70}")

    for idx, image_path in enumerate(image_list, 1):
        filename = os.path.basename(image_path)
        print(f"  [{idx}/{len(image_list)}] {filename[:40]:40s} ", end="")

        try:
            # 결함 박스 추출
            defect_boxes = get_multiple_defect_boxes(image_path, model_dinov2, device)

            defect_statistics[defect_type]["processed"] += 1

            if defect_boxes:
                defect_statistics[defect_type]["detected"] += 1
                print(f"✓ {len(defect_boxes)}개 결함 탐지")

                # 첫 번째 박스 사용
                defect_box = defect_boxes[0]

                # SAM 3 학습 데이터셋 형태로 변환
                img = Image.open(image_path)
                w, h = img.size
                norm_box = [defect_box[0] / w, defect_box[1] / h,
                            defect_box[2] / w, defect_box[3] / h]

                training_sample = {
                    "defect_type": defect_type,  # Defect 타입 라벨
                    "image_path": image_path,
                    "filename": filename,
                    "image": img,
                    "prompt_box": norm_box,
                    "raw_box": defect_box,
                    "all_boxes": defect_boxes,  # 모든 검출 박스 저장
                    "label_mask": None  # 실제 정답 마스크로 교체 필요
                }

                all_training_samples.append(training_sample)

            else:
                print("✗ 결함 미탐지")

        except Exception as e:
            defect_statistics[defect_type]["failed"] += 1
            print(f"✗ 에러: {str(e)[:30]}")
            continue

# 6. 최종 결과 요약
print(f"\n\n{'=' * 70}")
print(f"🎯 처리 완료 - 총 {len(all_training_samples)}개의 학습 샘플 생성")
print(f"{'=' * 70}\n")

print(f"{'Defect 타입':<15s} {'처리':>6s} {'탐지':>6s} {'실패':>6s} {'탐지율':>8s}")
print(f"{'-' * 70}")

for defect_type in sorted(defect_statistics.keys()):
    stats = defect_statistics[defect_type]
    detection_rate = (stats["detected"] / stats["processed"] * 100) if stats["processed"] > 0 else 0
    print(f"{defect_type:<15s} {stats['processed']:>6d} {stats['detected']:>6d} "
          f"{stats['failed']:>6d} {detection_rate:>7.1f}%")

print(f"{'=' * 70}\n")

# 7. Defect 타입별 샘플 수 확인
samples_by_type = defaultdict(int)
for sample in all_training_samples:
    samples_by_type[sample["defect_type"]] += 1

print("📊 학습 샘플 분포:")
for defect_type, count in sorted(samples_by_type.items()):
    print(f"  • {defect_type:15s}: {count:3d}개")

# 8. 예시: 첫 번째 샘플 정보 출력
if all_training_samples:
    print(f"\n{'=' * 70}")
    print("📝 샘플 예시 (첫 번째 이미지):")
    print(f"{'=' * 70}")
    sample = all_training_samples[0]
    print(f"  Defect 타입     : {sample['defect_type']}")
    print(f"  파일명          : {sample['filename']}")
    print(f"  이미지 크기     : {sample['image'].size}")
    print(f"  탐지된 박스 수  : {len(sample['all_boxes'])}")
    print(f"  정규화 박스     : {[f'{x:.3f}' for x in sample['prompt_box']]}")
    print(f"  원본 박스 [x,y,w,h]: {[f'{x:.1f}' for x in sample['raw_box']]}")
    print(f"{'=' * 70}")

# 9. (선택) 학습 샘플을 파일로 저장
# import pickle
# with open('training_samples.pkl', 'wb') as f:
#     pickle.dump(all_training_samples, f)
# print("\n✓ 학습 샘플이 'training_samples.pkl'에 저장되었습니다.")