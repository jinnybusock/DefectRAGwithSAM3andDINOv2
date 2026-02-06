import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import os
import random
from connection import initialize_project

# 프로젝트 경로 초기화
initialize_project()


class SemiconductorTripletDataset(Dataset):
    def __init__(self, base_path, transform=None):
        self.base_path = base_path
        self.transform = transform
        self.classes = [c for c in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, c))]

        # 'good' 클래스 이미지 리스트 확보 (Anchor & Positive 용)
        self.good_images = [os.path.join(base_path, 'good', f)
                            for f in os.listdir(os.path.join(base_path, 'good'))
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # 결함 클래스들 (Negative 용)
        self.defect_classes = [c for c in self.classes if c != 'good']

    def __len__(self):
        return len(self.good_images)

    def __getitem__(self, idx):
        # 1. Anchor: 현재 'good' 이미지
        anchor_path = self.good_images[idx]

        # 2. Positive: 또 다른 랜덤 'good' 이미지
        pos_path = random.choice(self.good_images)
        while pos_path == anchor_path:
            pos_path = random.choice(self.good_images)

        # 3. Negative: 결함 클래스 중 하나에서 랜덤 선택
        neg_class = random.choice(self.defect_classes)
        neg_folder = os.path.join(self.base_path, neg_class)
        neg_path = os.path.join(neg_folder, random.choice(os.listdir(neg_folder)))

        # 이미지 로드 및 전처리
        anchor = self.transform(Image.open(anchor_path).convert('RGB'))
        positive = self.transform(Image.open(pos_path).convert('RGB'))
        negative = self.transform(Image.open(neg_path).convert('RGB'))

        return anchor, positive, negative


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_train_path = r"C:\Users\hjchung\Desktop\RAG Train"

    # 1. 모델 로드 및 어댑터 부착
    print("DINOv2 ViT-L/14 로딩 중...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)

    # 가중치 고정 여부 결정 (Backbone도 살짝 튜닝하는 것이 성능에 좋음)
    for param in dinov2.parameters():
        param.requires_grad = True

    # 2. 전처리 설정
    transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. 데이터 로더
    dataset = SemiconductorTripletDataset(base_train_path, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 4. 손실 함수 및 최적화 (TripletMarginLoss 사용)
    # $L(a, p, n) = \max(d(a, p) - d(a, n) + \text{margin}, 0)$
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.AdamW(dinov2.parameters(), lr=1e-5)

    print(f"🚀 대조 학습 시작 (이미지: {len(dataset)}쌍)...")

    num_epochs = 10

    accumulation_steps = 4     # 4번 모아서 업데이트
    for epoch in range(num_epochs):
        scaler= torch.amp.GradScaler('cuda')
        model_loss = 0.0
        optimizer.zero_grad()     # 루프 밖으로 이동

        for i, (anc, pos, neg) in enumerate(loader):
            anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)

            with torch.amp.autocast('cuda'):     # 자동 정밀도 조절
                # 특징 추출 및 손실 계산
                e_anc= dinov2(anc)
                e_pos = dinov2(pos)
                e_neg = dinov2(neg)

                loss= criterion(e_anc, e_pos, e_neg)/ accumulation_steps     # 손실 나눗셈

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 누적 횟수가 차면 가중치 업데이트
            if (i+ 1)%accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            optimizer.zero_grad()

            # 특징 추출
            e_anc = dinov2(anc)
            e_pos = dinov2(pos)
            e_neg = dinov2(neg)

            loss = criterion(e_anc, e_pos, e_neg)
            loss.backward()
            optimizer.step()

            model_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {model_loss / len(loader):.4f}")

    # 5. 모델 저장
    save_path = "dinov2_semicon_contrastive.pt"
    torch.save(dinov2.state_dict(), save_path)
    print(f"✅ 학습 완료! 가중치 저장됨: {save_path}")


if __name__ == "__main__":
    train()