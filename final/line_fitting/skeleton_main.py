import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm # 진행률 표시줄 생성을 위한 라이브러리
import torch.nn.functional as F
import wandb # 실험 추적 및 시각화를 위한 라이브러리
import math
from skeleton_unet import Skeleton_UNet # 커스텀 Skeleton U-Net 모델
from skeleton_dataset import IntersectionDataset # 데이터셋 클래스
from skeleton_utils import calc_dice, calc_angle_errors # 성능 평가 유틸리티 함수
from torch.nn.utils import clip_grad_norm_ # 그래디언트 클리핑을 위한 유틸리티


class WeightedFocalLoss(nn.Module):
    """
    가중치가 적용된 Focal Loss 구현
    클래스 불균형 문제를 해결하기 위한 손실 함수
    
    수식:
    FL(pt) = -αt(1-pt)^γ * log(pt)

    Args:
        alpha: positive 클래스에 대한 가중치 (기본값: 0.01), negative는 1-alpha
        gamma: focusing 파라미터 (기본값: 3) - 높을수록 어려운 샘플에 더 집중

    특징:
    - 예측이 쉬운 샘플의 영향을 감소시키고 어려운 샘플에 집중
    - 클래스 불균형 문제를 해결하기 위한 가중치 적용
    - NaN/Inf 방지를 위한 안정성 처리
    """
    def __init__(self, alpha=0.01, gamma=3):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]) # positive와 negative 클래스에 대한 가중치 설정
        self.gamma = gamma # focusing 파라미터 (높을수록 어려운 샘플에 더 집중)

    def forward(self, preds, targets):
        # BCE Loss를 기반으로 Focal Loss 계산
        BCE_loss = F.binary_cross_entropy(preds.view(-1), targets.view(-1).float(), reduction='none')
        targets = targets.type(torch.long) # 타겟을 long 타입으로 변환
        self.alpha = self.alpha.to(preds.device) # 가중치 텐서를 현재 디바이스로 이동
        
        # 클래스별 가중치 적용
        at = self.alpha.gather(0, targets.data.view(-1)) # 각 샘플에 해당하는 가중치 선택
        pt = torch.exp(-BCE_loss) # BCE Loss를 예측 확률로 변환

        # Focal Loss 계산: -(1-pt)^gamma * log(pt)
        F_loss = at*(1-pt)**self.gamma * BCE_loss # focusing 파라미터를 사용하여 어려운 샘플 강조
        F_loss = F_loss.mean() # 배치의 평균 손실 계산

        # NaN이나 Inf 방지를 위함
        if math.isnan(F_loss) or math.isinf(F_loss):
            F_loss = torch.zeros(1).to(preds.device)

        return F_loss

class DiceLoss(nn.Module):
    """
    Dice Loss 구현
    세그멘테이션 작업에서 클래스 불균형 문제를 해결하기 위한 손실 함수
    
    수식:
    DiceLoss = 1 - (2|X∩Y| + ε)/(|X| + |Y| + ε)
    
    Args:
        smooth (float): 분모가 0이 되는 것을 방지하기 위한 값
    
    특징:
    - IoU와 유사하지만 기울기가 더 안정적
    - 클래스 불균형에 강건함
    - 배치 단위로 계산되어 메모리 효율적
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth # 0으로 나누는 것을 방지하기 위한 값

    def forward(self, preds, targets):
        # 배치 차원을 유지하면서 나머지 차원을 평탄화
        batch_size = preds.size(0) # 현재 배치 크기 저장
        preds = preds.view(batch_size, -1) # [B, H*W] 형태로 변환
        targets = targets.view(batch_size, -1) # [B, H*W] 형태로 변환

        # 배치별 intersection과 union 계산
        intersection = torch.sum(preds * targets, dim=1) # 예측과 타겟의 겹치는 부분
        union = torch.sum(preds, dim=1) + torch.sum(targets, dim=1) # 전체 영역
        
        # 배치별 Dice coefficient 계산: 2|A∩B|/(|A|+|B|)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Dice Loss 계산 (1 - Dice coefficient의 평균)
        return (1. - dice).mean()

class Loss(nn.Module):
    """
    Weighted Focal Loss와 Dice Loss를 결합한 손실 함수

    특징:
    - Focal Loss: 어려운 샘플과 클래스 불균형 처리
    - Dice Loss: 세그멘테이션 품질 향상
    - 두 손실의 가중 합으로 계산
    
    가중치:
    - Dice Loss: 1.0
    - Focal Loss: 100.0 (Focal Loss에 더 큰 가중치 부여)
    
    Attributes:
        w_dice: Dice Loss 가중치
        w_focal: Focal Loss 가중치
    
    반환값:
    - dice_loss: Dice Loss
    - focal_loss: Focal Loss
    """
    def __init__(self):
        super(Loss, self).__init__()
        self.dice_loss = DiceLoss() # Dice Loss 인스턴스 생성
        self.focal_loss = WeightedFocalLoss() # Focal Loss 인스턴스 생성
        self.w_dice = 1.0 # Dice Loss 가중치
        self.w_focal = 100.0 # Focal Loss 가중치 (더 큰 가중치 부여)

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds.squeeze()) # sigmoid 활성화 함수 적용하여 예측값을 0-1 범위로 변환

        # 각각의 손실 계산 및 가중치 적용
        dice_loss = self.dice_loss(preds, targets) * self.w_dice # 가중치가 적용된 Dice Loss
        focal_loss = self.focal_loss(preds, targets) * self.w_focal # 가중치가 적용된 Focal Loss

        return dice_loss, focal_loss

class EarlyStopping:
    """
    Early Stopping 구현

    과적합을 방지하기 위해 validation loss가 연속된 10epoch 동안 개선되지 않으면 학습 조기 종료

    동작 방식:
    1. 매 에폭마다 validation loss 확인
    2. 이전 최적 성능 대비 개선이 없으면 카운터 증가
    3. 특정 에폭(patience) 동안 개선이 없으면 학습 중단
    4. 성능이 개선될 때마다 최적 모델 저장

    Args:
        patience: 성능 개선 없이 허용할 최대 에폭 수
        min_delta: 성능 개선으로 인정할 최소 변화량
        verbose: 상세 로깅 여부
    
    Attributes:
        counter: validation loss이 개선되지 않는 연속 에폭 수 카운터
        best_loss: 현재까지의 최고 loss
        early_stop: 조기 종료 플래그
        val_loss_min: 최소 validation loss 값
    """
    def __init__(self, patience=10, min_delta=0, verbose=False):
        self.patience = patience # 성능 개선 없이 기다릴 에폭 수
        self.min_delta = min_delta # 성능 개선으로 인정할 최소 변화량
        self.verbose = verbose # 상세 로깅 여부
        self.counter = 0 # 성능 개선 없는 에폭 카운터
        self.best_loss = None # 최고 성능 점수
        self.early_stop = False # 조기 종료 플래그
        self.val_loss_min = float('inf')
        self.best_val_loss = float('inf')

    def __call__(self, val_loss, model, path):
        # 첫 에폭인 경우 초기화
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
        
        # 성능 개선이 없는 경우
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1 # 카운터 증가
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # patience 도달 시 조기 종료
            if self.counter >= self.patience:
                self.early_stop = True

        # 성능이 개선된 경우
        else:
            self.best_loss = val_loss # 최고 loss 갱신
            self.save_checkpoint(val_loss, model, path) # 모델 저장
            self.counter = 0 # 카운터 초기화

    def save_checkpoint(self, val_loss, model, path):
        """
        validation loss가 개선된 경우 모델 저장
        
        동작 과정:
        1. 현재 validation loss와 이전 최소 손실 비교
        2. 개선된 경우 모델의 가중치를 파일로 저장
        3. 최소 loss 값 업데이트
        
        Args:
            val_loss: 현재 validation loss 값
            model: 저장할 모델
            path: 모델 저장 경로
        """
        if self.val_loss_min is None or val_loss < self.val_loss_min:
            if self.verbose:
                if self.val_loss_min is None:
                    print(f'Validation Loss: {val_loss:.6f}. Saving initial model...')
                else:
                    print(f'Validation Loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
            torch.save(model.state_dict(), path)
            self.val_loss_min = val_loss
            self.best_val_loss = val_loss

if __name__ == "__main__":
    """
    학습 실행 및 설정
    
    하이퍼파라미터:
    - INITIAL_LEARNING_RATE(초기 학습률): 3e-4 (AdamW 옵티마이저에 적합)
    - BATCH_SIZE: 8 (GPU 메모리를 고려한 설정)
    - EPOCHS: 100 (Early Stopping으로 실제 학습은 더 일찍 종료)
    
    GPU 설정:
    - DataParallel로 병렬 GPU 학습
    
    모델 체크포인트:
    - 손실 기준 / F1 score 기준 별도 저장
    
    wandb 설정:
    - 프로젝트명: "skeleton_detection"
    - 주요 추적 지표: 손실, F1 score
    - 학습률 변화 및 모델 성능 실시간 모니터링
    """
    # 하이퍼파라미터 설정(초기 학습률, 배치사이즈, 에폭 수)
    INITIAL_LEARNING_RATE = 3e-4 # AdamW 옵티마이저의 초기 학습률
    BATCH_SIZE = 8 # 배치 크기 (GPU 메모리 고려)
    EPOCHS = 100 # 총 학습 에폭 수
    
    # 데이터 및 모델 저장 경로 설정
    INTERSECTION_MASK_DIR = "/media/jaemin/새 볼륨/jaemin/final/line_fitting/data/pred_intersection_masks" # "/path/to/intersection_masks"
    SKELETON_MASK_DIR = "/media/jaemin/새 볼륨/jaemin/final/line_fitting/data/line_masks" # "/path/to/line_masks"
    MODEL_SAVE_PATH = "/media/jaemin/새 볼륨/jaemin/final/line_fitting/models" # "/path/to/models"

    # wandb 초기화
    wandb.init(project="skeleton_detection", entity="222449")
    
    # 프로젝트 루트 디렉토리 설정
    project_root = os.path.dirname(os.path.abspath(__file__))

    # wandb 설정
    wandb.config = {
        "initial_learning_rate": INITIAL_LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "model_architecture": "UNet-skeleton",
        "optimizer": "AdamW",
        "lr_scheduler": "ReduceLROnPlateau",
        "loss": "WeightedFocal + Dice"
    }

    # CUDA 사용 가능 여부 확인 및 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # CUDA가 이용 가능한 경우 "cuda"를 사용, 그렇지 않으면 "cpu"를 사용
    print(f"Using device: {device}")
    gpu_ids = [0, 1]  # 사용할 GPU ID 리스트

    # 데이터셋 로드 및 분할
    dataset = IntersectionDataset(intersection_mask_dir=INTERSECTION_MASK_DIR, skeleton_mask_dir=SKELETON_MASK_DIR)

    # 재현성을 위한 시드 값 42로 설정
    torch.manual_seed(42)

    # 데이터셋을 학습/검증/테스트 세트로 분할 (8:1:1 비율)
    total_size = len(dataset) # 학습 데이터셋 비율 설정
    train_size = int(0.8 * total_size) # 학습 데이터셋 크기 계산
    val_size = int(0.1 * total_size) # 검증 데이터셋 크기 계산
    test_size = total_size - train_size - val_size # 테스트 데이터셋 크기 계산

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size]) # 학습, 검증, 테스트 데이터셋 분할

    # 학습 데이터로더 생성
    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True) # 학습 데이터는 에폭마다 섞음
    # 검증 데이터로더 생성
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False) # 검증 데이터는 순서 유지
    # 테스트 데이터로더 생성
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False) # 테스트 데이터는 순서 유지

    # 분할된 데이터셋 크기 출력
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # 모델 생성 및 DataParallel 설정
    model = Skeleton_UNet(in_channels=1, num_classes=1) # 1채널 입력, 1채널 출력
    if torch.cuda.device_count() > 1:
        print(f"Using {len(gpu_ids)} GPUs!")
        model = nn.DataParallel(model, device_ids=gpu_ids) # 다중 GPU 병렬 처리 설정
    model.to(device) # 모델을 지정된 디바이스로 이동

    # 옵티마이저, 스케줄러, 손실함수 설정
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LEARNING_RATE) # AdamW 옵티마이저 - momentum + RMSprop
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5) # 학습률 조정 스케줄러
    criterion = Loss() # Weighted Focal + Dice 손실 함수

    # Early Stopping 설정
    early_stopping = EarlyStopping(patience=10, verbose=True)

    # 모델 저장 디렉토리 생성
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # 최고 성능 추적을 위한 변수 초기화
    best_f1_score = 0.0

    # 학습 루프
    for epoch in range(EPOCHS): # 에폭 반복문
        model.train() # 모델을 학습 모드로 설정
        train_total_loss = 0 # 총 학습 손실
        train_dice_loss = 0 # Dice 손실
        train_focal_loss = 0 # Focal 손실

        # 훈련 단계
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} - Train"): # 학습 데이터 로더에 대해 tqdm 반복문 설정
            intersection_mask = batch['input'].float().to(device) # 용접선과 비전처리 표면의 교집합 마스크 데이터를 가져와서 device로 이동
            skeleton_mask = batch['target'].float().to(device) # 라벨링 마스크 데이터를 가져와서 device로 이동
            target_384 = batch['target_384'].float().to(device) # 1/2로 다운샘플된 라벨링 마스크 데이터를 가져와서 device로 이동
            target_192 = batch['target_192'].float().to(device) # 1/4로 다운샘플된 라벨링 마스크 데이터를 가져와서 device로 이동
            target_96 = batch['target_96'].float().to(device) # 1/8로 다운샘플된 라벨링 마스크 데이터를 가져와서 device로 이동

            # 그래디언트 초기화
            optimizer.zero_grad()

            # 순전파
            outputs = model(intersection_mask)
            output, aux_384, aux_192, aux_96 = outputs # 주 출력과 보조 출력

            # 주 손실과 보조 손실 계산
            dice_loss, focal_loss = criterion(output, skeleton_mask)
            aux_dice_384, aux_focal_384 = criterion(aux_384, target_384)
            aux_dice_192, aux_focal_192 = criterion(aux_192, target_192)
            aux_dice_96, aux_focal_96 = criterion(aux_96, target_96)

            # 전체 손실 계산 (auxiliary 손실 포함) - 여러 scale의 출력에 서로 다른 가중치 적용
            total_dice_loss = 0.5*dice_loss + 0.3*aux_dice_384 + 0.2*aux_dice_192 + 0.1*aux_dice_96
            total_focal_loss = 0.5*focal_loss + 0.3*aux_focal_384 + 0.2*aux_focal_192 + 0.1*aux_focal_96
            loss = total_dice_loss + total_focal_loss
            
            # 역전파
            loss.backward()
            
            # 그래디언트 클리핑 적용 (기울기 폭발 방지)
            clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            # 파라미터 업데이트
            optimizer.step()

            # 손실 누적
            train_total_loss += loss.item()
            train_dice_loss += total_dice_loss.item()
            train_focal_loss += total_focal_loss.item()

        # 평균 훈련 손실 계산
        train_total_loss /= len(train_dataloader)
        train_dice_loss /= len(train_dataloader)
        train_focal_loss /= len(train_dataloader)

        # 검증 단계
        model.eval() # 평가 모드로 설정
        val_total_loss = 0
        val_dice_loss = 0
        val_focal_loss = 0
        val_f1_score = 0.0

        # 각도 오차 추적을 위한 딕셔너리 초기화
        angle_errors = {
            'horizontal': {'total_error': 0, 'count': 0},
            'vertical': {'total_error': 0, 'count': 0}
        }

        with torch.no_grad(): # 그래디언트 계산 비활성화
            for batch_idx, batch in enumerate(tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation")):
                intersection_mask = batch['input'].float().to(device) # 용접선과 비전처리 표면의 교집합 마스크 데이터를 가져와서 device로 이동
                skeleton_mask = batch['target'].float().to(device) # 라벨링 마스크 데이터를 가져와서 device로 이동
                target_384 = batch['target_384'].float().to(device) # 1/2로 다운샘플된 라벨링 마스크 데이터를 가져와서 device로 이동
                target_192 = batch['target_192'].float().to(device) # 1/4로 다운샘플된 라벨링 마스크 데이터를 가져와서 device로 이동
                target_96 = batch['target_96'].float().to(device) # 1/8로 다운샘플된 라벨링 마스크 데이터를 가져와서 device로 이동
        
                # Debugging - 입력 데이터 범위 확인
                print(f"\nBatch {batch_idx}")
                print(f"Input mask range: [{intersection_mask.min():.4f}, {intersection_mask.max():.4f}]")
                print(f"Target mask range: [{skeleton_mask.min():.4f}, {skeleton_mask.max():.4f}]")   

                # 순전파
                outputs = model(intersection_mask)
                output, aux_384, aux_192, aux_96 = outputs

                # Debugging - 모델 출력 범위 확인
                print(f"Raw output range: [{output.min():.4f}, {output.max():.4f}]")

                # 손실 계산
                dice_loss, focal_loss = criterion(output, skeleton_mask)
                aux_dice_384, aux_focal_384 = criterion(aux_384, target_384)
                aux_dice_192, aux_focal_192 = criterion(aux_192, target_192)
                aux_dice_96, aux_focal_96 = criterion(aux_96, target_96)

                # 전체 손실 계산 (스케일별 가중치 적용)
                total_dice_loss = 0.5*dice_loss + 0.3*aux_dice_384 + 0.2*aux_dice_192 + 0.1*aux_dice_96
                total_focal_loss = 0.5*focal_loss + 0.3*aux_focal_384 + 0.2*aux_focal_192 + 0.1*aux_focal_96
                loss = total_dice_loss + total_focal_loss

                # Debugging - Sigmoid 후 값 체크
                pred = torch.sigmoid(output) # sigmoid를 적용하여 예측값을 0-1 사이의 확률값으로 변환
                print(f"After sigmoid range: [{pred.min():.4f}, {pred.max():.4f}]")
                
                # Debugging - Threshold 적용 후 값 체크
                pred_binary = (pred > 0.5).float() # 이진화를 위한 임계값 적용 (0.5 기준)
                print(f"Number of positive predictions: {pred_binary.sum().item()}")
                print(f"Number of positive targets: {skeleton_mask.sum().item()}")

                # Debugging - 계산 과정 체크                
                intersection = (pred_binary * skeleton_mask).sum().item()
                print(f"Intersection: {intersection}")
                total = pred_binary.sum().item() + skeleton_mask.sum().item()
                print(f"Total (pred + target): {total}")

                # Debugging - F1 score(Dice coefficient) 계산
                batch_f1 = calc_dice(pred, skeleton_mask)
                print(f"Batch F1 score: {batch_f1.item():.4f}")

                # 손실과 메트릭 누적
                val_total_loss += loss.item()
                val_dice_loss += total_dice_loss.item()
                val_focal_loss += total_focal_loss.item()
                val_f1_score += batch_f1.item()

                # 각도 오차 계산
                pred = pred.cpu().numpy()
                gt = skeleton_mask.cpu().numpy()

                # 배치 내 각 샘플에 대해 각도 오차 계산
                for i in range(pred.shape[0]):
                    try:
                        errors = calc_angle_errors(pred[i], gt[i])
                    except Exception as e:
                        print(f"Error in sample {i}")
                        print(f"pred shape: {pred[i].shape}, gt shape: {gt[i].shape}")
                        print(f"pred range: [{pred[i].min()}, {pred[i].max()}]")
                        print(f"gt range: [{gt[i].min()}, {gt[i].max()}]")
                        raise e
                    
                    # 수평선이 검출된 경우
                    if errors['horizontal_error'] is not None:
                        angle_errors['horizontal']['total_error'] += errors['horizontal_error'] # 누적 오차 합산
                        angle_errors['horizontal']['count'] += 1 # 검출된 수평선 개수 증가

                    # 수직선이 검출된 경우
                    if errors['vertical_error'] is not None:
                        angle_errors['vertical']['total_error'] += errors['vertical_error'] # 누적 오차 합산
                        angle_errors['vertical']['count'] += 1 # 검출된 수직선 개수 증가

        # 평균 검증 메트릭 계산
        val_total_loss /= len(val_dataloader)
        val_dice_loss /= len(val_dataloader)
        val_focal_loss /= len(val_dataloader)
        val_f1_score /= len(val_dataloader)

        # 평균 각도 오차 계산
        avg_angle_errors = {}
        for direction in ['horizontal', 'vertical']:
            if angle_errors[direction]['count'] > 0:
                # 각 방향의 평균 각도 오차 계산
                avg_angle_errors[direction] = angle_errors[direction]['total_error'] / angle_errors[direction]['count']
            else: # 해당 방향의 직선이 검출되지 않은 경우
                avg_angle_errors[direction] = float('inf')

        # 학습률 스케줄러 업데이트
        scheduler.step(val_total_loss)

        """
        학습 완료 후 저장되는 모델 파일 설명
        
        1. validation loss 기준 모델
        - 파일명: checkpoint.pt
        - 설명: validation loss가 가장 낮았을 때의 모델 가중치
        - 저장 시점: EarlyStopping 클래스에 의해 validation loss 개선 시 자동 저장
        - 과적합 방지에 중점
        
        2. F1 score 기준 모델 (F1 score-based)
        - 파일명: best_f1_checkpoint.pt
        - 설명: validation F1 score가 가장 높았을 때의 모델 가중치
        - 용도: 세그멘테이션 성능이 최고인 시점의 가중치
        - 저장 시점: 각 에폭에서 이전 최고 f1 score 갱신 시 저장
        
        저장 경로: MODEL_SAVE_PATH = "/path/to/models"
        파일 포맷: PyTorch 모델 state_dict (.pt)
        로드 방법: torch.load(path, map_location=device)
        
        사용 시 주의사항:
        - 테스트하려는 모델 구조가 저장 시점과 동일해야 함
        - GPU/CPU 환경이 달라도 호환 가능
        """

        # Early Stopping 체크
        early_stopping(val_total_loss, model.module if isinstance(model, nn.DataParallel) else model, f'{MODEL_SAVE_PATH}/checkpoint.pt')
        
        # Best F1 score 모델 저장
        if val_f1_score > best_f1_score:
            best_f1_score = val_f1_score # 최고 점수 업데이트
            # DataParallel 사용 시 model.module, 아닐 경우 model 그대로 사용
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), f'{MODEL_SAVE_PATH}/best_f1_checkpoint.pt')
            else:
                torch.save(model.state_dict(), f'{MODEL_SAVE_PATH}/best_f1_checkpoint.pt')
            print(f'New best F1 Score: {best_f1_score:.4f}. Saving model...')

        # Early stopping 확인
        if early_stopping.early_stop:
            print("Early stopped")
            break

        # 현재 에폭의 결과 출력
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss - Total: {train_total_loss:.4f}, Dice: {train_dice_loss:.4f}, Focal: {train_focal_loss:.4f}")
        print(f"Val Loss - Total: {val_total_loss:.4f}, Dice: {val_dice_loss:.4f}, Focal: {val_focal_loss:.4f}")
        print(f"Val F1 Score: {val_f1_score:.4f} (Best: {best_f1_score:.4f})")
        
        # 각도 오차 출력
        for direction in ['horizontal', 'vertical']:
            if avg_angle_errors[direction] != float('inf'):
                print(f"{direction.capitalize()} Angle Error: {avg_angle_errors[direction]:.2f}")

        # wandb에 현재 에폭의 메트릭 기록
        log_dict = {
            "epoch": epoch + 1,
            "train/total_loss": train_total_loss,
            "train/dice_loss": train_dice_loss,
            "train/focal_loss": train_focal_loss,
            "val/total_loss": val_total_loss,
            "val/dice_loss": val_dice_loss,
            "val/focal_loss": val_focal_loss,
            "val/f1_score": val_f1_score,
            "learning_rate": optimizer.param_groups[0]['lr']
        }

        # 각도 메트릭 로깅
        for direction in ['horizontal', 'vertical']:
            if avg_angle_errors[direction] != float('inf'):
                log_dict[f"val/{direction}_angle_error"] = avg_angle_errors[direction]

        try:
            wandb.log(log_dict)
        except Exception as e:
            print(f"wandb logging failed: {e}")
            print(f"Attempted to log: {log_dict}")

    # 학습 완료 후 최종 모델 로드 및 결과 출력
    best_model_path = f'{MODEL_SAVE_PATH}/best_f1_checkpoint.pt'
    if os.path.exists(best_model_path): # best_f1_checkpoint.pt 파일이 있는 경우
        checkpoint = torch.load(best_model_path)
        if isinstance(model, nn.DataParallel): # DataParallel 모델인 경우
            model.module.load_state_dict(checkpoint)
        else: # 일반 모델인 경우
            model.load_state_dict(checkpoint)
        print(f"Best model loaded from {best_model_path}")
        print(f"Best F1 Score: {best_f1_score:.4f}")
    else: # best_f1_checkpoint.pt 파일이 없는 경우
        print("No best model found. Using the last model state.")

    print(f"Training completed. Best model saved with F1 Score: {best_f1_score:.4f}")

    # wandb 종료
    wandb.finish()