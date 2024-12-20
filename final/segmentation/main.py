import os  # os 모듈 임포트
import torch  # torch 라이브러리 임포트
import torch.nn as nn
from torch import optim, nn  # torch.optim, torch.nn 모듈 임포트
from torch.utils.data import DataLoader, random_split  # DataLoader, random_split 함수 임포트
from tqdm import tqdm  # tqdm 라이브러리 임포트
import torch.nn.functional as F  # torch.nn.functional 모듈 임포트
import wandb  # wandb 라이브러리 임포트 (Weights & Biases: 실험 추적 도구)
from unet import UNet  # unet 모듈에서 UNet 클래스 임포트
from dataset import WeldlineDataset  # dataset 모듈에서 NewDataset 클래스 임포트
from utils import calc_iou, calc_dice, calc_accuracy, calc_precision_recall # 평가 메트릭 함수들
from torch.nn.utils import clip_grad_norm_ # 그래디언트 클리핑 도구


class DiceBCELoss(nn.Module):
    """
    Dice Loss와 Binary Cross Entropy Loss를 결합한 손실 함수

    특징:
    1. Dice Loss: 클래스 불균형에 강건한 세그멘테이션 손실 함수
       - 전경/배경 픽셀 비율 불균형 문제 해결
       - 작은 객체에 대한 검출 성능 향상
    
    2. BCE Loss: 픽셀 단위 이진 분류에서 표준적으로 사용되는 손실 함수
       - 각 픽셀의 예측 확률에 대한 정확한 학습
       - 경계부분의 세밀한 학습에 효과적

    결합 이점:
    - Dice Loss의 전역적 학습과 BCE의 지역적 학습 효과 동시 획득
    - 다양한 크기의 객체에 대해 균형 잡힌 학습 가능
    
    Args:
        None
        
    Returns:
        torch.Tensor: Dice Loss와 BCE의 합으로 계산된 최종 손실값
    """
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # 입력을 시그모이드 함수로 변환 (0~1 사이 값)
        inputs = torch.sigmoid(inputs)
        
        # input과 target을 1차원으로 평탄화
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 교집합 계산 (예측과 실제가 모두 1인 픽셀 수)
        intersection = (inputs * targets).sum()
        # Dice Loss 계산: 1 - (2|X∩Y| / (|X|+|Y|))                      
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        # Binary Cross Entropy 계산
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        # Dice Loss와 BCE를 결합
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class EarlyStopping:
    """
    클래스별(용접선, 비전처리 표면) Early Stopping 구현
    
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
        task_name: 태스크 구분을 위한 이름 (용접선/비전처리 표면)
    
    Attributes:
        counter: validation loss이 개선되지 않는 연속 에폭 수 카운터
        best_score: 현재까지의 최고 성능 점수
        early_stop: 조기 종료 플래그
        val_loss_min: 최소 validation los s값
    """
    def __init__(self, patience=10, min_delta=0, verbose=False, task_name=""):
        self.patience = patience  # 성능 개선 없이 기다릴 에폭 수
        self.min_delta = min_delta  # 성능 개선으로 인정할 최소 변화량
        self.verbose = verbose  # 상세 로깅 여부
        self.counter = 0  # 성능 개선 없는 에폭 카운터
        self.best_score = None  # 최고 성능 점수
        self.early_stop = False  # 조기 종료 플래그
        self.val_loss_min = float('inf')
        self.best_val_loss = float('inf')
        self.task_name = task_name # 태스크 구분을 위한 이름 (용접선/비전처리 표면)

    def __call__(self, val_loss, model, path):
        score = -val_loss # 검증 손실의 부호를 반전 (점수화)

        # 첫 에폭인 경우 초기화
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)

        # 성능 개선이 없는 경우
        elif score < self.best_score + self.min_delta:
            self.counter += 1 # 카운터 증가
            if self.verbose:
                print(f'{self.task_name} EarlyStopping counter: {self.counter} out of {self.patience}')
            # patience 도달 시 조기 종료
            if self.counter >= self.patience:
                self.early_stop = True

        # 성능이 개선된 경우
        else:
            self.best_score = score # 최고 점수 갱신
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
                    print(f'{self.task_name} Validation Loss: {val_loss:.6f}. Saving initial model...')
                else:
                    print(f'{self.task_name} Validation Loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
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
    - 태스크(용접선/비전처리 표면)별 최적 모델 독립적 저장
    - 손실 기준 / 정확도 기준 별도 저장
    - 최종 앙상블 모델 생성
    
    wandb 설정:
    - 프로젝트명: "weldline"
    - 주요 추적 지표: 손실, IoU, Dice score, 정확도 등
    - 학습률 변화 및 모델 성능 실시간 모니터링
    """
    # 하이퍼파라미터 설정(초기 학습률, 배치사이즈, 에폭 수)
    INITIAL_LEARNING_RATE = 3e-4 # AdamW 옵티마이저의 초기 학습률
    BATCH_SIZE = 8 # 배치 크기 (GPU 메모리 고려)
    EPOCHS = 100 # 총 학습 에폭 수

    # 데이터 및 모델 저장 경로 설정
    IMAGE_DIR = "/media/jaemin/새 볼륨/jaemin/final/segmentation/data/imgs" # "/path/to/images"
    WELD_MASK_DIR = "/media/jaemin/새 볼륨/jaemin/final/segmentation/data/weldline_masks" # "/path/to/weld_masks"
    UNPRETREATED_MASK_DIR = "/media/jaemin/새 볼륨/jaemin/final/segmentation/data/unpretreated_masks" # "/path/to/unpretreated_masks"

    PRETRAINED_MODEL_PATH = "/media/jaemin/새 볼륨/jaemin/final/segmentation/models/best_model_accuracy_averaged.pt" # "/path/to/pretrained_model"
    MODEL_SAVE_PATH = "/media/jaemin/새 볼륨/jaemin/final/segmentation/models" # "/path/to/models"

    # wandb 초기화
    wandb.init(project="weldline", entity="222449")  

    # 프로젝트 루트 디렉토리 설정
    project_root = os.path.dirname(os.path.abspath(__file__))

    # wandb 설정
    wandb.config = {  
    "initial_learning_rate": INITIAL_LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "model_architecture": "UNet",
    "optimizer": "AdamW",
    "lr_scheduler": "ReduceLROnPlateau"
    }

    # CUDA 사용 가능 여부 확인 및 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CUDA가 이용 가능한 경우 "cuda"를 사용, 그렇지 않으면 "cpu"를 사용
    print(f"Using device: {device}")
    gpu_ids = [0, 1]  # 사용할 GPU ID 리스트

    # 데이터셋 로드 및 분할
    dataset = WeldlineDataset(
        image_dirs=IMAGE_DIR,
        weld_mask_dirs=WELD_MASK_DIR,
        unpretreated_mask_dirs=UNPRETREATED_MASK_DIR
    )

    # 재현성을 위한 시드 값 42로 설정
    torch.manual_seed(42)

    # 데이터셋을 학습/검증/테스트 세트로 분할 (8:1:1 비율)
    total_size = len(dataset) # 학습 데이터셋 비율 설정
    train_size = int(0.8 * total_size) # 학습 데이터셋 크기 계산
    val_size = int(0.1 * total_size) # 검증 데이터셋 크기 계산
    test_size = total_size - train_size - val_size # 테스트 데이터셋 크기 계산
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size]) # 학습, 검증, 테스트 데이터셋 분리

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
    model = UNet(in_channels=3, num_classes=2) # 3채널 입력, 2채널 출력
    if torch.cuda.device_count() > 1:
        print(f"Using {len(gpu_ids)} GPUs!")
        model = nn.DataParallel(model, device_ids=gpu_ids) # 다중 GPU 병렬 처리 설정
    model.to(device) # 모델을 지정된 디바이스로 이동

    # 사전 훈련된 가중치 로드
    if os.path.exists(PRETRAINED_MODEL_PATH): # 사전 훈련된 가중치가 있을 시
        print(f"Loading pretrained weights from {PRETRAINED_MODEL_PATH}")
        state_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
    else: # 사전 훈련된 가중치가 없을 시
        print(f"Pretrained weights not found at {PRETRAINED_MODEL_PATH}. Starting with random initialization.")

    # 옵티마이저, 스케줄러, 손실함수 설정
    optimizer_weld = optim.AdamW(model.parameters(), lr=INITIAL_LEARNING_RATE) # AdamW 옵티마이저 - momentum + RMSprop
    optimizer_unpretreated = optim.AdamW(model.parameters(), lr=INITIAL_LEARNING_RATE) # AdamW 옵티마이저 - momentum + RMSprop
    scheduler_weld = optim.lr_scheduler.ReduceLROnPlateau(optimizer_weld, mode='min', factor=0.5, patience=5) # 학습률 조정 스케줄러
    scheduler_unpretreated = optim.lr_scheduler.ReduceLROnPlateau(optimizer_unpretreated, mode='min', factor=0.5, patience=5) # 학습률 조정 스케줄러
    criterion = DiceBCELoss() # Dice + Binary Cross Entropy 손실 함수

    # 클래스별 Early Stopping 설정
    early_stopping_weld = EarlyStopping(patience=10, verbose=True, task_name="Weldline")
    early_stopping_unpretreated = EarlyStopping(patience=10, verbose=True, task_name="Unpretreated")

    # 모델 저장 디렉토리 생성
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)  

    # 학습 메트릭 초기화
    best_val_accuracy_weld = 0
    best_val_accuracy_unpretreated = 0
    best_model_state_weld = None
    best_model_state_unpretreated = None
    weld_stopped = False
    unpretreated_stopped = False

    # 학습 루프
    for epoch in range(EPOCHS):  # 에폭 반복문
        model.train()  # 모델을 학습 모드로 설정
        train_loss_weld = 0
        train_loss_unpretreated = 0
        train_loss_average = 0

        # 훈련 단계
        for img, weld_mask, unpretreated_mask, _ in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} - Train"):  # 학습 데이터 로더에 대해 tqdm 반복문 설정
            img = img.float().to(device)  # 이미지 데이터를 가져와서 device로 이동
            weld_mask = weld_mask.float().to(device)  # 용접선 마스크 데이터를 가져와서 device로 이동
            unpretreated_mask = unpretreated_mask.float().to(device) # 비전처리 표면 마스크 데이터를 가져와서 device로 이동

            # 그래디언트 초기화
            optimizer_weld.zero_grad() 
            optimizer_unpretreated.zero_grad()

            # 순전파
            weld_out, unpretreated_out = model(img)

            # 손실 계산
            loss_weld = criterion(weld_out, weld_mask)
            loss_unpretreated = criterion(unpretreated_out, unpretreated_mask)
            
            # 역전파
            loss_weld.backward(retain_graph=True)
            loss_unpretreated.backward()

            # 그래디언트 클리핑 적용 (기울기 폭발 방지)
            clip_grad_norm_(model.parameters(), max_norm=10.0)

            # 파라미터 업데이트
            optimizer_weld.step()
            optimizer_unpretreated.step()

            # 손실 누적
            train_loss_weld += loss_weld.item()
            train_loss_unpretreated += loss_unpretreated.item()

        # 평균 훈련 손실 계산
        train_loss_weld /= len(train_dataloader)
        train_loss_unpretreated /= len(train_dataloader)
        train_loss_average = (train_loss_weld + train_loss_unpretreated) / 2

        # 검증 단계
        model.eval() # 평가 모드로 설정
        val_loss_average = 0
        val_loss_weld = 0
        val_loss_unpretreated = 0
        val_iou_weld = 0
        val_iou_unpretreated = 0
        val_dice_weld = 0
        val_dice_unpretreated = 0
        val_accuracy_weld = 0
        val_accuracy_unpretreated = 0
        val_precision_weld = 0
        val_precision_unpretreated = 0
        val_recall_weld = 0
        val_recall_unpretreated = 0

        with torch.no_grad(): # 그래디언트 계산 비활성화
            for img, weld_mask, unpretreated_mask, _ in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation"):
                img = img.float().to(device) # 이미지 데이터를 가져와서 device로 이동
                weld_mask = weld_mask.float().to(device)  # 용접선 마스크 데이터를 가져와서 device로 이동
                unpretreated_mask = unpretreated_mask.float().to(device) # 비전처리 표면 마스크 데이터를 가져와서 device로 이동

                # 순전파
                weld_out, unpretreated_out = model(img)

                # sigmoid를 적용하여 예측값을 0-1 사이의 확률값으로 변환
                pred_weld = torch.sigmoid(weld_out)
                pred_unpretreated = torch.sigmoid(unpretreated_out)

                # 손실 계산
                loss_weld = criterion(weld_out, weld_mask)
                loss_unpretreated = criterion(unpretreated_out, unpretreated_mask)

                val_loss_weld += loss_weld.item()
                val_loss_unpretreated += loss_unpretreated.item()

                # 메트릭 계산 및 누적 (용접선)
                val_iou_weld += calc_iou(pred_weld, weld_mask).item()
                val_dice_weld += calc_dice(pred_weld, weld_mask).item()
                val_accuracy_weld += calc_accuracy(pred_weld, weld_mask).item()
                precision_weld, recall_weld = calc_precision_recall(pred_weld, weld_mask)
                val_precision_weld += precision_weld.item()
                val_recall_weld += recall_weld.item()

                # 메트릭 계산 및 누적 (비전처리 표면)
                val_iou_unpretreated += calc_iou(pred_unpretreated, unpretreated_mask).item()
                val_dice_unpretreated += calc_dice(pred_unpretreated, unpretreated_mask).item()
                val_accuracy_unpretreated += calc_accuracy(pred_unpretreated, unpretreated_mask).item()
                precision_unpretreated, recall_unpretreated = calc_precision_recall(pred_unpretreated, unpretreated_mask)
                val_precision_unpretreated += precision_unpretreated.item()
                val_recall_unpretreated += recall_unpretreated.item()

        # 평균 검증 메트릭 계산
        val_loss_weld /= len(val_dataloader)
        val_loss_unpretreated /= len(val_dataloader)
        val_loss_average = (val_loss_weld + val_loss_unpretreated) / 2
        
        val_iou_weld /= len(val_dataloader)
        val_iou_unpretreated /= len(val_dataloader)
        val_iou_average = (val_iou_weld + val_iou_unpretreated) / 2
        
        val_dice_weld /= len(val_dataloader)
        val_dice_unpretreated /= len(val_dataloader)
        val_dice_average = (val_dice_weld + val_dice_unpretreated) / 2

        val_accuracy_weld /= len(val_dataloader)
        val_accuracy_unpretreated /= len(val_dataloader)
        val_accuracy_average = (val_accuracy_weld + val_accuracy_unpretreated) / 2

        val_precision_weld /= len(val_dataloader)
        val_precision_unpretreated /= len(val_dataloader)
        val_precision_average = (val_precision_weld + val_precision_unpretreated) / 2

        val_recall_weld /= len(val_dataloader)
        val_recall_unpretreated /= len(val_dataloader)
        val_recall_average = (val_recall_weld + val_recall_unpretreated) / 2

        # 학습률 스케줄러 업데이트
        scheduler_weld.step(val_loss_weld)
        scheduler_unpretreated.step(val_loss_unpretreated)

        """
        학습 완료 후 저장되는 모델 파일 설명

        1. validation loss 기준 모델
        - 파일명: best_model_loss_weld.pt, best_model_loss_unpretreated.pt
        - 설명: validation loss가 가장 낮았을 때의 모델 가중치
        - 용도: 일반적으로 가장 안정적인 성능을 보이는 모델
        - 저장 시점: EarlyStopping 클래스에 의해 validation loss 개선시 자동 저장

        2. 정확도 기준 모델 (Accurcay-based)
        - 파일명: best_model_accuracy_weld.pt, best_model_accuracy_unpretreated.pt
        - 설명: validation 정확도가 가장 높았을 때의 모델 가중치
        - 용도: 특정 태스크에 대해 최고의 성능을 보이는 모델
        - 저장 시점: 각 에폭에서 이전 최고 정확도 갱신 시 저장

        3. 앙상블 모델 (Averaged Ensemble)
        - 파일명: best_model_accuracy_averaged.pt
        - 설명: 용접선과 비전처리 표면 각각의 최고 정확도 모델의 가중치 평균
        - 용도: 두 태스크 간의 균형잡힌 성능을 보이는 범용 모델
        - 특징: 개별 태스크의 과적합을 완화하고 일반화 성능 향상
        - 저장 시점: 학습 완료 후 최종 단계에서 한 번 저장

        저장 경로: MODEL_SAVE_PATH = "/path/to/models"
        파일 포맷: PyTorch 모델 state_dict (.pt)
        로드 방법: torch.load(path, map_location=device)

        참고:
        - 실제 사용 시에는 검증 세트에서의 성능을 비교하여 가장 적합한 모델 선택
        - 실제 최고 성능을 보인 모델 파일: 앙상블 모델(best_model_accuracy_averaged.pt)
        - 테스트하려는 모델 구조가 저장 시점과 동일해야 함
        """

        # Early Stopping 체크 및 Best Accuracy 모델 저장
        # 용접선 태스크 처리
        if not weld_stopped: # 용접선 태스크가 아직 조기 종료되지 않은 경우
            # 검증 손실 기반 Early Stopping 체크 및 최적 모델 저장
            # DataParallel 사용 시 model.module, 아닐 경우 model 그대로 사용
            early_stopping_weld(val_loss_weld, model.module if isinstance(model, nn.DataParallel) else model, f'{MODEL_SAVE_PATH}/best_model_loss_weld.pt')
            if val_accuracy_weld > best_val_accuracy_weld: # 현재 검증 정확도가 이전 최고 정확도보다 높은 경우
                best_val_accuracy_weld = val_accuracy_weld # 최고 정확도 갱신
                best_model_state_weld = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict() # 현재 모델의 가중치를 저장 (DataParallel 고려)
            if early_stopping_weld.early_stop: # Early Stopping 조건 충족 시
                print("Early stopping for weldline task.")
                weld_stopped = True # 용접선 태스크 종료 플래그 설정
                torch.save(best_model_state_weld, f'{MODEL_SAVE_PATH}/best_model_accuracy_weld.pt') # 용접선 최고 정확도 모델 저장

        # 비전처리 표면 태스크 처리 (용접선과 동일한 로직)
        if not unpretreated_stopped: # 비전처리 표면 태스크가 아직 조기 종료되지 않은 경우
            # 검증 손실 기반 Early Stopping 체크 및 최적 모델 저장
            # DataParallel 사용 시 model.module, 아닐 경우 model 그대로 사용
            early_stopping_unpretreated(val_loss_unpretreated, model.module if isinstance(model, nn.DataParallel) else model, f'{MODEL_SAVE_PATH}/best_model_loss_unpretreated.pt')
            if val_accuracy_unpretreated > best_val_accuracy_unpretreated: # 현재 검증 정확도가 이전 최고 정확도보다 높은 경우
                best_val_accuracy_unpretreated = val_accuracy_unpretreated # 최고 정확도 갱신
                best_model_state_unpretreated = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict() # 현재 모델의 가중치를 저장 (DataParallel 고려)
            if early_stopping_unpretreated.early_stop: # Early Stopping 조건 충족 시
                print("Early stopping for unpretreated task.")
                unpretreated_stopped = True # 비전처리 표면 태스크 종료 플래그 설정
                torch.save(best_model_state_unpretreated, f'{MODEL_SAVE_PATH}/best_model_accuracy_unpretreated.pt') # 비전처리 표면 최고 정확도 모델 저장

        # 두 태스크 모두 Early stopping 되면 학습 종료
        if weld_stopped and unpretreated_stopped:
            print("Both tasks have stopped. Ending training.")
            break

        # 현재 에폭의 학습 결과 출력
        print(f"Epoch {epoch+1}/{EPOCHS}")

        # 용접선 태스크가 진행 중인 경우 해당 메트릭 출력
        if not weld_stopped:
            print(f"Weldline - Train Loss: {train_loss_weld:.4f}, Val Loss: {val_loss_weld:.4f}")
            print(f"Weldline - Val IoU: {val_iou_weld:.4f}, Dice(F1): {val_dice_weld:.4f}, Accuracy: {val_accuracy_weld:.4f}")
            print(f"Weldline - Val Precision: {val_precision_weld:.4f}, Recall: {val_recall_weld:.4f}")
        # 비전처리 표면 태스크가 진행 중인 경우 해당 메트릭 출력
        if not unpretreated_stopped:
            print(f"Unpretreated - Train Loss: {train_loss_unpretreated:.4f}, Val Loss: {val_loss_unpretreated:.4f}")
            print(f"Unpretreated - Val IoU: {val_iou_unpretreated:.4f}, Dice(F1): {val_dice_unpretreated:.4f}, Accuracy: {val_accuracy_unpretreated:.4f}")
            print(f"Unpretreated - Val Precision: {val_precision_unpretreated:.4f}, Recall: {val_recall_unpretreated:.4f}")
        # 두 태스크 모두 진행 중인 경우 평균 메트릭 출력
        if not weld_stopped and not unpretreated_stopped:
            print(f"Average - Train Loss: {train_loss_average:.4f}, Val Loss: {val_loss_average:.4f}")
            print(f"Average - Val IoU: {val_iou_average:.4f}, Dice(F1): {val_dice_average:.4f}, Accuracy: {val_accuracy_average:.4f}")
            print(f"Average - Val Precision: {val_precision_average:.4f}, Recall: {val_recall_average:.4f}")

        # Wandb(Weights & Biases) 로깅을 위한 딕셔너리 초기화
        log_dict = {
            "epoch": epoch + 1, # 현재 에폭 번호
        }

        # 용접선 태스크 진행 중인 경우 해당 메트릭 로깅
        if not weld_stopped: 
            log_dict.update({
                "Train Loss(Weld)": train_loss_weld,
                "Val Loss(Weld)": val_loss_weld,
                "Val IoU(Weld)": val_iou_weld,
                "Val Dice(F1)(Weld)": val_dice_weld,
                "Val Accuracy(Weld)": val_accuracy_weld,
                "Val Precision(Weld)": val_precision_weld,
                "Val Recall(Weld)": val_recall_weld,
                "Learning Rate(Weld)": optimizer_weld.param_groups[0]['lr'] # 현재 학습률
            })

        # 비전처리 태스크 진행 중인 경우 해당 메트릭 로깅
        if not unpretreated_stopped:
            log_dict.update({
                "Train Loss(Unpretreated)": train_loss_unpretreated,
                "Val Loss(Unpretreated)": val_loss_unpretreated,
                "Val IoU(Unpretreated)": val_iou_unpretreated,
                "Val Dice(F1)(Unpretreated)": val_dice_unpretreated,
                "Val Accuracy(Unpretreated)": val_accuracy_unpretreated,
                "Val Precision(Unpretreated)": val_precision_unpretreated,
                "Val Recall(Unpretreated)": val_recall_unpretreated,
                "Learning Rate(Unpretreated)": optimizer_unpretreated.param_groups[0]['lr'] # 현재 학습률
            })

        # 두 태스크 모두 진행 중인 경우 평균 메트릭 로깅
        if not weld_stopped and not unpretreated_stopped:
            log_dict.update({
                "Train Loss(Average)": train_loss_average,
                "Val Loss(Average)": val_loss_average,
                "Val Iou(Average)": val_iou_average,
                "Val Dice(F1)(Average)": val_dice_average,
                "Val Accuracy(Average)": val_accuracy_average,
                "Val Precision(Average)": val_precision_average,
                "Val Recall(Average)": val_recall_average,
            })

        try:
            wandb.log(log_dict)
        except Exception as e:
            print(f"wandb logging failed: {e}") # 로깅 실패 시 에러 출력
            print(f"Attempted to log: {log_dict}") # 로깅 시도한 데이터 출력

    # 학습 완료 후 최종 모델 저장 - 용접선과 비전처리 각각의 최고 정확도 모델 저장
    torch.save(best_model_state_weld, f'{MODEL_SAVE_PATH}/best_model_accuracy_weld.pt')
    torch.save(best_model_state_unpretreated, f'{MODEL_SAVE_PATH}/best_model_accuracy_unpretreated.pt')

    # 두 Best Accuracy 모델의 가중치 평균 계산 및 모델 저장 (앙상블 모델)
    averaged_state_dict = {}
    for key in best_model_state_weld.keys():
        averaged_state_dict[key] = (best_model_state_weld[key] + best_model_state_unpretreated[key]) / 2 # 각 레이어의 가중치를 평균
    
    torch.save(averaged_state_dict, f'{MODEL_SAVE_PATH}/best_model_accuracy_averaged.pt') # 앙상블 모델 저장

    # 최종 결과 및 저장된 모델 정보 출력
    print(f"Training completed.")
    print(f"Best loss model (weld) saved to {MODEL_SAVE_PATH}/best_model_loss_weld.pt with loss: {early_stopping_weld.best_val_loss:.6f}")
    print(f"Best loss model (unpretreated) saved to {MODEL_SAVE_PATH}/best_model_loss_unpretreated.pt with loss: {early_stopping_unpretreated.best_val_loss:.6f}")
    print(f"Best accuracy model (weld) saved to {MODEL_SAVE_PATH}/best_model_accuracy_weld.pt with accuracy: {best_val_accuracy_weld:.4f}")
    print(f"Best accuracy model (unpretreated) saved to {MODEL_SAVE_PATH}/best_model_accuracy_unpretreated.pt with accuracy: {best_val_accuracy_unpretreated:.4f}")
    print(f"Averaged best accuracy model saved to {MODEL_SAVE_PATH}/best_model_accuracy_averaged.pt")

    # DataParallel 사용한 경우 원래 모델로 변환
    if isinstance(model, nn.DataParallel):
        model = model.module

    print(f"Training completed.") # 학습 완료 시 문구 표시