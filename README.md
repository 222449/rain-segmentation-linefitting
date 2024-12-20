# Segmentation & LineFitting Project (전처리 로봇 연구 - 용접선 & 비전처리 표면 검출 및 직선 피팅)

## 📌 프로젝트 개요

이 프로젝트는 1-2 branches(encoder 4번째 레이어 분기) U-Net을 통해 이미지에서 용접선과 비전처리 표면을 검출하고, 검출된 둘의 교집합 영역에서 U-Net 기반의 skeletonization을 통해 직선을 피팅하는 시스템을 구현합니다.

### 주요 기능

1. 세그멘테이션 모델
   - 용접선 검출
   - 비전처리 표면 검출
   - 멀티 태스크 학습

2. 직선 피팅 모델
   - 교집합 영역 스켈레톤화
   - RANSAC 기반 직선 검출
   - 수직/수평 직선 분류

## 🛠 시스템 요구사항

### Python 환경
- Python 3.10
- CUDA 지원

### 주요 패키지
```
torch==1.12.0
torchaudio==0.12.0
torchvision==0.13.0
numpy==1.26.4
opencv-python==4.9.0.80
Pillow==10.3.0
wandb==0.17.0
tqdm==4.66.4
```

## 📂 프로젝트 구조

```
final/
│
├── in_video/                                  # 테스트 input 영상
│   ├── 10_shape.mp4                              # 십자형
│   ├── T_shpae.mp4                               # T자형
│   └── with_laser_straight.mp4                   # 일자형
├── Only_Segmentation_video                    # 영상에서의 segmentation_inference.py 실행 결과
├── Segmentation_RANSAC_Kalman_video           # 영상에서의 RANSAC_inference.py 실행 결과
├── Segmentation_Skeleton_RANSAC_Kalman_video  # 영상에서의 skeleton_RANSAC_inference.py 실행 결과
├── segmentation_inference_results             # segmentation 결과 성능과 샘플 시각화 (segmentation_inference.py 실행 결과)
├── skeleton_inference_results                 # skeleton 결과 성능과 샘플 시각화 (skeleton_RANSAC_inference.py 실행 결과)
├── segmentation_inference.py                  # segmentation(1-2 branches U-Net) 모델 추론 파일
├── RANSAC_inference.py                        # segmentation된 영역에서 RANSAC+Kalman filter 직선 피팅 추론 파일
├── skeleton_RANSAC_inference.py               # segmentation과 U-Net 기반 skeletonization 후 RANSAC+Kalman filter 직선 피팅 추론 파일
├── __init__.py                                # 디렉토리 패키지 인식 파일
├── requirements.txt                           # 가상환경(conda)에 설치된 오픈 소스 라이브러리 버전 정보
│
├── segmentation/
│   ├── data               # 세그멘테이션용 데이터셋(RGB 이미지, 용접선 이진 마스크, 비전처리 표면 이진 마스크) 디렉토리
│   ├── models             # 저장된 모델 파일 디렉토리
│   ├── __init__.py        # 디렉토리 패키지 인식 파일
│   ├── dataset.py         # 세그멘테이션용 데이터셋 전처리 파일
│   ├── main.py            # 세그멘테이션 학습 실행 파일
│   ├── unet.py            # 1-2 branches(encoder 4번째 레이어 분기) U-Net 모델 구조 파일
│   ├── unet_parts.py      # U-Net 구성요소 파일
│   └── utils.py           # 평가 메트릭 파일
│
└── line_fitting/
    ├── data                    # 라인 피팅 데이터셋(segmentation 결과 예측된 교집합 이진 마스크, json 라벨링 파일 기반 직선 마스크, 라벨링 결과 json 파일) 디렉토리
    ├── models                  # 저장된 모델 파일 디렉토리
    ├── __init__.py             # 디렉토리 패키지 인식 파일
    ├── skeleton_dataset.py     # 라인 피팅 데이터셋 전처리 파일
    ├── skeleton_main.py        # 직선 피팅 학습 실행 파일
    ├── skeleton_unet.py        # skeleton 기반 U-Net 모델 구조 파일
    ├── skeleton_unet_parts.py  # 모델 구성요소 파일
    └── skeleton_utils.py       # RANSAC(직선 피팅) 함수 및 평가 메트릭 파일
```

## 💡 주요 특징

### 1. 세그멘테이션 모델

- **이중 분기(1-2 branches) U-Net 구조**
  - 공유 인코더로 특징 추출 효율성 향상
  - 각 태스크별 독립적 디코더를 통한 개별 특징 학습
  - Skip connection으로 edge 등과 같은 세부 정보 보존

- **손실 함수**
  - Dice Loss + Binary Cross Entropy
  - 클래스 불균형 문제 해결
  - 평균 손실로 가중치 업데이트

- **학습 최적화**
  - Early Stopping
  - Learning Rate Scheduling
  - 그래디언트 클리핑 (그래디언트 폭발 방지)

### 2. 직선 피팅 모델

- **Skeletonization U-Net**
  - Attention 메커니즘 적용 (채널 Attention/공간 Attention)
  - 다중 스케일 예측 (Auxiliary learning task)
  - 채널/공간 Attention 결합

- **RANSAC 직선 검출**
  - 반복적 모델 피팅
  - 이상치(outlier) 제거
  - 각도 기반 방향별 직선 분류 (horizontal/vertical)

- **Kalman Filter 기반 직선 안정화**
  - 직선의 각도와 각속도를 상태 벡터로 추적
  - 방향별(수평/수직) 독립적인 필터 적용
  - 급격한 변화 감지 및 완화를 통한 직선 떨림 현상 개선
  - 위치와 각도 기반의 이중 안정화 메커니즘

- **손실 함수**
  - Weighted Focal Loss + Dice Loss
  - 다중 스케일 auxiliary loss 가중합
  - 클래스 불균형 해결

## 🚀 사용 방법

### 1. 환경 설정

#### 1.1 zip 파일 다운로드

#### 1.2 Conda 가상환경 설정
```bash
# Conda 가상환경 생성
conda create -n [생성할 가상환경 이름] python=3.10
conda activate [생성할 가상환경 이름]

# PyTorch 설치 (CUDA 11.3 버전 기준)
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

# 필수 패키지 설치
pip install -r requirements.txt
```

### 2. 모델 학습


#### 2.1 세그멘테이션 모델 학습
```bash
# 기본 설정으로 학습
final 디렉토리 열고 python segmentation/main.py 명령어 실행

    --image_dir /path/to/images \
    --weld_mask_dir /path/to/weld_masks \
    --unpretreated_mask_dir /path/to/unpretreated_masks \
    [--batch_size 8] \
    [--initial_learning_rate 3e-4] \
    [--epochs 100]


# 하이퍼파라미터 조정 및 데이터와 모델 저장 경로 설정 예시
final 디렉토리 열고 python segmentation/main.py 명령어 실행

    --INITIAL_LEARNING_RATE = 1e-4 \
    --BATCH_SIZE = 4 \
    --EPOCHS = 150 \

    --IMAGE_DIR = segmentation/data/imgs \
    --WELD_MASK_DIR = segmentation/data/weldline_masks \
    --UNPRETREATED_MASK_DIR = segmentation/data/unpretreated_masks \

    --PRETRAINED_MODEL_PATH = segmentation/models/best_model_accuracy_averaged.pt \
    --MODEL_SAVE_PATH = segmentation/models
```

#### 2.2 직선 피팅 모델 학습
```bash
# 기본 설정으로 학습
final 디렉토리 열고 python line_fitting/skeleton_main.py 명령어 실행

    --intersection_mask_dir /path/to/intersection_masks \
    --skeleton_mask_dir /path/to/line_masks \
    [--batch_size 8] \
    [--initial_learning_rate 3e-4] \
    [--epochs 100]


# 하이퍼파라미터 조정 및 데이터와 모델 저장 경로 설정 예시
final 디렉토리 열고 python line_fitting/skeleton_main.py 명령어 실행

    --INITIAL_LEARNING_RATE = 1e-4 \
    --BATCH_SIZE = 4 \
    --EPOCHS = 150 \

    --INTERSECTION_MASK_DIR = line_fitting/data/pred_intersection_masks \
    --SKELETON_MASK_DIR = line_fitting/data/line_masks \

    --MODEL_SAVE_PATH = line_fitting/models
```

### 3. 모델 추론

#### 3.1 세그멘테이션 추론 (segmentation_inference.py)
```bash
# 기본 설정으로 추론
final 디렉토리 열고 python segmentation_inference.py 명령어 실행

    --image_dir segmentation/data/imgs \
    --weld_mask_dir segmentation/data/weldline_masks \
    --unpretreated_mask_dir segmentation/data/unpretreated_masks \

    --save_dir segmentation_inference_results
    --input_video in_video/10_shape.mp4 \
    --model_path segmentation/models/best_model_accuracy_averaged.pt \


# test_model과 visualize_predictions 함수 사용하여 추론 (데이터 경로와 모델 저장 경로 설정에 따른 예시)
final 디렉토리부터 상대경로 설정한 후, python segmentation_inference.py 명령어 실행

    --image_dir segmentation/data/test_imgs \
    --weld_mask_dir segmentation/data/test_weldline_masks \
    --unpretreated_mask_dir segmentation/data/test_unpretreated_masks \

    --batch_size 16 \

    --save_dir inference_results \
    --model_path segmentation/models/best_model_accuracy_averaged.pt


# 여러 비디오 용접선 형태별 추론 - final 디렉토리 열고 아래 명령어 실행
python segmentation_inference.py 명령어 실행
    --input_video in_video/10_shape.mp4 \     # 십자형
    --model_path segmentation/models/best_model_accuracy_averaged.pt

python segmentation_inference.py 명령어 실행
    --input_video in_video/T_shape.mp4 \      # T자형
    --model_path segmentation/models/best_model_accuracy_averaged.pt

python segmentation_inference.py 명령어 실행
    --input_video in_video/with_laser_straight.mp4 \  # 일자형
    --model_path segmentation/models/best_model_accuracy_averaged.pt
```

**주요 파라미터 설명**:
- `--image_dir`: 테스트 이미지 디렉토리 경로
- `--weld_mask_dir`: 용접선 마스크 디렉토리 경로
- `--unpretreated_mask_dir`: 비전처리 표면 마스크 디렉토리 경로
- `--model_path`: 학습된 모델 가중치 파일 경로
- `--save_dir`: 결과 저장 디렉토리 (기본값: 'segmentation_inference_results')
- `--batch_size`: 배치 크기 (기본값: 8)
- `--input_video`: 입력 비디오 파일 경로
- `--use_wandb`: Weights & Biases 로깅 사용 여부 (플래그)


#### 3.2 RANSAC + Kalman filter 직선 피팅 추론 (RANSAC_inference.py)
```bash
# 기본 설정으로 추론
final 디렉토리 열고 python RANSAC_inference.py 명령어 실행

    --input_video in_video/10_shape.mp4 \
    --model_path segmentation/models/best_model_accuracy_averaged.pt


# 여러 비디오 용접선 형태별 추론 - final 디렉토리 열고 아래 명령어 실행
python RANSAC_inference.py 명령어 실행
    --input_video in_video/10_shape.mp4 \      # 십자형
    --model_path segmentation/models/best_model_accuracy_averaged.pt \

python RANSAC_inference.py 명령어 실행
    --input_video in_video/T_shape.mp4 \      # T자형
    --model_path segmentation/models/best_model_accuracy_averaged.pt \

python RANSAC_inference.py 명령어 실행
    --input_video in_video/with_laser_straight.mp4 \      # 일자형
    --model_path segmentation/models/best_model_accuracy_averaged.pt
```

**주요 파라미터 설명**:
- `--input_video`: 입력 비디오 파일 경로
- `--model_path`: 세그멘테이션 모델 가중치 파일 경로

#### 3.3 Skeleton과 RANSAC + Kalman filter 결합 직선 피팅 추론 (skeleton_RANSAC_inference.py)
```bash
# 기본 설정으로 추론
final 디렉토리 열고 python skeleton_RANSAC_inference.py 명령어 실행

    --intersection_mask_dir line_fitting/data/pred_intersection_masks \
    --skeleton_mask_dir line_fitting/data/line_masks \

    --save_dir skeleton_inference_results \
    --input_video in_video/10_shape.mp4 \
    --seg_model_path segmentation/models/best_model_accuracy_averaged.pt \
    --skeleton_model_path line_fitting/models/best_f1_checkpoint.pt


# test_model과 visualize_predictions 함수 사용하여 추론 (데이터 경로와 모델 저장 경로 설정에 따른 예시)
final 디렉토리부터 상대경로 설정한 후, python skeleton_RANSAC_inference.py 명령어 실행

    --intersection_mask_dir line_fitting/data/test_intersection_masks \
    --skeleton_mask_dir line_fitting/data/test_line_masks \

    --batch_size 16 \
    --save_dir detailed_results \
    --seg_model_path segmentation/models/best_model_accuracy_averaged.pt \
    --skeleton_model_path line_fitting/models/best_f1_checkpoint.pt


# 여러 비디오 용접선 형태별 추론 - final 디렉토리 열고 아래 명령어 실행
python skeleton_RANSAC_inference.py 명령어 실행
    --input_video in_video/10_shape.mp4 \      # 십자형
    --seg_model_path segmentation/models/best_model_accuracy_averaged.pt \
    --skeleton_model_path line_fitting/models/best_f1_checkpoint.pt


python skeleton_RANSAC_inference.py 명령어 실행
    --input_video in_video/T_shape.mp4 \      # T자형
    --seg_model_path segmentation/models/best_model_accuracy_averaged.pt \
    --skeleton_model_path line_fitting/models/best_f1_checkpoint.pt


python skeleton_RANSAC_inference.py  명령어 실행
    --input_video in_video/with_laser_straight.mp4 \      # 일자형
    --seg_model_path segmentation/models/best_model_accuracy_averaged.pt \
    --skeleton_model_path line_fitting/models/best_f1_checkpoint.pt
```

**주요 파라미터 설명**:
- `--input_video`: 입력 비디오 파일 경로
- `--intersection_mask_dir`: 용접선과 비전처리 표면의 교집합 마스크 디렉토리 경로
- `--skeleton_mask_dir`: 라벨링 라인 마스크 디렉토리 경로
- `--seg_model_path`: 세그멘테이션 모델 가중치 파일 경로
- `--skeleton_model_path`: 직선 피팅 모델 가중치 파일 경로
- `--save_dir`: 결과 저장 디렉토리 (기본값: 'skeleton_inference_results')
- `--batch_size`: 배치 크기 (기본값: 8)
- `--use_wandb`: Weights & Biases 로깅 사용 여부 (플래그)

## 📊 성능 평가

### 세그멘테이션 성능 지표
- IoU (Intersection over Union)
- Dice coefficient (F1 score)
- Precision & Recall
- 픽셀 단위 Accuracy

### 직선 피팅 성능 지표
- 방향별 각도 오차 (MAE)
- Dice coefficient (F1 score)

## 📝 학습 결과 저장 모델 파일

### 세그멘테이션 모델
- `best_model_loss_weld.pt`: 용접선 최소 손실 모델 파일
- `best_model_accuracy_weld.pt`: 용접선 최고 정확도 모델 파일
- `best_model_loss_unpretreated.pt`: 비전처리 표면 최소 손실 모델 파일 
- `best_model_accuracy_unpretreated.pt`: 비전처리 표면 최고 정확도 모델 파일
- `best_model_accuracy_averaged.pt`: 용접선과 비전처리 표면에 대한 최고 정확도 앙상블 모델 파일

### 직선 피팅 모델
- `checkpoint.pt`: 라인 예측 최소 손실 모델 파일
- `best_f1_checkpoint.pt`: 최고 F1 score 모델 파일

## 🔍 모니터링

프로젝트는 Weights & Biases (wandb)를 사용하여 학습 과정을 모니터링합니다:
- 실시간 손실 추적
- 메트릭 시각화
- 하이퍼파라미터 로깅
- 실험 결과 비교

## 📈 실험 관리

- wandb 프로젝트:
  - 세그멘테이션: "weldline"
  - 직선 피팅: "skeleton_detection"

## ⚠️ 주의사항

- GPU 메모리에 따라 배치 크기 조정 필요
- DataParallel 사용 시 GPU ID 설정 확인
- 데이터셋 경로 설정 시 절대 경로 사용 권장