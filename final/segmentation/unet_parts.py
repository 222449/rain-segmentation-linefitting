import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    연속된 두 개의 3x3 컨볼루션 레이어 블록

    각 컨볼루션 연산 후:
    1. 배치 정규화로 학습 안정성 향상
    2. ReLU 활성화로 비선형성 추가
    3. 3x3 커널과 패딩=1로 특징맵 크기 유지

    Args:
        in_channels (int): 입력 특징맵의 채널 수
        out_channels (int): 출력 특징맵의 채널 수
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            # 첫 번째 컨볼루션 레이어 블록
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 3x3 컨볼루션
            nn.BatchNorm2d(out_channels), # 배치 정규화
            nn.ReLU(inplace=True),  # ReLU 활성화 함수

            # 두 번째 컨볼루션 레이어 블록
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # 3x3 컨볼루션
            nn.BatchNorm2d(out_channels), # 배치 정규화
            nn.ReLU(inplace=True)  # ReLU 활성화 함수
        )

    def forward(self, x):
        # 연속된 두 개의 3x3 컨볼루션, 배치 정규화, ReLU 적용
        return self.conv_op(x)

class DownSample(nn.Module):
    """
    U-Net 인코더의 다운샘플링 블록

    동작 과정:
    1. DoubleConv로 특징 추출
    2. MaxPooling으로 공간 해상도 1/2로 축소
    3. Skip connection을 위해 풀링 전 특징맵 저장

    Args:
        in_channels: 입력 특징맵의 채널 수
        out_channels: 출력 특징맵의 채널 수

    Returns:
        tuple: (skip_features, pooled_features)
            - skip_features: 디코더의 skip connection에 사용될 풀링 전 특징맵
            - pooled_features: 다음 인코더 층으로 전달될 다운샘플링된 특징맵
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # max pooling

    def forward(self, x):
        down = self.conv(x) # 특징 추출
        p = self.pool(down) # 다운샘플링으로 크기 축소 (1/2)
        return down, p # skip connection용 특징맵과 다운샘플링된 특징맵 반환

class UpSample(nn.Module):
    """
    U-Net 디코더의 업샘플링 블록
    
    동작 과정:
    1. 전치 컨볼루션으로 특징맵 크기 2배 확장
    2. 인코더의 동일 레벨 특징맵을 보간법으로 크기 맞춤
    3. 두 특징맵을 채널 방향으로 연결 (concatenation)
    4. DoubleConv로 최종 특징 추출
    
    Args:
        in_channels: 입력 특징맵의 채널 수
        out_channels: 출력 특징맵의 채널 수
    
    참고:
        bilinear 보간법 사용 시 align_corners=True로 설정하여 
        업샘플링된 특징맵의 경계가 원본과 정확히 일치하도록 함
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2) # 전치 컨볼루션
        self.conv = DoubleConv(in_channels, out_channels) # DoubleConv

    def forward(self, x1, x2):
       x1 = self.up(x1) # 업샘플링(특징맵 크기 2배 확장)
       x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)  # x2를 x1의 크기에 맞게 조정 (양선형 보간) - 주변 4개 픽셀의 가중 평균을 사용하여 새로운 값 계산
       x = torch.cat([x1, x2], dim=1)  # 채널 방향으로 특징맵 결합
       return self.conv(x) # 특징 추출 및 통합