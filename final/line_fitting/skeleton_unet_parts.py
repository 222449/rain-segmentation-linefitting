import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    이중 컨볼루션 블록
    Conv -> BN -> ReLU -> Conv -> BN -> ReLU 구조
    
    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv_op = nn.Sequential(
            # 첫 번째 컨볼루션 블록
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), # 3x3 컨볼루션
            nn.BatchNorm2d(out_channels), # 배치 정규화
            nn.ReLU(inplace=False), # ReLU 활성화 함수

            # 두 번째 컨볼루션 블록
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), # 3x3 컨볼루션
            nn.BatchNorm2d(out_channels), # 배치 정규화
            nn.ReLU(inplace=False) # ReLU 활성화 함수
        )

    def forward(self, x):
        return self.conv_op(x) # 순차적으로 모든 연산 적용

class AttentionGroup(nn.Module):
    """
    Attention Group 모듈

    동작 원리:
    1. 입력 특징맵을 세 개의 병렬 경로로 처리
    2. 각 경로는 독립적인 특징 추출 수행
    3. Attention 가중치로 세 경로의 특징을 결합
    4. Residual connection으로 그래디언트 흐름 개선

    특징:
    - 다중 경로 특징 추출로 표현력 향상
    - Attention mechanism으로 경로 간 가중치 조절
    - 1x1 컨볼루션으로 경량화된 가중치 생성
    - Residual conneciton으로 학습 안정성 확보
    
    Args:
        num_channels: 입력/출력 채널 수
    """
    def __init__(self, num_channels):
        super(AttentionGroup, self).__init__()
        # 세 개의 병렬 특징 추출 경로
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1), # 첫 번째 head의 특징 추출 경로
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=False)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1), # 두 번째 head의 특징 추출 경로
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=False)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1), # 세 번째 head의 특징 추출 경로
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=False)
        )

        # Attention 가중치를 생성하는 1x1 컨볼루션
        self.conv_1x1 = nn.Conv2d(num_channels, 3, kernel_size=1) # 3개 경로에 대한 가중치 생성

    def forward(self, x):
        """
        Args:
            x (Tensor): 입력 특징맵 [B, C, H, W]
            
        Returns:
            Tensor: Attention이 적용된 특징맵 [B, C, H, W]
        """
        # 세 가지 특징 추출 경로
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        # Attention 가중치 계산 및 적용
        s = torch.softmax(self.conv_1x1(x), dim=1) # 세 경로에 대한 소프트맥스 활성화함수 적용
        
        # 가중치를 적용하여 특징맵 결합
        att = s[:,0,:,:].unsqueeze(1) * x1 + s[:,1,:,:].unsqueeze(1) * x2 + s[:,2,:,:].unsqueeze(1) * x3

        # 잔차 연결(residual connection)
        return x + att

class ChannelAttention(nn.Module):
    """
    채널 Attention 모듈: 각 채널의 중요도를 학습하여 특징을 강화/억제

    동작 원리:
    1. 전역 평균 풀링과 전역 최대 풀링으로 채널별 전역 정보 추출
    2. 공유 MLP로 채널 간 관계 학습
    3. 시그모이드로 채널별 가중치 생성
    
    구조:
    Input → GAP/GMP → Shared MLP → Sigmoid → Scale
    
    특징:
    - 이중 풀링으로 완성도 높은 채널 정보 추출
    - 채널 압축-복원으로 효율적인 계산
    - SE(Squeeze-and-Excitation) 블록 기반의 검증된 구조
    
    Args:
        in_planes: 입력 채널 수
        ratio: 채널 압축 비율 (기본값: 16)
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 전역 평균 풀링, 출력 크기 Cx1x1
        self.max_pool = nn.AdaptiveMaxPool2d(1) # 전역 최대 풀링, 출력 크기 Cx1x1
        
        # 공유 MLP (Multi-Layer Perceptron)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), # 채널 수를 ratio로 줄임 (특징 압축)
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False) # 원래 채널 수로 복원 (특징 복원)
        )
        self.sigmoid = nn.Sigmoid() # 0~1 사이의 attention 가중치 생성

    def forward(self, x):
        """
        Args:
            x (Tensor): 입력 특징맵 [B, C, H, W]
        
        Returns:
            Tensor: 채널 attention 가중치 [B, C, 1, 1]
        """
        avg_out = self.fc(self.avg_pool(x)) # 평균 풀링 경로
        max_out = self.fc(self.max_pool(x)) # 최대 풀링 경로
        out = avg_out + max_out # 두 경로의 출력을 합산
        return self.sigmoid(out) # Sigmoid를 적용하여 0~1 사이의 가중치 생성 -> 중요한 채널은 강조하고 덜 중요한 채널은 억제

class SpatialAttention(nn.Module):
    """
    공간 Attention 모듈: 특징맵의 각 위치의 중요도를 학습
    
    동작 원리:
    1. 채널 방향 평균/최대값으로 공간 특징 압축
    2. 컨볼루션으로 공간적 관계 학습
    3. 시그모이드로 공간별 가중치 생성
    
    특징:
    - 채널 방향의 평균과 최댓값 정보 활용
    - 패딩으로 경계 정보 보존
    - 위치별 attention 가중치 생성
    - 7x7 컨볼루션으로 넓은 수용 영역 확보
    
    구현 세부사항:
    - 입력: [B, C, H, W]
    - 채널 집계: [B, 2, H, W]
    - 출력: [B, 1, H, W]의 attention map

    Args:
        kernel_size: 컨볼루션 커널 크기 (기본값: 7)
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 2개 채널(avg, max)을 입력받아 1개 채널의 attention map 생성
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (Tensor): 입력 특징맵 [B, C, H, W]
            
        Returns:
            Tensor: 공간 attention 가중치 [B, 1, H, W]
        """
        avg_out = torch.mean(x, dim=1, keepdim=True) # 채널 방향으로 평균값 계산 (1xHxW)
        max_out, _ = torch.max(x, dim=1, keepdim=True) # 채널 방향으로 최대값 계산 (1xHxW)
        x = torch.cat([avg_out, max_out], dim=1)  # 평균값과 최대값을 채널 방향으로 연결
        x = self.conv1(x) # 7x7 컨볼루션 적용하여 단일 채널의 attention map 생성 -> 공간적 관계 학습
        return self.sigmoid(x) # Sigmoid를 적용하여 0~1 사이의 가중치 생성 -> 중요한 공간의 위치는 강조하고 덜 중요한 위치는 억제

class DownSample(nn.Module):
    """
    다운샘플링 블록

    아키텍처:
    Input → DoubleConv → AttentionGroup → MaxPool → Output
    
    채널 변화:
    H×W×C → H×W×C' → H×W×C' → H/2×W/2×C'
    
    특징:
    1. 특징 추출
       - 이중 컨볼루션으로 특징 학습
       - 배치 정규화로 학습 안정화
       - ReLU 활성화 함수 적용
    
    2. 특징 강화
       - Attention으로 중요 특징 강조
       - 3개의 병렬 경로로 다양한 특징 추출
       - 잔차 연결(residual connection)로 그래디언트 전파 개선
    
    3. 해상도 축소
       - MaxPooling으로 공간 크기 축소와 동시에 주요 특징 보존
       - 공간 크기 1/2로 축소
       - 파라미터 수 감소
    
    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
    
    Returns:
        tuple: (
            down: skip connection용 특징맵 [B, C', H, W],
            p: 다운샘플링된 특징맵 [B, C', H/2, W/2]
        )
    """
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels) # 특징 추출
        self.att_group = AttentionGroup(out_channels) # Attention 적용
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 맥스 풀링을 통한 크기 축소

    def forward(self, x):
        """
        Args:
            x (Tensor): 입력 특징맵 [B, C, H, W]
            
        Returns:
            tuple: (skip connection용 특징맵, 다운샘플링된 특징맵)
        """
        down = self.conv(x) # 이중 컨볼루션 적용
        down = self.att_group(down) # Attention 적용
        p = self.pool(down) # 맥스 풀링으로 크기 축소
        return down, p # Skip connection용 특징맵과 다음 층의 입력 반환

class UpSample(nn.Module):
    """
    업샘플링 블록

    아키텍처:
    Input → TransposedConv → Concat → DoubleConv → ChannelAtt → SpatialAtt → Output
    
    채널/해상도 변화:
    H×W×C → 2H×2W×C/2 → 2H×2W×C → 2H×2W×C' → 2H×2W×C' → 2H×2W×C'
    
    특징:
    1. 해상도 복원
       - 전치 컨볼루션으로 해상도 2배 증가
       - 학습 가능한 업샘플링
    
    2. 특징 결합
       - Skip connection으로 세부 정보 보존
       - bilinear interpolation으로 크기 맞추기
       - 채널 방향으로 연결하여 정보 통합
    
    3. 특징 정제
       - 이중 컨볼루션으로 특징 추출
       - Channel Attention으로 채널 간 중요도 학습
       - Spatial Attention으로 위치별 중요도 학습
       - Channel/Spatial Attention 결합
    
    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
    
    Returns:
        Tensor: Attention이 적용된 출력 특징맵 [B, C', 2H, 2W]
    """
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2) # 해상도 복원을 위한 전치 컨볼루션
        self.conv = DoubleConv(in_channels, out_channels) # 특징 추출
        self.ca = ChannelAttention(out_channels) # 채널 attention
        self.sa = SpatialAttention() # 공간 attention

    def forward(self, x1, x2):
        """
        Args:
            x1 (Tensor): 업샘플링할 특징맵 [B, C, H, W]
            x2 (Tensor): Skip connection 특징맵 [B, C/2, 2H, 2W]
            
        Returns:
            Tensor: Attention이 적용된 출력 특징맵
        """
        x1 = self.up(x1) # 특징맵 크기와 해상도 복원
        x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True) # x2를 x1의 크기에 맞게 조정 (양선형 보간) - 주변 4개 픽셀의 가중 평균을 사용하여 새로운 값 계산
        x = torch.cat([x1, x2], dim=1) # 특징맵 채널 방향 결합
        x = self.conv(x) # 이중 컨볼루션 적용
        x = self.ca(x) * x # 채널 attention 적용
        x = self.sa(x) * x # 공간 attention 적용
        return x