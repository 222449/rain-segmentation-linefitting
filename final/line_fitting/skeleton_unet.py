import sys
import os
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from skeleton_unet_parts import DoubleConv, AttentionGroup, DownSample, UpSample


class Skeleton_UNet(nn.Module):
    """
    스켈레톤화를 위한 변형된 U-Net 구조

    아키텍처 특징:
    1. 인코더
       - 4개의 다운샘플링 블록
       - 각 블록: DoubleConv → Attention → MaxPool
       - 채널 수: 1 → 64 → 128 → 256 → 512
    
    2. 병목층
       - 1024 채널의 특징 맵 생성
       - Attention 메커니즘으로 중요 특징 강화
    
    3. 디코더
       - 4개의 업샘플링 블록
       - 각 블록: TransposedConv → Concat → DoubleConv → Channel/Spatial Attention
       - 채널 수: 1024 → 512 → 256 → 128 → 64
    
    4. Auxiliary task learning
       - 4개의 출력 헤드 (다중 스케일 예측)
       - 원본, 1/2, 1/4, 1/8 크기의 예측 생성
    
    Args:
        in_channels: 입력 이미지의 채널 수
        num_classes: 출력 클래스 수 (이진 분류의 경우 1)
    """
    def __init__(self, in_channels, num_classes):
        super(Skeleton_UNet, self).__init__()
        
        # 인코더 (다운샘플링 경로)
        self.down1 = DownSample(in_channels, 64) # 입력 -> 64채널, 크기: H×W → H/2×W/2
        self.down2 = DownSample(64, 128) # 64 -> 128채널, 크기: H/2×W/2 → H/4×W/4
        self.down3 = DownSample(128, 256) # 128 -> 256채널, 크기: H/4×W/4 → H/8×W/8
        self.down4 = DownSample(256, 512) # 256 -> 512채널, 크기: H/8×W/8 → H/16×W/16
        
        # 병목층 (브리지)
        self.bottleneck = DoubleConv(512, 1024) # 512 -> 1024채널, 특징 추출 강화
        self.bottleneck_att = AttentionGroup(1024) # 병목층에 attention 적용
        
        # 디코더 (업샘플링 경로)
        self.up1 = UpSample(1024, 512) # 1024 -> 512채널, 크기: H/16×W/16 → H/8×W/8
        self.up2 = UpSample(512, 256) # 512 -> 256채널, 크기: H/8×W/8 → H/4×W/4
        self.up3 = UpSample(256, 128) # 256 -> 128채널, 크기: H/4×W/4 → H/2×W/2
        self.up4 = UpSample(128, 64) # 128 -> 64채널, 크기: H/2×W/2 → H×W
        
        # 최종 출력층 (1×1 컨볼루션)
        self.out = nn.Conv2d(64, num_classes, kernel_size=1) # 64 → 1채널
        
        # 보조 출력
        self.aux_out_384 = nn.Conv2d(128, num_classes, kernel_size=1) # 1/2 크기 출력
        self.aux_out_192 = nn.Conv2d(256, num_classes, kernel_size=1) # 1/4 크기 출력
        self.aux_out_96 = nn.Conv2d(512, num_classes, kernel_size=1) # 1/8 크기 출력

    def forward(self, x):
        """
        순전파 함수
        
        Args:
            x (Tensor): 입력 이미지 [B, C, H, W]
            
        Returns:
            tuple: (주 출력, 보조 출력1, 보조 출력2, 보조 출력3)
                - 주 출력: 원본 크기
                - 보조 출력1: 1/2 크기
                - 보조 출력2: 1/4 크기
                - 보조 출력3: 1/8 크기
        """
        # 인코더 경로 - 특징 추출 및 다운샘플링
        down1, p1 = self.down1(x) # 첫 번째 다운샘플링
        down2, p2 = self.down2(p1) # 두 번째 다운샘플링
        down3, p3 = self.down3(p2) # 세 번째 다운샘플링
        down4, p4 = self.down4(p3) # 네 번째 다운샘플링
        
        # 병목층 - 가장 깊은 특징 추출
        bottle = self.bottleneck(p4) # 병목층에서 이중 컨볼루션을 통한 특징 추출
        bottle = self.bottleneck_att(bottle) # Attention mechanism 적용
        
        # 디코더 경로 - 특징 복원 및 업샘플링 (각 단계마다 인코더의 대응되는 특징과 결합)
        up1 = self.up1(bottle, down4) # 첫 번째 업샘플링 및 특징 결합
        up2 = self.up2(up1, down3) # 두 번째 업샘플링 및 특징 결합
        up3 = self.up3(up2, down2) # 세 번째 업샘플링 및 특징 결합
        up4 = self.up4(up3, down1) # 네 번째 업샘플링 및 특징 결합
        
        # 메인 출력 - 원본 크기로 복원된 세그멘테이션 맵
        out = self.out(up4)
        
        # 보조 출력 - 다양한 해상도의 예측을 생성
        aux_384 = self.aux_out_384(up3) # 384x384 크기의 보조 출력
        aux_192 = self.aux_out_192(up2) # 192x192 크기의 보조 출력
        aux_96 = self.aux_out_96(up1) # 96x96 크기의 보조 출력
        
        return out, aux_384, aux_192, aux_96