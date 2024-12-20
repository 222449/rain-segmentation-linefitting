import os
import sys
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from unet_parts import DoubleConv, DownSample, UpSample


class UNet(nn.Module):
    """
    용접선과 비전처리 표면을 동시에 검출하는 이중 분기 구조인 1-2 branches U-Net

    특징:
    1. 단일 인코더를 공유하여 특징 추출 효율성 증대
    2. 두 개의 독립적인 디코더를 통해 각 태스크에 특화된 특징 학습
    3. Skip connection을 통한 고해상도 특징 보존

    아키텍처:
    - 인코더: 4개의 다운샘플링 블록 (해상도 1/2씩 감소, 채널 수 2배씩 증가)
    - 브릿지: 각 태스크별 독립적인 병목 계층
    - 디코더: 각 태스크별 4개의 업샘플링 블록 (해상도 2배씩 증가, 채널 수 1/2씩 감소)

    Args:
        in_channels: 입력 이미지의 채널 수 (RGB의 경우 3)
        num_classes: 출력 클래스 수 (현재 구현에서는 사용되지 않음, 각 브랜치가 1채널 이진 마스크 출력)
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # 공유 인코더 부분
        self.down_convolution_1 = DownSample(in_channels, 64)  # 용접선 & 비전처리 표면 클래스 다운샘플링 레이어 1
        self.down_convolution_2 = DownSample(64, 128)  # 용접선 & 비전처리 표면 클래스 다운샘플링 레이어 2
        self.down_convolution_3 = DownSample(128, 256)  # 용접선 & 비전처리 표면 클래스 다운샘플링 레이어 3
        self.down_convolution_4 = DownSample(256, 512)  # 용접선 & 비전처리 표면 클래스 다운샘플링 레이어 4

        # 용접선 검출을 위한 bottlenceck과 디코더 브랜치
        self.weld_bottle_neck = DoubleConv(512, 1024)  # 용접선 클래스 bottleneck 레이어
        self.weld_up_convolution_1 = UpSample(1024, 512)  # 용접선 클래스 업샘플링 레이어 1
        self.weld_up_convolution_2 = UpSample(512, 256)  # 용접선 클래스 업샘플링 레이어 2
        self.weld_up_convolution_3 = UpSample(256, 128)  # 용접선 클래스 업샘플링 레이어 3
        self.weld_up_convolution_4 = UpSample(128, 64)  # 용접선 클래스 업샘플링 레이어 4
        self.weld_out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)  # 용접선 클래스 출력 레이어

        # 비전처리 표면 검출을 위한 bottlenceck과 디코더 브랜치
        self.unpretreated_bottle_neck = DoubleConv(512, 1024)  # 비전처리 표면 클래스 bottleneck 레이어
        self.unpretreated_up_convolution_1 = UpSample(1024, 512)  # 비전처리 표면 클래스 업샘플링 레이어 1
        self.unpretreated_up_convolution_2 = UpSample(512, 256)  # 비전처리 표면 클래스 업샘플링 레이어 2
        self.unpretreated_up_convolution_3 = UpSample(256, 128)  # 비전처리 표면 클래스 업샘플링 레이어 3
        self.unpretreated_up_convolution_4 = UpSample(128, 64)  # 비전처리 표면 클래스 업샘플링 레이어 4
        self.unpretreated_out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)  # 비전처리 표면 클래스 출력 레이어

    def forward(self, x):
        """
        순전파 함수
        인코더에서 추출된 특징을 두 개의 디코더로 전달하여
        용접선과 비전처리 표면을 각각 검출
        """
        # 인코더 부분 - 특징 추출 및 다운샘플링
        down_1, p1 = self.down_convolution_1(x) # 1번째 다운샘플링 (입력 크기의 1/2)
        down_2, p2 = self.down_convolution_2(p1) # 2번째 다운샘플링 (입력 크기의 1/4)
        down_3, p3 = self.down_convolution_3(p2) # 3번째 다운샘플링 (입력 크기의 1/8)
        down_4, p4 = self.down_convolution_4(p3) # 4번째 다운샘플링 (입력 크기의 1/16)
        
        # 용접선 브랜치
        weld_b = self.weld_bottle_neck(p4) # 병목층 통과
        weld_up_1 = self.weld_up_convolution_1(weld_b, down_4) # 1번째 업샘플링 + skip connection
        weld_up_2 = self.weld_up_convolution_2(weld_up_1, down_3) # 2번째 업샘플링 + skip connection
        weld_up_3 = self.weld_up_convolution_3(weld_up_2, down_2) # 3번째 업샘플링 + skip connection
        weld_up_4 = self.weld_up_convolution_4(weld_up_3, down_1) # 4번째 업샘플링 + skip connection
        weld_out = self.weld_out(weld_up_4) # 최종 출력층 (1채널 이진 마스크)
        
        # 비전처리 표면 브랜치
        unpretreated_b = self.unpretreated_bottle_neck(p4) # 병목층 통과
        unpretreated_up_1 = self.unpretreated_up_convolution_1(unpretreated_b, down_4) # 1번째 업샘플링 + skip connection
        unpretreated_up_2 = self.unpretreated_up_convolution_2(unpretreated_up_1, down_3) # 2번째 업샘플링 + skip connection
        unpretreated_up_3 = self.unpretreated_up_convolution_3(unpretreated_up_2, down_2) # 3번째 업샘플링 + skip connection
        unpretreated_up_4 = self.unpretreated_up_convolution_4(unpretreated_up_3, down_1) # 4번째 업샘플링 + skip connection
        unpretreated_out = self.unpretreated_out(unpretreated_up_4) # 최종 출력층 (1채널 이진 마스크)

        return weld_out, unpretreated_out