import os
import re
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset


class IntersectionDataset(Dataset):
    """
    용접선과 미전처리 표면의 교집합 부분 스켈레톤화를 위한 커스텀 데이터셋

    교집합 마스크와 라벨링 마스크 파일의 이름에서 숫자만 매칭하여 쌍을 이루도록 구현
    예: 교집합 마스크 디렉토리 내의 00001.png와 라벨링 마스크 디렉토리 내의 00001.png가 한 쌍으로 매칭됨
    
    데이터 구조:
    1. 입력: 용접선과 비전처리 표면의 교집합 마스크
    2. 출력: 예측 직선 마스크 (4개의 해상도)
       - 원본 크기 (H x W)
       - 1/2 크기 (H/2 x W/2)
       - 1/4 크기 (H/4 x W/4)
       - 1/8 크기 (H/8 x W/8)

    특징:
        - 교집합 마스크와 라벨링 마스크를 쌍으로 제공
        - 여러 스케일의 ground truth 생성
        - 이미지 이진화 전처리
        - 파일명 숫자 매칭을 통한 데이터 쌍 구성

    Args:
        intersection_mask_dir: 용접선과 미전처리 표면의 교집합 마스크 디렉토리 경로
        skeleton_mask_dir: 라벨링 마스크 디렉토리 경로
        transform: 이미지 변환 함수
    """
    def __init__(self, intersection_mask_dir, skeleton_mask_dir, transform=None):
        # 디렉토리 경로 저장
        self.intersection_mask_dir = intersection_mask_dir  # 교집합 마스크 디렉토리
        self.skeleton_mask_dir = skeleton_mask_dir  # 라벨링 마스크 디렉토리
        
        # 기본 텐서 변환 정의
        self.mask_transform = transforms.ToTensor() # PIL Image를 텐서로 변환

        def extract_number(filename):
            """파일 이름에서 숫자 추출"""
            return int(re.search(r'\d+', filename).group())

        # 각 디렉토리의 파일들을 숫자 기준으로 매핑
        intersection_mask_files = {extract_number(f): f for f in os.listdir(intersection_mask_dir) if f.endswith('.png')}
        skeleton_mask_files = {extract_number(f): f for f in os.listdir(skeleton_mask_dir) if f.endswith('.png')}

        # 두 디렉토리에 모두 존재하는 숫자만 선택
        valid_numbers = set(intersection_mask_files.keys()) & set(skeleton_mask_files.keys())

        # 유효한 파일 쌍으로 목록 생성
        self.valid_files = [(intersection_mask_files[num], skeleton_mask_files[num]) for num in valid_numbers]

    def __len__(self):
        """데이터셋의 총 샘플 수 반환"""
        return len(self.valid_files)

    def __getitem__(self, idx):
        """
        지정된 인덱스의 데이터 샘플 반환
        
        Args:
            idx: 데이터 인덱스
        
        Returns:
            dict: {
                'input': 교차 마스크 텐서 [1, H, W],
                'target': 스켈레톤 마스크 텐서 [1, H, W],
                'target_384': 1/2 크기의 타겟 [1, H/2, W/2],
                'target_192': 1/4 크기의 타겟 [1, H/4, W/4],
                'target_96': 1/8 크기의 타겟 [1, H/8, W/8],
                'path': 입력 마스크 경로
            }
        """
        # 인덱스에 해당하는 파일 경로들 가져오기
        intersection_mask_name, skeleton_mask_name = self.valid_files[idx]
        
        # 파일 경로 생성
        intersection_mask_path = os.path.join(self.intersection_mask_dir, intersection_mask_name)
        skeleton_mask_path = os.path.join(self.skeleton_mask_dir, skeleton_mask_name)
        
        # 마스크 로드 및 변환
        intersection_mask = Image.open(intersection_mask_path).convert("L")  # 그레이스케일로 변환
        skeleton_mask = Image.open(skeleton_mask_path).convert("L")  # 그레이스케일로 변환
        
        # transform 적용 - 텐서로 변환
        intersection_mask = self.mask_transform(intersection_mask) # 교집합 마스크
        skeleton_mask = self.mask_transform(skeleton_mask) # 라벨링 마스크

        intersection_mask = (intersection_mask > 0.5).float() # 임계값 0.5로 이진화
        skeleton_mask = (skeleton_mask > 0.5).float()  # 임계값 0.5로 이진화

        # 다중 스케일 ground truth 생성 
        H, W = skeleton_mask.shape[1:]
        skeleton_mask_384 = transforms.Resize((H//2, W//2), interpolation=transforms.InterpolationMode.NEAREST)(skeleton_mask) # 최근접 이웃 보간법 사용 (0과 1 값만 유지)
        skeleton_mask_192 = transforms.Resize((H//4, W//4), interpolation=transforms.InterpolationMode.NEAREST)(skeleton_mask) # 최근접 이웃 보간법 사용 (0과 1 값만 유지)
        skeleton_mask_96 = transforms.Resize((H//8, W//8), interpolation=transforms.InterpolationMode.NEAREST)(skeleton_mask) # 최근접 이웃 보간법 사용 (0과 1 값만 유지)

        return {
            'input': intersection_mask,  # [1, H, W]
            'target': skeleton_mask,     # [1, H, W]
            'target_384': skeleton_mask_384,  # [1, H/2, W/2]
            'target_192': skeleton_mask_192,    # [1, H/4, W/4]
            'target_96': skeleton_mask_96,    # [1, H/8, W/8]
            'path': intersection_mask_path    # 디버깅/시각화용 경로
        }