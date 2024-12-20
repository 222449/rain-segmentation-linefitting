import os
import re
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset


class WeldlineDataset(Dataset):
    """
    용접선과 비전처리 표면 검출을 위한 커스텀 데이터셋

    이미지와 마스크 파일의 이름에서 숫자만 매칭하여 쌍을 이루도록 구현
    예: image_00001.png와 weldline_mask_00001.png, unpretreated_mask_00001.png가 한 쌍으로 매칭됨
    
    Args:
        image_dirs: 원본 이미지 디렉토리 경로
        weld_mask_dirs: 용접선 마스크 디렉토리 경로
        unpretreated_mask_dirs: 비전처리 표면 마스크 디렉토리 경로
        transform: 이미지 변환을 위한 transform 함수 (기본값 None인 경우 ImageNet 평균/표준편차로 정규화하는 기본 transform 사용)
    """
    def __init__(self, image_dirs, weld_mask_dirs, unpretreated_mask_dirs, transform=None):
        # 입력 경로들을 리스트 형태로 변환
        self.image_dirs = image_dirs if isinstance(image_dirs, list) else [image_dirs]
        self.weld_mask_dirs = weld_mask_dirs if isinstance(weld_mask_dirs, list) else [weld_mask_dirs]
        self.unpretreated_mask_dirs = unpretreated_mask_dirs if isinstance(unpretreated_mask_dirs, list) else [unpretreated_mask_dirs]
        
        # transform이 지정되지 않은 경우 기본 transform 설정 (정규화 포함)
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(), # PIL 이미지를 텐서로 변환 (값 범위: 0~1))
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], # ImageNet RGB 채널별 평균값
                std=[0.229, 0.224, 0.225] # ImageNet RGB 채널별 표준편차
                )
        ])
        self.mask_transform = transforms.ToTensor() # 마스크용 transform (정규화 없이 텐서 변환만 수행)

        # 유효한 이미지-마스크 쌍 찾기
        self.valid_files = []

        # 각 디렉토리 세트에 대해 반복
        for img_dir, weld_dir, unpretreated_dir in zip(self.image_dirs, self.weld_mask_dirs, self.unpretreated_mask_dirs):

            # 각 디렉토리의 파일들을 숫자 기준으로 매핑
            image_files = {self.extract_number(f): f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))}
            weld_mask_files = {self.extract_number(f): f for f in os.listdir(weld_dir) if f.endswith('.png')}
            unpretreated_mask_files = {self.extract_number(f): f for f in os.listdir(unpretreated_dir) if f.endswith('.png')}
            
            # 세 디렉토리에 모두 존재하는 숫자만 선택
            valid_numbers = set(image_files.keys()) & set(weld_mask_files.keys()) & set(unpretreated_mask_files.keys())

            # 유효한 파일 경로들을 리스트에 추가
            self.valid_files.extend([
                (os.path.join(img_dir, image_files[num]), # 이미지 경로
                 os.path.join(weld_dir, weld_mask_files[num]), # 용접선 마스크 경로
                 os.path.join(unpretreated_dir, unpretreated_mask_files[num])) # 비전처리 표면 마스크 경로
                for num in valid_numbers
            ])

    def extract_number(self, filename):
        """파일 이름에서 숫자 추출"""
        return int(re.search(r'\d+', filename).group())

    def __len__(self):
        """데이터셋의 총 샘플 수 반환"""
        return len(self.valid_files)

    def __getitem__(self, idx):
        """
        주어진 인덱스의 샘플 반환
        
        Returns:
            tuple: (이미지, 용접선 마스크, 비전처리 표면 마스크, 이미지 경로)
        """
        # 인덱스에 해당하는 파일 경로들 가져오기
        img_path, weld_mask_path, unpretreated_mask_path = self.valid_files[idx]
        
        # 이미지와 마스크 파일 로드
        img = Image.open(img_path).convert("RGB") # RGB 이미지로 변환
        weld_mask = Image.open(weld_mask_path).convert("L") # 그레이스케일로 변환
        unpretreated_mask = Image.open(unpretreated_mask_path).convert("L") # 그레이스케일로 변환
        
        # transform 적용
        img = self.transform(img) # 이미지 정규화

        # 마스크 이진화 (0 또는 1로 변환)
        weld_mask = self.mask_transform(weld_mask)        
        weld_mask = (weld_mask > 0.5).float() # 임계값 0.5로 이진화
        unpretreated_mask = self.mask_transform(unpretreated_mask)
        unpretreated_mask = (unpretreated_mask > 0.5).float() # 임계값 0.5로 이진화

        return img, weld_mask, unpretreated_mask, img_path