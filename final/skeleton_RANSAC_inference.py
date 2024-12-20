import sys
import os
import torch
import numpy as np
import cv2
import time
import subprocess
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast
import argparse
import wandb

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from segmentation.utils import calc_dice
from segmentation.unet import UNet
from line_fitting.skeleton_unet import Skeleton_UNet
from line_fitting.skeleton_utils import calc_dice, calc_angle_errors, get_lines_from_mask
from line_fitting.skeleton_dataset import IntersectionDataset
import warnings
warnings.filterwarnings('ignore')


class AngleKalmanFilter: # 칼만필터 사용하여 video(연속된 frame)상에서 직선 위치 튀는 현상 완화
    '''
    Kalman Filter 클래스
    - 목적: 연속된 프레임에서 검출되는 직선의 각도를 안정화
    - 방식: 각도와 각속도를 추정하여 급격한 변화 완화
    '''
    def __init__(self, is_vertical, process_noise_std=0.1, measurement_noise_std=1.0):
        self.is_vertical = is_vertical # vertical/horizontal 직선 구분을 위한 플래그
        
        # 상태 벡터 [각도, 각속도] 초기화
        self.state = np.array([0.0, 0.0]) # 초기 각도와 각속도를 0으로 설정
        
        # 2x2 공분산 행렬 초기화
        # 초기 공분산 행렬 P: 상태 추정의 불확실성을 나타냄
        self.P = np.array([[100.0, 0], # 각도의 초기 불확실성(100.0)이 큼
                        [0, 10.0]])   # 각속도의 초기 불확실성(10.0)은 상대적으로 작음
        
        # 시스템 노이즈 Q: 상태 전이 과정의 불확실성 (상태 변수들이 얼마나 변할 수 있는지를 나타냄)
        self.Q = np.array([[process_noise_std**2, 0],
                        [0, (process_noise_std/10)**2]]) # 각속도의 노이즈는 각도의 1/10
        
        # 측정 노이즈 R: 측정값의 불확실성
        self.R = measurement_noise_std**2 # 측정 노이즈의 분산
    
    def predict(self, dt):
        """
        다음 상태를 예측하는 단계
        
        Args:
            dt: 시간 간격 (초)
        """
        # 상태 전이 행렬 F: 현재 상태로부터 다음 상태를 예측
        F = np.array([[1, dt], # 새로운 각도 = 이전 각도 + dt * 각속도
                    [0, 1]])    # 각속도는 유지
                    
        # 상태 예측: 이전 상태에 상태 전이 행렬을 적용
        self.state = F @ self.state
        
        # 공분산 예측: 불확실성의 전파
        self.P = F @ self.P @ F.T + self.Q # 시스템 노이즈 Q를 더해 불확실성 증가

    def _calculate_angle_diff(self, angle1, angle2):
        """
        두 각도 간의 차이를 계산하는 함수
        
        수직/수평 여부에 따라 다른 계산 방식 적용:
        - 수직선(vertical): 단순 차이 계산
        - 수평선(horizontal): 0도와 180도 근처에서의 불연속 처리
        
        Args:
            angle1: 첫 번째 각도 (도 단위)
            angle2: 두 번째 각도 (도 단위)
        
        Returns:
            float: 두 각도의 차이 (-90 ~ 90도 범위)
        """
        if self.is_vertical: # 수직선의 경우
            return angle1 - angle2 # 단순 차이 계산
        else: # 수평선의 경우 
            # 0도와 180도 근처에서의 불연속 처리
            diff = angle1 - angle2
            if abs(diff) > 90:
                # 180도 근처에서의 차이를 0도 근처로 변환
                if diff > 0:
                    diff = -(180 - diff)
                else:
                    diff = 180 + diff
            return diff
    
    def update(self, measured_angle, dt):
        """
        새로운 측정값으로 상태를 업데이트하는 칼만 필터의 핵심 함수
        
        Args:
            measured_angle: 새로 측정된 각도 (도 단위)
            dt: 시간 간격 (초)
        
        Returns:
            float: 필터링된 각도 (도 단위)
        
        Process:
        1. 상태 예측
        2. 측정값과 예측값의 차이 계산
        3. Kalman gain 계산
        4. 상태 및 공분산 업데이트
        5. 각도 정규화 (0~180도 범위)
        """
        # 상태 예측 단계
        self.predict(dt)
        
        # 측정 행렬 H: 상태 벡터에서 각도만 측정 가능
        H = np.array([1, 0])
        
        # 혁신(Innovation) 계산: 측정값과 예측값의 차이
        innovation = self._calculate_angle_diff(measured_angle, self.state[0])
        
        # Kalman gain 계산: 예측과 측정 중 어느 것을 더 신뢰할지 결정
        S = H @ self.P @ H.T + self.R # S는 혁신 공분산 (예측의 불확실성 + 측정의 불확실성)
        K = (self.P @ H.T) / S        # Kalman gain: 2x1 벡터 [k1, k2]
        
        # 상태 업데이트: 예측값을 혁신값으로 보정
        self.state += K * innovation
        
        # 공분산 업데이트: 새로운 측정으로 불확실성 감소
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P
        
        # 각도를 0~180도 범위로 정규화
        self.state[0] = self.state[0] % 180
        
        return self.state[0]
        
    # def get_state(self):
    #     """현재 상태 정보 반환"""
    #     '''디버깅을 위한 함수'''
    #     return {
    #         'angle': self.state[0],       # 상태 벡터의 첫 번째 원소 (각도)
    #         'velocity': self.state[1],    # 상태 벡터의 두 번째 원소 (각속도)
    #         'uncertainty': np.sqrt(self.P[0,0])  # 각도의 불확실성 (공분산 행렬의 첫 번째 대각 원소의 제곱근)
    #     }

def process_video(seg_model, skeleton_model, video_path, device):
    """
    입력 비디오에 대해 교집합 영역 세그멘테이션 및 skeleton 모델로 직선 검출을 수행하는 함수
    
    Args:
        model: 학습된 U-Net 모델
        video_path: 입력 비디오 파일 경로
        device: 연산 장치 (CPU/GPU)
    
    Returns:
        str or None: 처리된 비디오 파일 경로 또는 실패 시 None
    """
    # 비디오 파일 이름 추출
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 출력 비디오 저장 경로 설정
    video_dir = 'Segmentation_Skeleton_RANSAC_Kalman_video'
    os.makedirs(video_dir, exist_ok=True) # 디렉토리가 없으면 생성
    output_path = os.path.join(video_dir, f"{video_name}.mp4")
    
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return None
    
    # 비디오 속성 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS)) # 초당 프레임 수
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1 # 총 프레임 수
    
    # FFmpeg 설정
    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'768x768', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-', '-c:v', 'mpeg4', '-q:v', '5',
        '-pix_fmt', 'yuv420p', output_path
    ]
    
    # FFmpeg 프로세스 시작
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        print(f"Error starting FFmpeg: {e}")
        return None
    
    # 모델 설정
    seg_model.eval() # segmentation - 평가 모드로 설정
    skeleton_model.eval() # skeleton - 평가 모드로 설정
    MODEL_INPUT_SIZE = (768, 768) # 입력 이미지 크기
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device) # 정규화: ImageNet RGB 채널별 평균값
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device) # 정규화: ImageNet RGB 채널별 표준편차

    # FPS 계산을 위한 변수 초기화
    frame_times = [] # 프레임별 처리 시간 저장
    fps_update_interval = 30  # 30프레임마다 FPS 업데이트

    # Kalman Filter 초기화
    vertical_filter = AngleKalmanFilter(
        is_vertical=True,
        process_noise_std=0.1,    # 상태 변화의 불확실성
        measurement_noise_std=0.7  # 측정값의 불확실성 (작을수록 측정값을 더 신뢰)
    )
    horizontal_filter = AngleKalmanFilter(
        is_vertical=False,
        process_noise_std=0.1,   # 상태 변화의 불확실성
        measurement_noise_std=0.7 # 측정값의 불확실성 (작을수록 측정값을 더 신뢰)
    )

    # 이전 프레임 정보 저장용 변수
    prev_lines = {'vertical': None, 'horizontal': None}
    prev_angles = {'vertical': None, 'horizontal': None}
    prev_filtered_angles = {'vertical': None, 'horizontal': None}
    
    # 직선 안정화를 위한 임계값 설정
    ANGLE_THRESHOLD = 1.0 # 각도 차이 임계값(도 단위): 이전 frame에서 검출된 직선과의 각도 차이가 임계값 이상이면 이전 frame 직선 정보 사용
    POSITION_THRESHOLD = 10 # 중심좌표 차이 임계값(픽셀 단위): 이전 frame에서 검출된 직선과의 중심 좌표 간 거리가 임계값 이상이면 이전 frame 직선 정보 사용
    try:
        for frame_count in tqdm(range(total_frames)):
            frame_start_time = time.time() # 프레임 처리 시작 시간 기록

            # frame별 이미지 불러오기
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame at {frame_count}")
                break
            
            # 프레임 전처리
            # 1. 중앙 크롭 (비디오 프레임을 정사각형으로 만들기)
            h, w = frame.shape[:2] # 프레임의 높이와 너비
            crop_size = min(h, w) # 정사각형 크기 결정
            start_x = (w - crop_size) // 2 # crop 시작 x 좌표
            start_y = (h - crop_size) // 2 # crop 시작 y 좌표
            cropped_frame = frame[start_y:start_y+crop_size, start_x:start_x+crop_size] # input video 사이즈 3840x2160 -> 2160x2160 으로 crop
            
            img = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB) # BGR -> RGB 변환
            img = cv2.resize(img, MODEL_INPUT_SIZE) # 2160x2160 -> 768x768로 크기 조정
            img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device) # 텐서 변환
            img = (img / 255.0 - mean) / std # 정규화
            

            '''1. Segmentation 모델로 예측'''
            with torch.no_grad(): # 그래디언트 계산 비활성화
                weld_output, unpretreated_output = seg_model(img)

            # Segmentation 마스크 생성
            pred_weld = torch.sigmoid(weld_output).squeeze().cpu().numpy()
            pred_unpretreated = torch.sigmoid(unpretreated_output).squeeze().cpu().numpy()
            
            weld_mask = (pred_weld > 0.5).astype(np.uint8)
            unpretreated_mask = (pred_unpretreated > 0.5).astype(np.uint8)
            
            # 교집합 마스크 생성
            intersection_mask = ((weld_mask > 0) & (unpretreated_mask > 0)).astype(np.uint8)

            '''2. Skeleton 모델로 예측'''
            with torch.no_grad(): # 그래디언트 계산 비활성화
                intersection_tensor = torch.from_numpy(intersection_mask).float().unsqueeze(0).unsqueeze(0).to(device)
                skeleton_outputs = skeleton_model(intersection_tensor)
                skeleton_output = skeleton_outputs[0]  # 주 출력만 사용
            
            skeleton_pred = (torch.sigmoid(skeleton_output) > 0.5).float().cpu().numpy().squeeze()

            '''3. RANSAC으로 직선 검출 (skeleton_pred를 입력으로 사용)'''
            detected_lines_dict = get_lines_from_mask(skeleton_pred)
            
            # 원본 이미지 복원 (정규화 역변환)
            original_img = img.squeeze().cpu().numpy()
            original_img = original_img.transpose(1, 2, 0) # (C,H,W) -> (H,W,C)로 변경
            original_img = (original_img * std.squeeze().cpu().numpy() + mean.squeeze().cpu().numpy()) * 255
            original_img = np.clip(original_img, 0, 255).astype(np.uint8)

            # 원본 이미지 복사
            overlay = original_img.copy()

            # Skeleton 연산 수행한 교집합 영역 표시 (녹색)
            overlay[skeleton_pred > 0] = [51, 153, 102]

            # 검출된 직선 정보와 보정된 직선 각도 저장
            current_angles = {'vertical': None, 'horizontal': None}
            current_lines = {'vertical': None, 'horizontal': None}
            filtered_angles = {'vertical': None, 'horizontal': None}


            # 현재 프레임에서 검출된 직선과 각도 계산
            for line in detected_lines_dict:
                direction = line['direction']
                coords = line['coords']
                # 직선의 시작점과 끝점 형식으로 변환
                current_lines[direction] = np.array([[coords[0], coords[1]], 
                                                [coords[2], coords[3]]])
                # 직선 각도 계산
                current_angles[direction] = line['angle']

            '''
            ANGLE_THRESHOLD 이상이면 이전 프레임 각도 사용!!
            POSITION_THRESHOLD 이상이면 이전 프레임 직선 중심좌표 사용!!
            '''
            # 각 방향(vertical/horizontal)에 대해 처리
            for direction in ['vertical', 'horizontal']:
                current_angle = current_angles[direction] # 현재 검출된 각도
                current_line = current_lines[direction] # 현재 검출된 직선
                # 직선별 각각 Kalman filter 가중치 사용
                filter_to_use = vertical_filter if direction == 'vertical' else horizontal_filter
                
                if current_angle is not None: # 첫 번째 프레임이면
                    use_previous = False    # 이전 정보 불러오지 않음
                    
                    # 이전 프레임 정보가 있는 경우, 각도와 중심좌표 차이 확인
                    if prev_angles[direction] is not None and prev_lines[direction] is not None:
                        # 이전 각도와의 차이 계산
                        angle_diff = abs(filter_to_use._calculate_angle_diff(current_angle, prev_angles[direction]))
                        
                        # 중심좌표 차이 확인
                        current_center = ((current_line[0][0] + current_line[1][0])/2, 
                                        (current_line[0][1] + current_line[1][1])/2)
                        prev_center = ((prev_lines[direction][0][0] + prev_lines[direction][1][0])/2,
                                        (prev_lines[direction][0][1] + prev_lines[direction][1][1])/2)
                        
                        center_diff = np.sqrt((current_center[0] - prev_center[0])**2 + 
                                            (current_center[1] - prev_center[1])**2)
                        
                        # 각도 또는 중심좌표 차이가 threshold를 초과하면 이전 프레임 정보 사용
                        use_previous = angle_diff > ANGLE_THRESHOLD or center_diff > POSITION_THRESHOLD
                    
                    if use_previous:
                        # threshold 초과 시 이전 프레임 정보 사용
                        current_angles[direction] = prev_angles[direction]
                        current_lines[direction] = prev_lines[direction]
                        filtered_angles[direction] = prev_filtered_angles[direction]
                    else:
                        # 정상 범위일 때만 필터 업데이트
                        dt = 1.0/fps
                        filtered_angle = filter_to_use.update(current_angle, dt)
                        filtered_angles[direction] = filtered_angle
                        
                        # 현재 정보를 이전 정보로 저장
                        prev_angles[direction] = current_angle
                        prev_lines[direction] = current_line
                        prev_filtered_angles[direction] = filtered_angle

            # 직선 그리기
            for direction in ['vertical', 'horizontal']:
                if current_lines[direction] is not None and filtered_angles[direction] is not None:
                    line = current_lines[direction]

                    # 직선의 중심점 계산
                    center_x = (line[0][0] + line[1][0]) / 2
                    center_y = (line[0][1] + line[1][1]) / 2
                    
                    # 직선의 길이 계산
                    line_length = np.sqrt((line[1][0] - line[0][0])**2 + (line[1][1] - line[0][1])**2)
                    
                    # 필터링된 각도로 새로운 시작점과 끝점 계산
                    angle_rad = np.radians(filtered_angles[direction]) # 도 -> 라디안

                    # 중심점에서 직선 길이의 절반만큼 양쪽으로 확장
                    dx = line_length/2 * np.cos(angle_rad) # x 방향 변위
                    dy = line_length/2 * np.sin(angle_rad) # y 방향 변위
                    
                    # 시작점과 끝점 좌표 계산
                    start_point = (int(center_x - dx), int(center_y - dy))
                    end_point = (int(center_x + dx), int(center_y + dy))
                    
                    # 필터링된 직선 그리기 (검정)
                    cv2.line(overlay, start_point, end_point, (0, 0, 0), 3)

            # 각도 정보 표시 (우측 상단)
            y_offset = 30
            for direction in ['vertical', 'horizontal']:
                if filtered_angles[direction] is not None:
                    display_angle = 180 - filtered_angles[direction]  # 각도 변환 (0~180도)
                    text = f"{direction.capitalize()}: {display_angle:.2f}"
                    # if current_angles[direction] is not None:
                    #     text += f" ({current_angles[direction]:.2f})"

                    # 각도 텍스트 표시 (검정)
                    cv2.putText(overlay, text,
                              (overlay.shape[1] - 250, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    y_offset += 30

            # FPS 계산
            frame_times.append(time.time() - frame_start_time) # 프레임 처리 시간 기록
            if len(frame_times) > fps_update_interval: # 지정된 간격(30프레임) 초과 시
                frame_times = frame_times[-fps_update_interval:] # 최근 기록만 유지
            
            if frame_count % fps_update_interval == 0: # 30 프레임마다
                current_fps = 1 / (sum(frame_times) / len(frame_times)) # 평균 FPS 계산
            else: # 그 외 프레임일 시
                current_fps = 1 / frame_times[-1]  # 현재 프레임의 FPS 계산

            # overlay에 FPS 텍스트 추가 (BGR로 변환하기 직전에 추가)
            cv2.putText(overlay, f"{current_fps:.2f} Hz", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            #  BGR로 변환 (OpenCV 형식)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

            # FFmpeg으로 프레임 전송
            try:
                process.stdin.write(overlay.tobytes()) # 프레임 데이터 전송
                process.stdin.flush() # 버퍼 비우기
            except BrokenPipeError:
                print("FFmpeg process terminated unexpectedly.")
                break

            if process.poll() is not None:
                print("FFmpeg process has terminated.")
                break

    finally:
        cap.release() # 비디오 캡처 객체 해제
        if process.stdin:
            process.stdin.close() # FFmpeg 입력 스트림 닫기
        process.wait() # FFmpeg 프로세스 종료 대기
    
    # 출력 파일 확인
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"Processed video saved to {output_path}")
        return output_path
    else:
        print(f"Failed to create video at {output_path}")
        return None

def test_model(model, test_loader, device):
    """
    스켈레톤 모델의 성능을 테스트하고 F1 score를 계산하는 함수
   
    Args:
        model: 학습된 skeleton 기반 U-Net 모델
        test_loader: 테스트 데이터셋의 데이터로더
        device: 연산 장치 (CPU/GPU)
    
    Returns:
        tuple:
            - avg_f1_score (float): 전체 테스트셋에 대한 평균 F1 점수 (0~1 범위)
            - avg_angle_errors (dict): 방향별 평균 각도 오차
                * 'horizontal': 수평 직선의 평균 각도 오차 (도 단위)
                * 'vertical': 수직 직선의 평균 각도 오차 (도 단위)
    """
    model.eval() # 모델 평가 모드
    total_angle_errors = { # 각 방향(수평/수직)별 각도 오차를 누적할 딕셔너리 초기화
        'horizontal': {'total_error': 0, 'count': 0},
        'vertical': {'total_error': 0, 'count': 0}
    }
    total_f1_score = 0 # 전체 F1 score를 누적할 변수 초기화
    
    with torch.no_grad(): # 추론을 위한 그래디언트 계산 비활성화
        for batch in tqdm(test_loader, desc="Testing"): # 테스트 데이터셋의 각 배치에 대해 반복
            # 교집합 마스크(입력)와 라벨링 마스크(ground truth)를 디바이스로 이동
            intersection_mask = batch['input'].float().to(device)
            skeleton_mask = batch['target'].float().to(device)
            
            with autocast(): # FP32(단정밀도)와 FP16(반정밀도)를 혼합 사용하여 메모리 사용량 감소 & 연산 속도 향상
                outputs = model(intersection_mask) # 모델 예측 수행 (순전파)
                output = outputs[0]  # 메인 출력만 사용 (768x768)
            
            pred_binary = (torch.sigmoid(output) > 0.5).float() # 예측값을 이진 마스크로 변환 (임계값 0.5)
            batch_f1 = calc_dice(pred_binary, skeleton_mask) # 현재 배치의 F1 점수 계산
            total_f1_score += batch_f1.item() # 전체 F1 점수에 누적

            # 예측값과 정답을 CPU의 NumPy 배열로 변환
            pred = pred_binary.cpu().numpy()
            gt = skeleton_mask.cpu().numpy()
            
            # 배치 내 각 샘플에 대해 각도 오차 계산
            for i in range(pred.shape[0]):
                errors = calc_angle_errors(pred[i], gt[i])
                
                # 수평선이 검출된 경우 오차 누적
                if errors['horizontal_error'] is not None:
                    total_angle_errors['horizontal']['total_error'] += errors['horizontal_error']
                    total_angle_errors['horizontal']['count'] += 1
                # 수직선이 검출된 경우 오차 누적
                if errors['vertical_error'] is not None:
                    total_angle_errors['vertical']['total_error'] += errors['vertical_error']
                    total_angle_errors['vertical']['count'] += 1

    # 평균 F1 score 계산
    avg_f1_score = total_f1_score / len(test_loader)
    
    # 각 방향별 평균 각도 오차 계산
    avg_angle_errors = {}
    for direction in ['horizontal', 'vertical']:
        if total_angle_errors[direction]['count'] > 0:
            # 해당 방향의 직선이 하나 이상 검출된 경우, 평균 계산
            avg_angle_errors[direction] = total_angle_errors[direction]['total_error'] / total_angle_errors[direction]['count']
        else:
            # 해당 방향의 직선이 하나도 검출되지 않은 경우, 무한대로 설정
            avg_angle_errors[direction] = float('inf')

    return avg_f1_score, avg_angle_errors

def visualize_predictions(model, test_loader, device, save_dir='visualization_results'):
    """
    모델의 예측 결과를 시각화하는 함수
    
    Args:
        model: 학습된 스켈레톤 모델
        test_loader: 테스트 데이터 로더
        device: 연산 장치 (CPU/GPU)
        save_dir: 시각화 결과를 저장할 디렉토리
    
    Returns:
        avg_f1_score: 전체 테스트 데이터에 대한 평균 F1 score
        avg_angle_errors: 방향별(vertical/horizontal) 평균 각도 오차
    """
    model.eval() # 모델 평가 모드
    os.makedirs(save_dir, exist_ok=True) # 저장 디렉토리 생성
    
    # 성능 메트릭 누적을 위한 변수 초기화
    total_f1_score = 0
    total_angle_errors = {
        'horizontal': {'total_error': 0, 'count': 0},
        'vertical': {'total_error': 0, 'count': 0}
    }

    with torch.no_grad(): # 추론을 위한 그래디언트 계산 비활성화
        # 각 배치에 대해 처리
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Visualizing")):
            # 1. 입력 데이터 준비 및 GPU 이동
            intersection_mask = batch['input'].float().to(device) # 교집합 마스크
            skeleton_mask = batch['target'].float().to(device) # Ground Truth 라벨링 마스크
            
            # 2. mixed precision으로 모델 예측 수행
            with autocast(): # FP32(단정밀도)와 FP16(반정밀도)를 혼합 사용하여 메모리 사용량 감소 & 연산 속도 향상
                outputs = model(intersection_mask) # 모델 예측 수행 (순전파)
                output = outputs[0] # 메인 출력만 사용 (768x768)
            
            # 3. 예측값을 이진화하고 F1 score 계산
            pred_binary = (torch.sigmoid(output) > 0.5).float() # 예측을 이진화
            batch_f1 = calc_dice(pred_binary, skeleton_mask)    # batch의 F1 score 계산
            total_f1_score += batch_f1.item()

             # 4. CPU의 NumPy 배열로 변환
            pred = pred_binary.squeeze(1).cpu().numpy() # 예측 마스크
            gt = skeleton_mask.squeeze(1).cpu().numpy() # Ground Truth 마스크
            input_mask = intersection_mask.squeeze(1).cpu().numpy() # 입력 마스크

            # 5. 배치에서 랜덤하게 1개의 샘플만 선택
            random_idx = np.random.randint(0, pred.shape[0])
            selected_pred = pred[random_idx] # 선택된 예측 마스크
            selected_gt = gt[random_idx]  # 선택된 Ground Truth 마스크
            selected_input = (input_mask[random_idx] * 255).astype(np.uint8)
            
            # 6. 입력 마스크를 3채널 BGR 이미지로 변환
            input_mask_3ch = cv2.cvtColor(selected_input, cv2.COLOR_GRAY2BGR)

            # 이미지 크기와 여백 설정
            h, w = input_mask_3ch.shape[:2]
            margin = 60  # 각도 표시를 위한 상단 여분 공간
            
            # 7. 시각화를 위한 4개의 서브 이미지 준비
            # 7.1 Ground Truth 라벨링 마스크
            vis_gt_skeleton = np.zeros((h + margin, w, 3), dtype=np.uint8) + 255 # 흰색 배경
            vis_gt_skeleton[margin:] = input_mask_3ch.copy()
            vis_gt_skeleton[margin:][selected_gt > 0.5] = [0, 255, 0]  # Ground Truth: 녹색
            
            # 7.2 예측된 마스크
            vis_pred_skeleton = np.zeros((h + margin, w, 3), dtype=np.uint8) + 255
            vis_pred_skeleton[margin:] = input_mask_3ch.copy()
            vis_pred_skeleton[margin:][selected_pred > 0.5] = [255, 0, 0]  # Prediction: 파란색
            
            # 7.3 Ground Truth 직선 visualization
            vis_gt_lines = np.zeros((h + margin, w, 3), dtype=np.uint8) + 255
            vis_gt_lines[margin:] = input_mask_3ch.copy()
            gt_lines = get_lines_from_mask(selected_gt)
            
            # Ground Truth 직선과 각도 정보 그리기
            gt_text = "GT Lines: "
            for line in gt_lines:
                x1, y1, x2, y2 = [int(coord) for coord in line['coords']]  # 좌표를 정수로 변환
                # x1, y1, x2, y2 = line['coords']
                cv2.line(vis_gt_lines, (x1, y1 + margin), (x2, y2 + margin), (0, 255, 0), 2)  # BGR: 녹색
                gt_text += f"{line['direction']}: {line['angle']:.2f}"
            cv2.putText(vis_gt_lines, gt_text, (10, margin-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # 7.4 예측된 직선 표시
            vis_pred_lines = np.zeros((h + margin, w, 3), dtype=np.uint8) + 255
            vis_pred_lines[margin:] = input_mask_3ch.copy()
            pred_lines = get_lines_from_mask(selected_pred)
            
            # 예측된 직선과 각도 정보 그리기
            pred_text = "Pred Lines: "
            for line in pred_lines:
                x1, y1, x2, y2 = [int(coord) for coord in line['coords']]  # 좌표를 정수로 변환
                # x1, y1, x2, y2 = line['coords']
                cv2.line(vis_pred_lines, (x1, y1 + margin), (x2, y2 + margin), (255, 0, 0), 2)  # BGR: 파란색
                pred_text += f"{line['direction']}: {line['angle']:.2f}"
            cv2.putText(vis_pred_lines, pred_text, (10, margin-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # 8. 4개의 이미지를 2x2 그리드로 결합
            top_row = np.hstack([vis_gt_skeleton, vis_pred_skeleton])
            bottom_row = np.hstack([vis_gt_lines, vis_pred_lines])
            combined_vis = np.vstack([top_row, bottom_row])
            
            # 9. 제목 추가 (샘플 번호와 F1 점수)
            title = f"Sample {batch_idx} - F1 Score: {batch_f1.item():.4f}"
            cv2.putText(combined_vis, title, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # 10. 결합된 시각화 이미지 저장
            cv2.imwrite(os.path.join(save_dir, f'sample_{batch_idx}.png'), combined_vis)

            # 11. wandb를 사용하는 경우 로그 기록
            if args.use_wandb:
                wandb.log({f"visualization_{batch_idx}": wandb.Image(combined_vis)})

            # 12. 각도 오차 계산 및 누적
            errors = calc_angle_errors(selected_pred, selected_gt)
            if errors['horizontal_error'] is not None:
                total_angle_errors['horizontal']['total_error'] += errors['horizontal_error']
                total_angle_errors['horizontal']['count'] += 1
            if errors['vertical_error'] is not None:
                total_angle_errors['vertical']['total_error'] += errors['vertical_error']
                total_angle_errors['vertical']['count'] += 1

    # 평균 F1 score 계산
    avg_f1_score = total_f1_score / len(test_loader)
    
    # 평균 각도 오차 계산
    avg_angle_errors = {}
    for direction in ['horizontal', 'vertical']:
        if total_angle_errors[direction]['count'] > 0:
            avg_angle_errors[direction] = total_angle_errors[direction]['total_error'] / total_angle_errors[direction]['count']
        else:
            avg_angle_errors[direction] = float('inf')

    return avg_f1_score, avg_angle_errors

# main 함수를 통해 실행
if __name__ == "__main__":
    # 명령줄 인자 파서 생성
    parser = argparse.ArgumentParser(description='Inference for skeleton detection')
    # 이미지 관련 인자
    parser.add_argument('--intersection_mask_dir', type=str, default='line_fitting/data/pred_intersection_masks', help='Directory containing intersection masks') # 테스트 이미지 디렉토리 경로
    parser.add_argument('--skeleton_mask_dir', type=str, default='line_fitting/data/line_masks', help='Directory containing line masks') # Ground Truth 라벨링 마스크 디렉토리 경로
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference') # 추론 시 배치 크기
    
    # 저장 디렉토리 관련 인자
    parser.add_argument('--save_dir', type=str, default='skeleton_inference_results', help='Directory to save results') # 결과 저장 디렉토리 경로
    
    # 비디오 관련 인자
    parser.add_argument('--input_video', type=str, default='in_video/10_shape.mp4', help='Path to the input video file') # 입력 비디오 파일 경로
    
    # 모델 관련 인자
    parser.add_argument('--seg_model_path', type=str, default='segmentation/models/best_model_accuracy_averaged.pt', help='Path to the saved segmentation model') # 학습된 segemntation 모델 가중치 파일 경로
    parser.add_argument('--skeleton_model_path', type=str, default='line_fitting/models/best_f1_checkpoint.pt', help='Path to the saved skeleton model') # 학습된 skeleton 직선 피팅 모델 가중치 파일 경로
    
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb for logging') # wandb 사용 여부 (플래그)
    args = parser.parse_args() # 인자 파싱

    # 예측 결과 저장 디렉토리 설정
    prediction_save_dir = os.path.join(args.save_dir, 'predictions')
    os.makedirs(args.save_dir, exist_ok=True) # 저장 디렉토리 생성

    # GPU 설정
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        device = torch.device("cuda:0")  # 메인 GPU를 cuda:0으로 설정
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.use_wandb:
        wandb.init(project="skeleton_inference", config=vars(args))

    try:
        # # 데이터셋 로드 및 분할
        # dataset = IntersectionDataset(intersection_mask_dir=args.intersection_mask_dir, skeleton_mask_dir=args.skeleton_mask_dir)

        # torch.manual_seed(42)
        
        # # 데이터셋 분할 (main.py와 동일한 비율 사용)
        # total_size = len(dataset)
        # train_size = int(0.8 * total_size) # 학습 데이터 80%
        # val_size = int(0.1 * total_size) # 검증 데이터 10%
        # test_size = total_size - train_size - val_size # 테스트 데이터 10%

        # _, _, test_dataset = random_split(dataset, [train_size, val_size, test_size]) # 테스트 데이터만 사용

        # # 테스트 데이터 로더 생성
        # test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
        # print(f"Loaded {len(test_dataset)} test samples")

        # Segmentation 모델 로드
        seg_model = UNet(in_channels=3, num_classes=2) # 3채널 입력, 2채널 출력
        seg_state_dict = torch.load(args.seg_model_path, map_location=device) # 저장된 모델 가중치 로드
        new_seg_state_dict = {k.replace('module.', ''): v for k, v in seg_state_dict.items()}
        seg_model.load_state_dict(new_seg_state_dict) # 모델에 가중치 적용
        seg_model = seg_model.to(device)

        # Skeleton 모델 로드
        skeleton_model = Skeleton_UNet(in_channels=1, num_classes=1) # 1채널 입력, 1채널 출력
        skeleton_state_dict = torch.load(args.skeleton_model_path, map_location=device) # 저장된 모델 가중치 로드
        skeleton_model.load_state_dict(skeleton_state_dict) # 모델에 가중치 적용
        skeleton_model = skeleton_model.to(device)

        # DataParallel 설정(병렬 처리)
        if torch.cuda.device_count() > 1:
            seg_model = nn.DataParallel(seg_model, device_ids=[0, 1]) # 사용할 GPU 번호 설정 (GPU 0, 1 사용)
            skeleton_model = nn.DataParallel(skeleton_model, device_ids=[0, 1]) # 사용할 GPU 번호 설정 (GPU 0, 1 사용)
            torch.cuda.set_device(0) # 메인 GPU 설정
            
        print(f"Segmentation Model loaded from {args.seg_model_path}")
        print(f"Skeleton Model loaded from {args.skeleton_model_path}")

        # # 모델 성능 테스트 함수 호출
        # test_f1_score, test_angle_errors = test_model(skeleton_model, test_loader, device)

        # # 테스트 결과 출력 및 저장
        # print("\nTest Results:")
        # results_str = "Test Results:\n"
        # print(f"Test F1 Score: {test_f1_score:.4f}")
        # results_str += f"Test F1 Score: {test_f1_score:.4f}\n"

        # for direction in ['horizontal', 'vertical']:
        #     if test_angle_errors[direction] != float('inf'):
        #         print(f"{direction.capitalize()} Angle Error: {test_angle_errors[direction]:.2f}°")
        #         results_str += f"{direction.capitalize()} Angle Error: {test_angle_errors[direction]:.2f}°\n"

        # # 모델 성능 저장 디렉토리 내에 txt 파일에 기록
        # with open(os.path.join(args.save_dir, 'test_results.txt'), 'w') as f:
        #     f.write(results_str)

        # if args.use_wandb:
        #     # 테스트 결과 wandb에 로깅
        #     wandb.log({
        #         "test/f1_score": test_f1_score,
        #         **{f"test/{direction}_angle_error": error
        #            for direction, error in test_angle_errors.items()
        #            if error != float('inf')}
        #     })

        # # 시각화 수행 - 배치당 1개 샘플만 랜덤 선택하여 시각화
        # print("\nGenerating visualizations...")
        # visualization_dir = os.path.join(args.save_dir, 'visualizations')
        # vis_f1_score, vis_angle_errors = visualize_predictions(skeleton_model, test_loader, device, save_dir=visualization_dir) # 모델 예측 결과 시각화 함수 호출
       
        # print("\nVisualization Results:")
        # print(f"Visualization F1 Score: {vis_f1_score:.4f}")
        # for direction in ['horizontal', 'vertical']:
        #     if vis_angle_errors[direction] != float('inf'):
        #         print(f"{direction.capitalize()} Angle Error: {vis_angle_errors[direction]:.2f}°")

        # 동영상 처리 수행
        print("Processing video...")
        output_path = process_video(seg_model, skeleton_model, args.input_video, device)
        if output_path:
            print(f"Video processing completed. Output saved to {output_path}")
        else:
            print("Video processing failed.")

    # 예외 처리
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

    if args.use_wandb:
        wandb.finish() # wandb 세션 종료