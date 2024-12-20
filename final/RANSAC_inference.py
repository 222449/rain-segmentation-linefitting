import sys
import os
import torch
import numpy as np
import cv2
import time
import subprocess
import torch.nn as nn
from tqdm import tqdm
import argparse
from sklearn.linear_model import RANSACRegressor

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from segmentation.unet import UNet
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
        else:# 수평선의 경우 
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

def process_video(model, video_path, device):
    """
    입력 비디오에 대해 교집합 영역 세그멘테이션 및 직선 검출을 수행하는 함수
    
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
    video_dir = 'Segmentation_RANSAC_Kalman_video'
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
    model.eval() # 평가 모드로 설정
    MODEL_INPUT_SIZE = (768, 768) # 입력 이미지 크기
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device) # 정규화: ImageNet RGB 채널별 평균값
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device) # 정규화: ImageNet RGB 채널별 표준편차

    # FPS 계산을 위한 변수 초기화
    frame_times = [] # 프레임별 처리 시간 저장
    fps_update_interval = 30  # 30 프레임마다 FPS 업데이트

    # Kalman Filter 초기화
    vertical_filter = AngleKalmanFilter(
        is_vertical=True,
        process_noise_std=0.1,    # 상태 변화의 불확실성
        measurement_noise_std=0.7  # 측정값의 불확실성 (작을수록 측정값을 더 신뢰)
    )
    horizontal_filter = AngleKalmanFilter(
        is_vertical=False,
        process_noise_std=0.1, # 상태 변화의 불확실성
        measurement_noise_std=0.7 # 측정값의 불확실성 (작을수록 측정값을 더 신뢰)
    )

    # 이전 프레임 정보 저장용 변수
    prev_lines = {'vertical': None, 'horizontal': None}
    prev_angles = {'vertical': None, 'horizontal': None}
    prev_filtered_angles = {'vertical': None, 'horizontal': None}
    
    # 직선 안정화를 위한 임계값 설정
    ANGLE_THRESHOLD = 1.0 # 각도 차이 임계값(도 단위): 이전 frame에서 검출된 직선과의 각도 차이가 임계값 이상이면 이전 frame 직선 정보 사용
    POSITION_THRESHOLD = 30 # 중심좌표 차이 임계값(픽셀 단위): 이전 frame에서 검출된 직선과의 중심 좌표 간 거리가 임계값 이상이면 이전 frame 직선 정보 사용

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
            
            # 모델 예측
            with torch.no_grad(): # test이므로 그래디언트 계산 비활성화
                weld_output, unpretreated_output = model(img)
            
            # sigmoid를 통해 확률값으로 변환 
            pred_weld = torch.sigmoid(weld_output).squeeze().cpu().numpy()
            pred_unpretreated = torch.sigmoid(unpretreated_output).squeeze().cpu().numpy()
            
            # 임계값(0.5) 적용 후 255 곱하여 scaling -> 이진 마스크 생성 (0 또는 255)
            weld_mask = (pred_weld > 0.5).astype(np.uint8) * 255 # pixel값이 0.5 이상인 pixel에 대해 용접선 클래스 부여
            unpretreated_mask = (pred_unpretreated > 0.5).astype(np.uint8) * 255 # pixel값이 0.5 이상인 pixel에 대해 비전처리 표면 클래스 부여
            
            # 원본 이미지 복원 (정규화 역변환)
            original_img = img.squeeze().cpu().numpy()
            original_img = original_img.transpose(1, 2, 0) # (C,H,W) -> (H,W,C)로 변경
            original_img = (original_img * std.squeeze().cpu().numpy() + mean.squeeze().cpu().numpy()) * 255
            original_img = np.clip(original_img, 0, 255).astype(np.uint8)

            # 원본 이미지 복사
            overlay = original_img.copy()

            # # 용접선 마스크 오버레이 (파란색)
            # weld_overlay = np.zeros_like(overlay) # 원본 이미지와 같은 크기의 빈 배열 생성
            # weld_overlay[weld_mask > 0] = [0, 0, 255]  # 용접선 영역을 파란색으로 설정
            # overlay = cv2.addWeighted(overlay, 1, weld_overlay, 0.75, 0) # 75% 투명도로 오버레이

            # # 비전처리 표면 마스크 오버레이 (빨간색)
            # unpretreated_overlay = np.zeros_like(overlay) # 원본 이미지와 같은 크기의 빈 배열 생성
            # unpretreated_overlay[unpretreated_mask > 0] = [255, 0, 0]  # 비전처리 표면 영역을 빨간색으로 설정
            # overlay = cv2.addWeighted(overlay, 1, unpretreated_overlay, 0.25, 0) # 25% 투명도로 오버레이

            # RANSAC용 교집합 마스크 생성 (용접선과 비전처리 표면의 교집합)
            intersection_mask = ((weld_mask > 0) & (unpretreated_mask > 0)).astype(np.uint8)
            overlay[intersection_mask > 0] = [51, 153, 102]  # 교집합 영역 표시 (녹색)

            # RANSAC으로 직선 검출
            detected_lines, v_midpoints, h_midpoints = detect_lines(intersection_mask)

            # 검출된 직선 정보와 보정된 직선 각도 저장
            current_angles = {'vertical': None, 'horizontal': None}
            current_lines = {'vertical': None, 'horizontal': None}
            filtered_angles = {'vertical': None, 'horizontal': None}


            # 현재 프레임에서 검출된 직선과 각도 계산
            for line in detected_lines:
                # 시작점과 끝점 좌표 추출
                start_point = (int(line[0][0]), int(line[0][1]))
                end_point = (int(line[1][0]), int(line[1][1]))
                
                # 직선의 각도 계산
                dx = end_point[0] - start_point[0] # x 방향 변화량
                dy = end_point[1] - start_point[1] # y 방향 변화량
                angle = np.degrees(np.arctan2(dy, dx)) # 각도 계산 (라디안 -> 도)
                
                # 각도를 0~180도 범위로 정규화
                if angle < 0:
                    angle += 180

                # 수직선/수평선 분류
                if 45 <= angle < 135:  # 수직선 (45도~135도)
                    current_angles['vertical'] = angle
                    current_lines['vertical'] = line
                else:  # 수평선 (0도~45도 또는 135도~180도)
                    current_angles['horizontal'] = angle
                    current_lines['horizontal'] = line

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

'''RANSAC 알고리즘'''
def find_horizontal_transitions(line, y_coord):
    """수평 라인에서 흑백 전환점 찾기
    Args:
        line: y_coord에서의 수평 라인 (768 픽셀)
        y_coord: 현재 검사 중인 y 좌표
    Returns:
        transitions: [(x1, y), (x2, y)] 형태의 전환점 쌍 리스트
    """
    transitions = [] # 전환점 쌍을 저장할 리스트
    start_point = None  # 현재 처리 중인 0->1 전환점

    # 모든 0->1, 1->0 전환점을 한 번에 찾기    
    changes = np.where(line[1:] != line[:-1])[0] + 1 # line[1:] != line[:-1]로 인접한 픽셀 간의 값 변화 감지
    
    # 2개씩 묶어서 쌍으로 처리 (0->1 전환과 1->0 전환을 쌍으로)
    for i in range(0, len(changes)-1, 2):
        if i+1 < len(changes):
            if line[changes[i]-1] == 0:  # 0->1 확인
                transitions.append([(changes[i], y_coord), (changes[i+1], y_coord)])
    
    return transitions

def find_vertical_transitions(line, x_coord):
    """수직 라인에서 흑백 전환점 찾기
    Args:
        line: x_coord에서의 수직 라인 (768 픽셀)
        x_coord: 현재 검사 중인 x 좌표
    Returns:
        transitions: [(x, y1), (x, y2)] 형태의 전환점 쌍 리스트
    """
    transitions = [] # 전환점 쌍을 저장할 리스트
    start_point = None  # 현재 처리 중인 0->1 전환점    

    # 모든 0->1, 1->0 전환점을 한 번에 찾기 
    changes = np.where(line[1:] != line[:-1])[0] + 1 # line[1:] != line[:-1]로 인접한 픽셀 간의 값 변화 감지
    
    # 2개씩 묶어서 쌍으로 처리 (0->1 전환과 1->0 전환을 쌍으로)
    for i in range(0, len(changes)-1, 2):
        if i+1 < len(changes):
            if line[changes[i]-1] == 0:  # 0->1 확인
                transitions.append([(x_coord, changes[i]), (x_coord, changes[i+1])])
    
    return transitions

def get_midpoints(transition_pairs):
    """전환점 쌍들의 중점 좌표를 계산
    
    Args:
        transition_pairs: 전환점 쌍들의 리스트 [[(x1,y1), (x2,y2)], ...]
    
    Returns:
        midpoints: 중점 좌표들의 리스트 [(mid_x, mid_y), ...]
    """
    midpoints = []
    for pair in transition_pairs:
        if len(pair) == 2: # 유효한 전환점 쌍인지 확인
            x1, y1 = pair[0] # 시작점
            x2, y2 = pair[1] # 끝점
            # 중점 계산 (정수 나눗셈 //)
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            
            # 중점 찾기
            if x1 == x2:  # vertical transitions인 경우 (x좌표 동일)
                midpoints.append((mid_x, mid_y))      # 중점
            else:  # horizontal transitions인 경우 (y좌표 동일)
                midpoints.append((mid_x, mid_y))      # 중점
    return midpoints

def normalize_horizontal_midpoints(midpoints):
    """수평 방향 중점들의 y좌표 편차를 감소시키는 정규화
    
    Args:
        midpoints: 원본 중점 좌표들의 리스트 [(x, y), ...]
    
    Returns:
        horizontal_normalized_points: y좌표가 정규화된 중점들의 배열
    """
    if not midpoints:
        return []
    
    points = np.array(midpoints)
    
    # y좌표의 중앙값(median) 계산
    y_median = np.median(points[:, 1])
    
    # 각 점들의 y좌표와 중앙값의 차이를 계산
    y_diff = points[:, 1] - y_median
    
    # 편차를 0.4 비율로 줄임 (또는 다른 적절한 비율로 조정 가능)
    reduction_factor = 0.4  # 조정 가능한 파라미터
    reduced_diff = y_diff * reduction_factor
    
    # 줄어든 편차를 중앙값에 더해 새로운 y좌표 계산
    horizontal_normalized_points = points.copy()
    horizontal_normalized_points[:, 1] = y_median + reduced_diff
    
    return horizontal_normalized_points

def fit_ransac_line(points, vertical=True):
    """RANSAC 알고리즘을 사용하여 points에 가장 적합한 직선을 찾는 함수
   
   Args:
       points: 직선을 피팅할 점들의 좌표 numpy 배열 [(x1,y1), (x2,y2), ...]
       vertical: 수직선 여부 (True: 수직, False: 수평)
   
   Returns:
       line_points: 검출된 직선의 양 끝점 좌표 [[x1,y1], [x2,y2]] 또는 None
       slope: 직선의 기울기 (ransac.estimator_.coef_[0]) 또는 None
       score: 피팅 점수 또는 float('inf')
   """
    
    # 직선 피팅에 최소 2개의 점이 필요
    if len(points) < 2:
        return None, None, float('inf')
    
    points = np.array(points) # 리스트를 numpy 배열로 변환
    
    try:
        # RANSAC 회귀 모델 초기화
        ransac = RANSACRegressor(
            residual_threshold=7.0,      # inlier로 판단할 점과 직선 사이 최대 허용 거리(픽셀)
            min_samples=2,               # 최소 2개 점으로 직선 피팅
            max_trials=200,              # RANSAC 알고리즘 최대 반복 횟수
            stop_probability=0.99,       # 최적해를 찾았다고 판단할 확률 임계값
            random_state=42
        )
        
        if vertical:
            # 수직선: x = my + b 형태로 피팅
            X = points[:, 0].reshape(-1, 1) # x좌표를 독립변수로
            y = points[:, 1]                # y좌표를 종속변수로
            
            ransac.fit(X, y) # RANSAC 피팅 수행
            
            # 직선의 양 끝점 계산
            x_min, x_max = points[:, 0].min(), points[:, 0].max() # x좌표 범위
            y_min = ransac.predict([[x_min]])[0] # 최소 x에서의 y 예측
            y_max = ransac.predict([[x_max]])[0] # 최대 x에서의 y 예측
            line_points = np.array([[x_min, y_min], [x_max, y_max]])
        else:
            # 수평선: y = mx + b 형태로 피팅
            X = points[:, 1].reshape(-1, 1)  # y좌표를 독립변수로
            y = points[:, 0]                 # x좌표를 종속변수로
            
            ransac.fit(X, y) # RANSAC 피팅 수행

            # 직선의 양 끝점 계산
            y_min, y_max = points[:, 1].min(), points[:, 1].max() # y좌표 범위
            x_min = ransac.predict([[y_min]])[0] # 최소 y에서의 x 예측
            x_max = ransac.predict([[y_max]])[0] # 최대 y에서의 x 예측
            
            # x좌표와 y좌표 순서로 point 구성
            line_points = np.array([[x_min, y_min], [x_max, y_max]]) # 검출된 직선을 구성하는 두 좌표
        
        # 모델 검증 1: inlier 개수가 너무 적으면 None 반환
        n_inliers = np.sum(ransac.inlier_mask_)
        if n_inliers < 10:  # 최소 10개의 inlier 필요
            return None, None, float('inf')
        
        # 모델 검증 2: 피팅 점수 계산 (R² score)
        score = ransac.score(X, y)
        
        # 모델 검증 3: 직선의 각도 계산 및 검사
        dx = line_points[1][0] - line_points[0][0] # x 변화량
        dy = line_points[1][1] - line_points[0][1] # y 변화량
        angle = np.degrees(np.arctan2(dy, dx))     # 각도 계산 (라디안 -> 도)
        
        '''검출중인 직선의 각도 debugging'''
        # print(f"- angle: {angle:.2f}")

        # 각도를 0-180도 범위로 변환
        if angle < 0:
            angle += 180
            
        # 수직/수평 여부에 따른 각도 검증
        if vertical:
            if not (45 <= angle <= 135):  # 수직선은 90도에 가까워야 함
                return None, None, float('inf')
        else:
            if not (0 <= angle < 45 or 135 < angle <= 180):  # 수평선은 0도나 180도에 가까워야 함
                return None, None, float('inf')
            
        return line_points, ransac.estimator_.coef_[0], score, # 검출된 직선을 구성하는 두 좌표, 기울기, 점수
    except:
        return None, None, float('inf')

def find_mask_boundary_points(line_points, binary_mask, margin=2):
    """RANSAC으로 검출한 직선을 마스크 영역 경계까지 확장
    
    Args:
        line_points: RANSAC으로 검출된 직선의 시작점과 끝점 [[x1,y1], [x2,y2]]
        binary_mask: 이진 마스크
        margin: 직선 주변 탐색 범위 (픽셀)
    
    Returns:
        np.array: 확장된 직선의 양 끝점 좌표 [[x1,y1], [x2,y2]] 또는 None
    """
    height, width = binary_mask.shape
    x1, y1 = line_points[0] # 시작점
    x2, y2 = line_points[1] # 끝점
    
    # 수직/수평 판단 (x 변화보다 y 변화가 큰지)
    if abs(x2 - x1) < abs(y2 - y1): # Vertical line이면 
        # y 방향으로 스캔
        y_range = range(height) if y2 > y1 else range(height-1, -1, -1)
        valid_points = []
        # 직선의 방정식 계산 (y = mx + b)
        m = (x2 - x1) / (y2 - y1) if y2 != y1 else float('inf')
        b = x1 - m * y1 if y2 != y1 else x1
        
        # y 좌표를 따라가며 마스크 내의 점 찾기
        for y in y_range:
            x = int(round(m * y + b)) if y2 != y1 else int(round(x1))
            if 0 <= x < width: # x가 이미지 범위 내인지 확인
                x_start = max(0, x-margin)
                x_end = min(width, x+margin+1)
                # margin 내에 마스크 값이 1인 픽셀이 있는지 확인
                if np.any(binary_mask[y, x_start:x_end] == 1):
                    valid_points.append([x, y])
    
    else: # Horizontal line이면
        # x 방향으로 스캔
        x_range = range(width) if x2 > x1 else range(width-1, -1, -1)
        valid_points = []
        # 직선의 방정식 계산 (y = mx + b)
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        
        # x 좌표를 따라가며 마스크 내의 점 찾기
        for x in x_range:
            y = int(round(m * x + b))
            if 0 <= y < height: # y가 이미지 범위 내인지 확인
                y_start = max(0, y-margin)
                y_end = min(height, y+margin+1)
                # margin 내에 마스크 값이 1인 픽셀이 있는지 확인
                if np.any(binary_mask[y_start:y_end, x] == 1):
                    valid_points.append([x, y])
    
    if len(valid_points) < 2: # 유효한 점이 2개 미만이면 실패
        return None
    
    # 첫 점과 마지막 점을 양 끝점으로 사용
    return np.array([valid_points[0], valid_points[-1]])

def detect_lines(binary_mask):
    """이진 마스크에서 수직/수평 직선 검출
    
    Args:
        binary_mask: 이진 마스크 (0과 1로 구성)
    
    Returns:
        lines: 검출된 직선들의 좌표 리스트 [vertical_line, horizontal_line]
        vertical_midpoints: 수직 방향 중점들의 좌표
        horizontal_midpoints: 수평 방향 중점들의 좌표
    
    Process:
    1. 수직 직선 검출:
       - 수평 방향으로 스캔하여 전환점 찾기
       - 중점 계산
       - RANSAC으로 직선 피팅
       
    2. 수평 직선 검출:
       - 수직 방향으로 스캔하여 전환점 찾기
       - 중점 계산
       - 수직선이 있는 경우, 정규화 적용 여부 결정
       - RANSAC으로 직선 피팅
       
    3. 검출된 직선들 마스크 경계까지 확장
    """
    height, width = binary_mask.shape
    
    # 원본 마스크 복사 (horizontal 검출용)
    mask_for_horizontal = binary_mask.copy()
    
    # 1. 수직 직선 검출
    vertical_transitions = []
    for y in range(height):
        row = binary_mask[y, :]
        transitions = find_horizontal_transitions(row, y)
        vertical_transitions.extend(transitions)
    
    vertical_midpoints = get_midpoints(vertical_transitions)
    vertical_line = None

    if vertical_midpoints:
        line_points, slope, score = fit_ransac_line(vertical_midpoints, vertical=True)
        if line_points is not None:
            final_line = find_mask_boundary_points(line_points, binary_mask)
            if final_line is not None:
                vertical_line = final_line

    # 2. 수평 직선 검출 - 두 단계 접근
    horizontal_transitions = []
    for x in range(width):
        col = mask_for_horizontal[:, x]
        transitions = find_vertical_transitions(col, x)
        horizontal_transitions.extend(transitions)
    
    # 첫 번째 시도: 정규화 없이 원본 midpoint로 검출
    horizontal_midpoints = get_midpoints(horizontal_transitions)
    horizontal_line = None

    if horizontal_midpoints:
        if vertical_line is not None:  # vertical line이 존재하는 경우 (십자형)
            # 우선 정규화 없이 RANSAC 적용
            line_points, slope, score = fit_ransac_line(horizontal_midpoints, vertical=False)
                        
            if line_points is not None:
                # 각도 계산
                dx = line_points[1][0] - line_points[0][0]
                dy = line_points[1][1] - line_points[0][1]
                dx = - dx
                angle = np.degrees(np.arctan2(dy, dx))
                if angle < 0:
                    angle += 180
                    
                # 각도가 거의 수평이면 (0~7도 또는 173~180도 사이)
                if (0 <= angle <= 7) or (173 <= angle <= 180):
                    # 두 번째 시도: 정규화된 midpoint로 다시 시도
                    normalized_points = normalize_horizontal_midpoints(horizontal_midpoints)
                    
                    # 이전 결과 초기화
                    line_points = None
                    horizontal_line = None
                    
                    # 정규화된 점들로 RANSAC 다시 적용
                    line_points, slope, score = fit_ransac_line(normalized_points, vertical=False)
        
        else:  # vertical line이 없는 경우 (일자형)
            # 정규화 없이 바로 RANSAC 적용
            line_points, slope, score = fit_ransac_line(horizontal_midpoints, vertical=False)

        # 3. 검출된 직선들 마스크 경계까지 확장
        if line_points is not None:
            final_line = find_mask_boundary_points(line_points, binary_mask)
            if final_line is not None:
                horizontal_line = final_line
    
    # 검출된 라인들 모으기
    lines = []

    if vertical_line is not None:
        lines.append(vertical_line)

    if horizontal_line is not None:
        lines.append(horizontal_line)

    return lines, vertical_midpoints, horizontal_midpoints

# main 함수를 통해 실행
if __name__ == "__main__":
    # 명령줄 인자 파서 생성
    parser = argparse.ArgumentParser(description='Video inference for weldline detection')

    # 비디오 관련 인자
    parser.add_argument('--input_video', type=str, default='in_video/10_shape.mp4', help='Path to the input video file') # 입력 비디오 파일 경로

    # 모델 관련 인자
    parser.add_argument('--model_path', type=str, default='segmentation/models/best_model_accuracy_averaged.pt', help='Path to the saved averaged model') # 학습된 모델 가중치 파일 경로

    args = parser.parse_args() # 인자 파싱

    # GPU 설정
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        device = torch.device("cuda:0")  # 메인 GPU를 cuda:0으로 설정
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # 모델 로드
        model = UNet(in_channels=3, num_classes=2) # 3채널 입력, 2채널 출력
        state_dict = torch.load(args.model_path, map_location=device) # 저장된 모델 가중치 로드
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict) # 모델에 가중치 적용
        model = model.to(device)

        # DataParallel 설정(병렬 처리)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0, 1]) # 사용할 GPU 번호 설정 (GPU 0, 1 사용)
            torch.cuda.set_device(0)  # 메인 GPU 설정
            
        print(f"Model loaded from {args.model_path}")

        # 동영상 처리 수행
        print("Processing video...")
        output_path = process_video(model, args.input_video, device)
        if output_path:
            print(f"Video processing completed. Output saved to {output_path}")
        else:
            print("Video processing failed.")

    # 예외 처리
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise