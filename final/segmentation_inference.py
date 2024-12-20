import sys
import os
import torch
import random
import numpy as np
import cv2
import time
import subprocess
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
import argparse
import wandb

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from segmentation.unet import UNet
from segmentation.dataset import WeldlineDataset
from segmentation.utils import calc_iou, calc_dice, calc_accuracy, calc_precision_recall


def process_video(model, video_path, device):
    """비디오를 프레임별로 처리하여 segmentation 결과를 시각화하는 함수
    
    Args:
        model: 학습된 U-Net 모델
        video_path: 입력 비디오 파일 경로
        device: 연산 장치 (CPU/GPU)
    
    Returns:
        str or None: 처리된 비디오 파일 경로 또는 실패 시 None
    
    Process:
    1. 비디오 로딩 및 출력 설정
    2. 프레임별 처리
       - 전처리
       - 모델 예측
       - 결과 시각화
       - 인코딩 및 저장
    """

    # 비디오 파일 이름 추출
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 출력 비디오 경로 설정
    video_dir = 'Only_Segmentation_video'
    os.makedirs(video_dir, exist_ok=True)
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

    try:
        for frame_count in tqdm(range(total_frames)):
            frame_start_time = time.time()

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
            
            # 2. 이미지 전처리
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
            weld_mask = (pred_weld > 0.5).astype(np.uint8) * 255 # pixel 값이 0.5 이상인 pixel에 대해 용접선 클래스 부여
            unpretreated_mask = (pred_unpretreated > 0.5).astype(np.uint8) * 255 # pixel 값이 0.5 이상인 pixel에 대해 비전처리 표면 클래스 부여
            
            # 원본 이미지 복원 (정규화 역변환)
            original_img = img.squeeze().cpu().numpy()
            original_img = original_img.transpose(1, 2, 0) # (C,H,W) -> (H,W,C)로 변경
            original_img = (original_img * std.squeeze().cpu().numpy() + mean.squeeze().cpu().numpy()) * 255
            original_img = np.clip(original_img, 0, 255).astype(np.uint8)

            # 원본 이미지 복사
            overlay = original_img.copy()

            # 용접선 마스크 오버레이 (파란색)
            weld_overlay = np.zeros_like(overlay) # 원본 이미지와 같은 크기의 빈 배열 생성
            weld_overlay[weld_mask > 0] = [0, 0, 255]  # 용접선 영역을 파란색으로 설정
            overlay = cv2.addWeighted(overlay, 1, weld_overlay, 0.75, 0) # 75% 투명도로 오버레이

            # 비전처리 표면 마스크 오버레이 (빨간색)
            unpretreated_overlay = np.zeros_like(overlay) # 원본 이미지와 같은 크기의 빈 배열 생성
            unpretreated_overlay[unpretreated_mask > 0] = [255, 0, 0]  # 비전처리 표면 영역을 빨간색으로 설정
            overlay = cv2.addWeighted(overlay, 1, unpretreated_overlay, 0.25, 0) # 25% 투명도로 오버레이

            # 교집합 영역 표시 (녹색)
            intersection_mask = ((weld_mask > 0) & (unpretreated_mask > 0)).astype(np.uint8)
            overlay[intersection_mask > 0] = [51, 153, 102]  # 두 마스크가 겹치는 부분을 녹색으로 표시

            # FPS 계산 및 표시
            frame_times.append(time.time() - frame_start_time) # 프레임 처리 시간 기록
            if len(frame_times) > fps_update_interval: # 지정된 간격(30프레임) 초과시
                frame_times = frame_times[-fps_update_interval:] # 최근 기록만 유지
            
            if frame_count % fps_update_interval == 0: # 30 프레임마다
                current_fps = 1 / (sum(frame_times) / len(frame_times)) # 평균 FPS 계산
            else: # 그 외 프레임일 시
                current_fps = 1 / frame_times[-1]  # 현재 프레임의 FPS 계산

            # overlay에 FPS 텍스트 추가 (BGR로 변환하기 직전에 추가)
            cv2.putText(overlay, f"{current_fps:.2f} Hz", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            #  BGR로 변환 (OpenCV 형식)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR) # OpenCV 색상 형식으로 변환

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

'''모델 성능 테스트'''
def test_model(model, test_loader, device):
    """학습된 모델의 성능을 테스트하는 함수
   
    Args:
        model: 학습된 U-Net 모델
        test_loader: 테스트 데이터셋의 데이터로더
        device: 연산 장치 (CPU/GPU)
    
    Returns:
        metrics: 딕셔너리 형태의 성능 지표
            {
                'weld': {         # 용접선 검출 성능
                    'iou':        # Intersection over Union
                    'dice(f1)':   # Dice coefficient (F1 score)
                    'accuracy':   # 정확도
                    'precision':  # 정밀도
                    'recall':     # 재현율
                },
                'unpretreated': {...},  # 비전처리 표면 검출 성능
                'average': {...}        # 두 클래스의 평균 성능
            }
    """
    model.eval() # 모델 평가 모드
    
    # 모델 성능 평가기준
    metrics = {
        'weld': {'iou': 0, 'dice(f1)': 0, 'accuracy': 0, 'precision': 0, 'recall': 0},
        'unpretreated': {'iou': 0, 'dice(f1)': 0, 'accuracy': 0, 'precision': 0, 'recall': 0},
        'average': {'iou': 0, 'dice(f1)': 0, 'accuracy': 0, 'precision': 0, 'recall': 0}
    }
    num_batches = 0
    
    with torch.no_grad(): # test이므로 gradient 계산 X
        for img, weld_mask, unpretreated_mask, _ in tqdm(test_loader, desc="Testing"): # 테스트 데이터 로더에서 배치 단위로 데이터 로드, 이미지 경로는 사용 x
            img = img.float().to(device) # 이미지를 디바이스로 이동
            weld_mask = weld_mask.float().to(device) # 용접선 마스크를 디바이스로 이동
            unpretreated_mask = unpretreated_mask.float().to(device) # 비전처리 표면 마스크를 디바이스로 이동
            
            with autocast(): # FP32(단정밀도)와 FP16(반정밀도)를 혼합 사용하여 메모리 사용량 감소 & 연산 속도 향상
                weld_output, unpretreated_output = model(img) # 모델 예측 수행 (순전파)
                
            # 예측 결과를 이진 마스크로 변환
            weld_pred = torch.sigmoid(weld_output).squeeze(1) > 0.5 # 배치 차원 제거 후 pixel값이 0.5 이상인 pixel에 대해 용접선 클래스 부여
            unpretreated_pred = torch.sigmoid(unpretreated_output).squeeze(1) > 0.5 # 배치 차원 제거 후 pixel값이 0.5 이상인 pixel에 대해 비전처리 표면 클래스 부여
            
            # 예측 결과를 NumPy 배열로 변환
            weld_pred_np = weld_pred.cpu().numpy().squeeze()
            unpretreated_pred_np = unpretreated_pred.cpu().numpy().squeeze()

            # 다시 텐서로 변환하여 GPU/CPU로 이동
            weld_pred = torch.from_numpy(weld_pred_np).float().to(device)
            unpretreated_pred = torch.from_numpy(unpretreated_pred_np).float().to(device)
            
            # 용접선에 대해 성능 평가
            metrics['weld']['iou'] += calc_iou(weld_pred, weld_mask).item()
            metrics['weld']['dice(f1)'] += calc_dice(weld_pred, weld_mask).item()
            metrics['weld']['accuracy'] += calc_accuracy(weld_pred, weld_mask).item()
            precision, recall = calc_precision_recall(weld_pred, weld_mask)
            metrics['weld']['precision'] += precision.item()
            metrics['weld']['recall'] += recall.item()
            
            # 비전처리 표면에 대해 성능 평가
            metrics['unpretreated']['iou'] += calc_iou(unpretreated_pred, unpretreated_mask).item()
            metrics['unpretreated']['dice(f1)'] += calc_dice(unpretreated_pred, unpretreated_mask).item()
            metrics['unpretreated']['accuracy'] += calc_accuracy(unpretreated_pred, unpretreated_mask).item()
            precision, recall = calc_precision_recall(unpretreated_pred, unpretreated_mask)
            metrics['unpretreated']['precision'] += precision.item()
            metrics['unpretreated']['recall'] += recall.item()
            
            num_batches += 1
    
    # 두 클래스의 평균 성능 계산
    for metric in metrics['weld'].keys(): # 각 분기(branch)에 대해 반복
        metrics['average'][metric] = (metrics['weld'][metric] + metrics['unpretreated'][metric]) / 2
    
    for branch in metrics: # 각 분기(branch)에 대해 반복
        for metric in metrics[branch]: # 각 메트릭에 대해 반복
            metrics[branch][metric] /= num_batches # 배치별 평균 메트릭 계산
    
    return metrics

def visualize_predictions(model, test_loader, device, save_dir='visualization_results'):
    """
    모델의 예측 결과를 시각화하는 함수
    
    Args:
        model: 학습된 Segmentation 모델
        test_loader: 테스트 데이터 로더
        device: 연산 장치 (CPU/GPU)
        save_dir: 시각화 결과를 저장할 디렉토리
    
    Returns:
        total_metrics: 딕셔너리 형태의 성능 지표
            {
                'Weld': {
                    'IoU':       # 용접선 IoU 평균
                    'Dice(F1)':  # 용접선 F1 score 평균
                    'Accuracy':  # 용접선 정확도 평균
                    'Precision': # 용접선 정밀도 평균
                    'Recall':    # 용접선 재현율 평균
                },
                'Unpretreated': {      # 비전처리 표면에 대한 동일한 지표
                    ...
                },
                'Average': {           # 두 클래스의 평균 지표
                    ...
                }
            }
    """
    model.eval() # 모델 평가 모드
    os.makedirs(save_dir, exist_ok=True) # 저장 디렉토리 생성

    # 전체 성능 지표 저장용 딕셔너리
    total_metrics = {'Weld': {}, 'Unpretreated': {}, 'Average': {}}
    
    # 배치별 처리
    for i, (original_img, weld_mask, unpretreated_mask, img_path) in enumerate(tqdm(test_loader, desc="Visualizing")):
        batch_size = original_img.size(0) # 배치 크기 확인
        random_index = random.randint(0, batch_size - 1) # 각 배치별 무작위로 시각화할 이미지 1개 선정

        original_img = original_img[random_index].float().to(device) # 이미지 디바이스로 이동
        weld_mask = weld_mask[random_index].float().to(device) # 용접선 마스크 디바이스로 불러오기
        unpretreated_mask = unpretreated_mask[random_index].float().to(device) # 비전처리 표면 마스크 gpu 디바이스로 불러오기
        img_path = img_path[random_index]

        # 모델 예측 
        with torch.no_grad(): # test이므로 gradient 계산 X
            weld_output, unpretreated_output = model(original_img.unsqueeze(0))

        # 예측 결과를 이진 마스크로 변환
        weld_pred = (torch.sigmoid(weld_output).squeeze() > 0.5).float() # pixel값이 0.5 이상인 pixel에 대해 용접선 클래스 부여
        unpretreated_pred = (torch.sigmoid(unpretreated_output).squeeze() > 0.5).float() # pixel값이 0.5 이상인 pixel에 대해 비전처리 표면 클래스 부여

        original_img_np = original_img.cpu().numpy().transpose(1, 2, 0) # (C,H,W) -> (H,W,C)로 변경
        
        mean = np.array([0.485, 0.456, 0.406]) # ImageNet RGB 채널별 평균값
        std = np.array([0.229, 0.224, 0.225]) # ImageNet RGB 채널별 표준편차
        original_img_np = std * original_img_np + mean # 역정규화 -> 원본 이미지 복원
        
        original_img_np = np.clip(original_img_np, 0, 1) # 값 범위 제한
        original_img_np = (original_img_np * 255).astype(np.uint8) # 0-255 범위로 변환
        
        # GPU 텐서를 CPU NumPy 배열로 변환
        weld_mask_np = weld_mask.cpu().numpy().squeeze()
        unpretreated_mask_np = unpretreated_mask.cpu().numpy().squeeze()
        weld_pred_np = weld_pred.cpu().numpy().squeeze()
        unpretreated_pred_np = unpretreated_pred.cpu().numpy().squeeze()

        # 예측 마스크를 0-255 범위로 변환
        weld_pred_np = (weld_pred_np * 255).astype(np.uint8)
        unpretreated_pred_np = (unpretreated_pred_np * 255).astype(np.uint8)
            
        # 교집합 영역 계산
        intersection_mask = weld_mask_np * unpretreated_mask_np # 실제 마스크 교집합
        intersection_pred = weld_pred_np * unpretreated_pred_np # 예측 마스크 교집합

        fig, axs = plt.subplots(3, 3, figsize=(15, 15))

        # 원본 이미지 
        axs[0, 0].imshow(original_img_np)
        axs[0, 0].set_title('Original Image')
        axs[0, 0].axis('off')
        
        # ground truth 용접선 마스크 이미지 
        axs[0, 1].imshow(weld_mask_np, cmap='gray')
        axs[0, 1].set_title('Ground Truth Weld Mask')
        axs[0, 1].axis('off')
        

        # 예측한 용접선 마스크 이미지
        axs[0, 2].imshow(weld_pred_np, cmap='gray')
        axs[0, 2].set_title('Predicted Weld Mask')
        axs[0, 2].axis('off')
        
        # 원본 이미지 
        axs[1, 0].imshow(original_img_np)
        axs[1, 0].set_title('Original Image')
        axs[1, 0].axis('off')

        # ground truth 비전처리 표면 마스크 이미지 
        axs[1, 1].imshow(unpretreated_mask_np, cmap='gray')
        axs[1, 1].set_title('Ground Truth Unpretreated Mask')
        axs[1, 1].axis('off')
        
        # 예측한 비전처리 표면 마스크 이미지 
        axs[1, 2].imshow(unpretreated_pred_np, cmap='gray')
        axs[1, 2].set_title('Predicted Unpretreated Mask')
        axs[1, 2].axis('off')
        
        # 원본 이미지 
        axs[2, 0].imshow(original_img_np)
        axs[2, 0].set_title('Original Image')
        axs[2, 0].axis('off')
        
        # ground truth 교집합 영역 마스크 이미지 
        axs[2, 1].imshow(intersection_mask, cmap='gray')
        axs[2, 1].set_title('Intersection of Ground Truth Masks')
        axs[2, 1].axis('off')
        
        # 예측한 교집합 영역 마스크 이미지 
        axs[2, 2].imshow(intersection_pred, cmap='gray')
        axs[2, 2].set_title('Predicted Intersection Mask')
        axs[2, 2].axis('off')
        
        # 각 이미지별 성능 평가
        metrics = {'Weld': {}, 'Unpretreated': {}, 'Average': {}}
        for mask_type in ['Weld', 'Unpretreated']:
            # 해당 클래스의 마스크와 예측값 선택
            if mask_type == 'Weld':
                mask = weld_mask
                pred = weld_pred
            else:
                mask = unpretreated_mask
                pred = unpretreated_pred

            # 성능 지표 계산
            metrics[mask_type] = {
                'IoU': calc_iou(pred, mask).item(),
                'Dice(F1)': calc_dice(pred, mask).item(),
                'Accuracy': calc_accuracy(pred, mask).item()
            }
            precision, recall = calc_precision_recall(pred.unsqueeze(0), mask.unsqueeze(0))
            metrics[mask_type]['Precision'] = precision.item()
            metrics[mask_type]['Recall'] = recall.item()

        # 평균 성능 계산
        for metric in metrics['Weld'].keys():
            metrics['Average'][metric] = (metrics['Weld'][metric] + metrics['Unpretreated'][metric]) / 2

        for branch in metrics: # 각 분기(branch)에 대해 반복
            for metric in metrics[branch]: # 각 메트릭에 대해 반복
                if metric not in total_metrics[branch]: # total_metrics에 해당 메트릭이 없으면 초기화
                    total_metrics[branch][metric] = 0
                total_metrics[branch][metric] += metrics[branch][metric] # 현재 배치의 성능을 전체 성능에 누적

        # 성능 시각화
        plt.figtext(0.5, 0.01, f"Image: {os.path.basename(img_path)}", ha="center", fontsize=14)
        plt.figtext(0.5, 0.04, 
                    f"Weld - IoU: {metrics['Weld']['IoU']:.4f}, Dice(F1): {metrics['Weld']['Dice(F1)']:.4f}, "
                    f"Acc: {metrics['Weld']['Accuracy']:.4f}, Prec: {metrics['Weld']['Precision']:.4f}, "
                    f"Recall: {metrics['Weld']['Recall']:.4f}", 
                    ha="center", fontsize=14)
        plt.figtext(0.5, 0.07, 
                    f"Unpretreated - IoU: {metrics['Unpretreated']['IoU']:.4f}, Dice(F1): {metrics['Unpretreated']['Dice(F1)']:.4f}, "
                    f"Acc: {metrics['Unpretreated']['Accuracy']:.4f}, Prec: {metrics['Unpretreated']['Precision']:.4f}, "
                    f"Recall: {metrics['Unpretreated']['Recall']:.4f}",
                    ha="center", fontsize=14)
        plt.figtext(0.5, 0.10, 
                    f"Average - IoU: {metrics['Average']['IoU']:.4f}, Dice(F1): {metrics['Average']['Dice(F1)']:.4f}, "
                    f"Acc: {metrics['Average']['Accuracy']:.4f}, Prec: {metrics['Average']['Precision']:.4f}, "
                    f"Recall: {metrics['Average']['Recall']:.4f}", 
                    ha="center", fontsize=14)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.20)

        # 결과 이미지 저장
        original_filename = os.path.basename(img_path)
        save_filename = f"{original_filename}"
        save_path = os.path.join(save_dir, save_filename)

        plt.savefig(save_path)
        plt.close(fig)

        if args.use_wandb:
            wandb.log({f"visualization_{i+1}": wandb.Image(save_path)})

    # 파일 저장
    print(f"All visualizations saved in {save_dir}")

    num_samples = len(test_loader)

    for branch in total_metrics: # 각 분기(branch)에 대해 반복
        for metric in total_metrics[branch]: # 각 메트릭에 대해 반복
            total_metrics[branch][metric] /= num_samples # 전체 평균 성능 계산

    return total_metrics

# main 함수를 통해 실행
if __name__ == "__main__":
    # 명령줄 인자 파서 생성
    parser = argparse.ArgumentParser(description='Inference for weldline detection')
    # 이미지 관련 인자
    parser.add_argument('--image_dir', type=str, default='segmentation/data/imgs', help='Directory containing test images') # 테스트 이미지 디렉토리 경로
    parser.add_argument('--weld_mask_dir', type=str, default='segmentation/data/weldline_masks', help='Directory containing weld masks') # 용접선 마스크 디렉토리 경로
    parser.add_argument('--unpretreated_mask_dir', type=str, default='segmentation/data/unpretreated_masks', help='Directory containing unpretreated masks') # 비전처리 표면 마스크 디렉토리 경로
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference') # 추론 시 배치 크기

    # 저장 디렉토리 관련 인자
    parser.add_argument('--save_dir', type=str, default='segmentation_inference_results', help='Directory to save results') # 결과 저장 디렉토리 경로

    # 비디오 관련 인자
    parser.add_argument('--input_video', type=str, default='in_video/T_shape.mp4', help='Path to the input video file') # 입력 비디오 파일 경로

    # 모델 관련 인자
    parser.add_argument('--model_path', type=str, default='segmentation/models/best_model_accuracy_averaged.pt', help='Path to the saved averaged model') # 학습된 모델 가중치 파일 경로

    parser.add_argument('--use_wandb', action='store_true', help='Use wandb for logging') # wandb 사용 여부 (플래그)
    args = parser.parse_args() # 인자 파싱

    # 예측 결과 저장 디렉토리 설정
    prediction_save_dir = os.path.join(args.save_dir, 'visualizations')
    os.makedirs(args.save_dir, exist_ok=True) # 저장 디렉토리 생성

    # GPU 사용 가능 여부 확인 및 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # wandb 초기화 (사용 설정된 경우)
    if args.use_wandb:
        wandb.init(project="weldline_inference", config=vars(args))

    try:
        # # 데이터셋 로드 및 분할
        # dataset = WeldlineDataset(image_dirs=args.image_dir, weld_mask_dirs=args.weld_mask_dir, unpretreated_mask_dirs=args.unpretreated_mask_dir)
        # total_size = len(dataset)
        # train_size = int(0.8 * total_size) # 학습 데이터 80%
        # val_size = int(0.1 * total_size) # 검증 데이터 10%
        # test_size = total_size - train_size - val_size # 테스트 데이터 10%

        # _, _, test_dataset = random_split(dataset, [train_size, val_size, test_size]) # 테스트 데이터만 사용

        # # 테스트 데이터 로더 생성
        # test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
        # print(f"Loaded {len(test_dataset)} test samples")

        # 모델 로드
        model = UNet(in_channels=3, num_classes=2) # 3채널 입력, 2채널 출력
        state_dict = torch.load(args.model_path, map_location=device) # 저장된 모델 가중치 로드
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict) # 모델에 가중치 적용
        model = model.to(device)

        # DataParallel 설정(병렬 처리)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model, device_ids=[0, 1]) # 사용할 GPU 번호 설정 (GPU 0, 1 사용)
            torch.cuda.set_device(0)  # 메인 GPU 설정
            
        print(f"Model loaded from {args.model_path}")

        # # 모델 성능 테스트 함수 호출
        # metrics = test_model(model, test_loader, device)

        # # 테스트 결과 출력 및 저장
        # print("Test Results:")
        # results_str = ""
        # for branch in ['weld', 'unpretreated', 'average']:
        #     print(f"{branch.capitalize()}:")
        #     results_str += f"{branch.capitalize()}:\n"
        #     for metric, value in metrics[branch].items():
        #         print(f"  {metric.capitalize()}: {value:.4f}")
        #         results_str += f"  {metric.capitalize()}: {value:.4f}\n"
        #     print()
        #     results_str += "\n"

        # # 모델 성능 저장 디렉토리 내에 txt 파일에 기록
        # with open(os.path.join(args.save_dir, 'test_results.txt'), 'w') as f:
        #     f.write(results_str)

        # prediction_save_dir = os.path.join(args.save_dir, 'visualizations')
        # visualize_predictions(model, test_loader, device, save_dir=prediction_save_dir) # 모델 예측 결과 시각화 함수 호출

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

    if args.use_wandb:
        wandb.finish() # wandb 세션 종료