import torch
import numpy as np
import math
from typing import List, Dict


def safe_divide(numerator, denominator, eps=1e-8):
    """
    안전한 나눗셈 연산을 수행하는 유틸리티 함수
    
    Args:
        numerator: 분자
        denominator: 분모
        eps: 분모가 0이 되는 것을 방지하기 위한 작은 값 (기본값: 1e-8)
    
    Returns:
        float: 나눗셈 결과
    """
    return numerator / (denominator + eps)

def calc_dice(pred, mask, threshold=0.5):
    """
    Dice coefficient(F1 score) 계산
    Dice = 2|X∩Y|/(|X|+|Y|)
    
    Args:
        pred: 모델의 예측값 (0~1 사이의 확률)
        mask: Ground truth 마스크
        threshold: 이진화를 위한 임계값 (기본값: 0.5)
    
    Returns:
        Tensor: Dice coefficient 값 (0~1 사이)
    """
    pred_binary = (pred > threshold).float().view(-1)
    mask_binary = (mask > threshold).float().view(-1)
    
    intersection = (pred_binary * mask_binary).sum()
    return safe_divide(2 * intersection, pred_binary.sum() + mask_binary.sum())

def get_line_angle(line: np.ndarray) -> float:
    """
    직선의 각도를 계산 (0~180도 범위)
    
    Args:
        line: [x1, y1, x2, y2] 형태의 직선 좌표
    
    Returns:
        float: 0~180도 범위의 각도
    """
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1 # 이미지 좌표계에서는 y축이 아래로 증가
    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 180 # 음수 각도를 양수로 변환
    return angle

def classify_line(angle: float) -> str:
    """
    각도를 기반으로 직선의 방향 분류
    
    Args:
        angle: 0~180도 범위의 각도
    
    Returns:
        str: 'vertical' 또는 'horizontal'
    """
    if 45 <= angle <= 135:
        return 'vertical'
    else:
        return 'horizontal'

def get_lines_from_mask(mask: np.ndarray) -> List[Dict]:
    """
    마스크에서 RANSAC을 사용하여 직선 검출

    알고리즘 프로세스:
    1. 마스크 전처리
       - 이진화
       - 픽셀 좌표 추출
    
    2. RANSAC 반복
       - 랜덤 2점 선택
       - 직선 모델 피팅
       - 인라이어 검출
       - 최적 모델 갱신
    
    3. 직선 정보 추출
       - 시작/끝점 결정
       - 길이 계산
       - 각도 계산
       - 방향 분류
    
    Args:
        mask: 이진 마스크 이미지
    
    Returns:
        List[Dict]: 검출된 직선들의 정보
        각 직선 정보는 다음을 포함:
        - coords: [x1, y1, x2, y2] 좌표
        - length: 직선 길이
        - angle: 각도 (0~180도)
        - direction: 방향 ('vertical' 또는 'horizontal')
    """
    # 입력 데이터 전처리
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy() # GPU 텐서인 경우 CPU NumPy 배열로 변환
    
    if mask.ndim > 2: # 마스크가 3차원 이상일 때
        mask = mask.squeeze() # 배치/채널 차원 제거하여 2D 이미지로 변환
    
    # 마스크 전처리
    mask = (mask > 0.5).astype(np.uint8) * 255 # 마스크를 0과 1로 이진화한 후, 시각화를 위해 0과 255로 스케일링
    
    # 마스크에서 흰색 픽셀(스켈레톤)의 모든 좌표 추출
    y_coords, x_coords = np.where(mask > 0) # mask > 0인 위치의 y, x 좌표 반환
    if len(x_coords) < 2: # 2개 점 미만일 때
        return [] # 빈 리스트 반환
    
    points = np.column_stack((x_coords, y_coords)) # x, y 좌표로 하나의 N×2 배열 생성 -> 각 행은 하나의 점 좌표 (x, y)
    
    def distance_to_line(points, params): # RANSAC 알고리즘에서 inlier를 찾기 위해 사용
        """
        점들과 직선 사이의 거리 계산
        
        Args:
            points: 점들의 좌표
            params: 직선 파라미터
        """
        if params['is_vertical']: # 수직선인 경우
            return np.abs(points[:, 0] - params['x']) # x 좌표 차이
        else: # 일반 직선의 경우
            slope, intercept = params['slope'], params['intercept'] 
            return np.abs(points[:, 1] - (slope * points[:, 0] + intercept)) / np.sqrt(1 + slope**2) # 점과 직선 사이의 수직 거리 계산

    def find_mask_boundary_points(line_points, binary_mask, margin=2):
        """RANSAC으로 찾은 직선을 마스크 영역 내 최대/최소 좌표까지 확장"""
        height, width = binary_mask.shape
        x1, y1 = line_points[0] # 시작점
        x2, y2 = line_points[1] # 끝점
        
        # 직선의 기울기와 y절편 계산
        if abs(x2 - x1) < abs(y2 - y1):  # 수직선인 경우 (y 변화가 x 변화보다 큰 경우)
            # 마스크에서 1인 픽셀들의 y좌표 찾기
            y_coords = np.where(binary_mask > 0)[0]
            if len(y_coords) == 0:
                return None
             
            y_min, y_max = np.min(y_coords), np.max(y_coords) # y좌표의 최소/최대값
            
            # 직선의 기울기와 절편 계산
            m = (x2 - x1) / (y2 - y1) if y2 != y1 else float('inf')
            b = x1 - m * y1 if y2 != y1 else x1
            
            # 새로운 x좌표 계산
            if m != float('inf'):
                x_min = int(round(m * y_min + b))
                x_max = int(round(m * y_max + b))
            else:
                x_min = x_max = int(round(x1))
            
            # 이미지 범위 내로 제한
            x_min = max(0, min(width-1, x_min))
            x_max = max(0, min(width-1, x_max))
            
            # 새로운 끝점 반환
            return np.array([[x_min, y_min], [x_max, y_max]])
            
        else:  # horizontal line  (x 변화가 y 변화보다 큰 경우)
            # 마스크에서 1인 픽셀들의 x좌표 찾기
            x_coords = np.where(binary_mask > 0)[1]
            if len(x_coords) == 0:
                return None
            
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            
            # 직선의 기울기와 절편 계산
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            
            # 새로운 y좌표 계산
            y_min = int(round(m * x_min + b))
            y_max = int(round(m * x_max + b))
            
            # 이미지 범위 내로 제한
            y_min = max(0, min(height-1, y_min))
            y_max = max(0, min(height-1, y_max))
            
            # 새로운 끝점 반환
            return np.array([[x_min, y_min], [x_max, y_max]])
        
    def fit_line_ransac(points, iterations=300, threshold=2.0, min_inliers_ratio=0.1):
        """
        RANSAC 알고리즘을 사용한 직선 피팅
        
        Args:
            points: 직선을 피팅할 포인트들의 좌표 (N x 2 배열, 각 행은 [x, y] 좌표)
            iterations: RANSAC 반복 횟수 (기본값: 300)
            threshold: 점과 직선 사이의 최대 허용 거리 (단위: 픽셀, 기본값: 2.0)
            min_inliers_ratio: 전체 점들 중 직선에 속하는 점(inlier)의 최소 비율 (기본값: 0.1)
        
        Returns:
            start_point: 검출된 직선의 시작점 [x, y]
            end_point: 검출된 직선의 끝점 [x, y]
            best_angle: 검출된 직선의 각도 (0~180도)
            best_inliers: 직선에 속하는 점들의 인덱스
        """
        if len(points) < 2: # 점이 2개 미만이면 직선 검출 불가능
            return None, None, None, None
            
        best_inliers = [] # 가장 좋은 직선의 inlier 점들
        best_params = None # 가장 좋은 직선의 파라미터
        best_angle = None # 가장 좋은 직선의 각도
        n_points = len(points) # 전체 점의 개수
        min_inliers = max(int(n_points * min_inliers_ratio), 5) # 최소 필요 inlier 개수=5
        
        for _ in range(iterations):
            # 랜덤하게 2개의 점 선택
            idx = np.random.choice(n_points, 2, replace=False) # 중복 선택 방지
            p1, p2 = points[idx] # 선택된 두 점의 좌표
            
            x1, y1 = p1 # 첫 번째 점의 x, y 좌표
            x2, y2 = p2 # 두 번째 점의 x, y 좌표
            
            # 직선 파라미터 계산
            if abs(x2 - x1) < 1e-10:  # 두 점의 x 좌표 차이가 매우 작으면 수직선으로 처리
                params = {'is_vertical': True, 'x': x1} # 수직선은 x 좌표만 저장
                current_angle = 90 # 수직선의 각도는 90도
            else: # 일반 직선은 기울기와 절편 계산
                slope = (y2 - y1) / (x2 - x1)  # 기울기 계산
                intercept = y1 - slope * x1 # y절편 계산
                params = {'is_vertical': False, 'slope': slope, 'intercept': intercept}
                current_angle = get_line_angle([x1, y1, x2, y2]) # 직선의 각도 계산
            
            # 모든 점에 대해 현재 모델과의 거리 계산
            distances = distance_to_line(points, params) # 모든 점과의 거리 계산
            inliers = np.where(distances < threshold)[0] # threshold보다 가까운 점들을 inlier로 선택
            
            # 현재 모델이 이전 최적 모델보다 더 많은 inlier를 가지면 업데이트
            if len(inliers) >= min_inliers and len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_params = params
                best_angle = current_angle
        
        # 적절한 모델을 찾지 못한 경우
        if best_params is None:
            return None, None, None, None
        
        # 최종 직선의 시작점과 끝점 결정
        inlier_points = points[best_inliers] # 최적 모델의 inlier 점들
        if best_params['is_vertical']: # 수직선의 경우
            sorted_points = inlier_points[np.argsort(inlier_points[:, 1])] #  y좌표로 정렬하여 끝점 결정
        else: # 일반 직선의 경우
            sorted_points = inlier_points[np.argsort(inlier_points[:, 0])] # x좌표로 정렬하여 끝점 결정
            
        start_point = sorted_points[0] # 정렬된 점들 중 첫 번째 점을 시작점으로
        end_point = sorted_points[-1] # 정렬된 점들 중 마지막 점을 끝점으로

        return start_point, end_point, best_angle, best_inliers
    
    # RANSAC으로 여러 선분 검출 시도
    detected_lines = [] # 검출된 직선들을 저장할 리스트
    remaining_points = points.copy() # 남은 점들 (이미 사용된 점들은 제거)
    
    # 최대 2개의 선분 검출 시도 (수직선과 수평선)
    for _ in range(2):
        if len(remaining_points) < 5:  # 최소 필요 점 개수보다 적으면 종료
            break

        # RANSAC으로 직선 피팅
        start, end, angle, inliers = fit_line_ransac(remaining_points)
        if start is None: # 시작점이 없으면 종료
            break 
        
        # RANSAC으로 찾은 직선의 마스크 내 양 끝점 찾기 (직선을 끝까지 늘려주기 위함)
        line_points = np.array([[start[0], start[1]], [end[0], end[1]]])
        boundary_points = find_mask_boundary_points(line_points, mask)
        
        if boundary_points is not None:
            start = boundary_points[0]
            end = boundary_points[1]

        # 검출된 직선의 정보를 딕셔너리로 저장
        line = {
            'coords': [start[0], start[1], end[0], end[1]], # 시작점과 끝점 좌표
            'length': np.linalg.norm(end - start), # 직선의 길이
            'angle': angle, # 직선의 각도
            'direction': classify_line(angle) # 수직/수평 방향 분류
        }
        
        # 같은 방향의 직선이 이미 검출되었는지 확인
        existing_direction = any(l['direction'] == line['direction'] for l in detected_lines)
        
        # 새로운 방향의 직선이면 추가
        if not existing_direction: 
            detected_lines.append(line)
        
        # inlier가 비어 있지 않으면 사용된 점들 제거
        if inliers is not None:
            remaining_points = np.delete(remaining_points, inliers, axis=0)
    
    # vertical, horizontal 순서로 정렬
    detected_lines.sort(key=lambda x: x['direction'], reverse=True)
    
    return detected_lines # 검출된 직선 반환

def calc_angle_errors(pred: np.ndarray, 
                     gt: np.ndarray) -> Dict[str, float]:
    """예측된 직선과 Ground Truth 간의 방향별 각도 오차 계산

    계산 프로세스:
    1. 예측과 GT에서 직선 검출
    2. 방향별 직선 매칭
    3. 각도 오차 계산

    오차 계산 방식: MAE(Mean Absolute Error)
    - 수평선: |θpred_horizontal - θgt_horizontal|
    - 수직선: |θpred_vertical - θgt_vertical|

    Args:
        pred: 예측된 직선 마스크
        gt: Ground Truth 라벨링 마스크

    Returns:
        Dict: {
            'horizontal_error': float or None,  # 수평 방향 각도 오차
            'vertical_error': float or None,    # 수직 방향 각도 오차
        }
    """
    # 예측과 GT에서 직선 검출
    gt_lines = get_lines_from_mask(gt)
    pred_lines = get_lines_from_mask(pred)
    
    # 방향별 Ground Truth 직선 찾기
    gt_horizontal = next((line for line in gt_lines if line['direction'] == 'horizontal'), None)
    gt_vertical = next((line for line in gt_lines if line['direction'] == 'vertical'), None)
    
    # 방향별 예측 직선 찾기
    pred_horizontal = next((line for line in pred_lines if line['direction'] == 'horizontal'), None)
    pred_vertical = next((line for line in pred_lines if line['direction'] == 'vertical'), None)
    
    # 결과 딕셔너리 초기화
    results = {
        'horizontal_error': None,
        'vertical_error': None,
        # 'horizontal_detected': False,
        # 'vertical_detected': False,
    }
    
    # 수평 방향 오차 계산
    if gt_horizontal and pred_horizontal:
        results['horizontal_error'] = abs(gt_horizontal['angle'] - pred_horizontal['angle'])
        results['horizontal_detected'] = True
    
    # 수직 방향 오차 계산
    if gt_vertical and pred_vertical:
        results['vertical_error'] = abs(gt_vertical['angle'] - pred_vertical['angle'])
        results['vertical_detected'] = True
    
    return results