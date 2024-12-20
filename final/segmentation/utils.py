

def safe_divide(numerator, denominator, eps=1e-8):
    """
    분모가 0이 되는 것을 방지하는 안전한 나눗셈 연산
    
    평가 메트릭 계산 시 분모가 0이 되는 경우를 방지하여 
    수치적 안정성 확보
    
    Args:
        numerator: 분자
        denominator: 분모
        eps: 분모에 더할 작은 값 (기본값: 1e-8)
    
    Returns:
        torch.Tensor: 안전하게 계산된 나눗셈 결과
    """
    return numerator / (denominator + eps)

def calc_iou(pred, mask, threshold=0.5):
    """
    Intersection over Union (IoU) 계산

    두 영역의 교집합을 합집합으로 나눈 값으로,
    예측과 실제 마스크 간의 중첩 정도를 평가

    동작 과정:
    1. 예측값과 마스크를 임계값으로 이진화
    2. 교집합 (AND 연산) 계산
    3. 합집합 (OR 연산) 계산
    4. IoU = 교집합 / 합집합
    
    Args:
        pred: 모델의 예측값 (0~1 사이의 확률)
        mask: 실제 마스크 (ground truth)
        threshold: 이진화를 위한 임계값 (기본값: 0.5)
    
    Returns:
        float: IoU 점수 (0~1 사이의 값, 1에 가까울수록 좋음)
    """
    # 예측값과 실제 마스크를 이진화
    pred = (pred > threshold).float().view(-1)
    mask = (mask > threshold).float().view(-1)

    # 교집합과 합집합 계산
    intersection = (pred * mask).sum()
    union = (pred + mask).sum() - intersection

    return safe_divide(intersection, union)

def calc_dice(pred, mask, threshold=0.5):
    """
    Dice coefficient (F1 score) 계산
    
    두 영역의 교집합 크기를 평균 영역 크기로 나눈 값으로,
    예측과 실제 마스크 간의 유사도를 평가
    
    동작 과정:
    1. 예측값과 마스크를 임계값으로 이진화
    2. 교집합 (AND 연산) 계산
    3. Dice = 2 * 교집합 / (예측영역 + 실제영역)
    
    Args:
        pred: 모델의 예측값 (0~1 사이의 확률)
        mask: 실제 마스크 (ground truth)
        threshold: 이진화를 위한 임계값 (기본값: 0.5)
    
    Returns:
        float: Dice 계수 (0~1 사이의 값, 1에 가까울수록 좋음)
    """
    # 예측값과 실제 마스크를 이진화
    pred = (pred > threshold).float().view(-1)
    mask = (mask > threshold).float().view(-1)
    intersection = (pred * mask).sum()

    return safe_divide(2 * intersection, pred.sum() + mask.sum())

def calc_accuracy(pred, mask, threshold=0.5):
    """
    픽셀 단위 정확도 계산
    
    전체 관심 영역(실제 영역과 예측된 영역의 합집합) 중
    올바르게 예측된 픽셀의 비율을 계산
    
    동작 과정:
    1. 예측값과 마스크를 임계값으로 이진화
    2. 실제 관심 영역과 예측된 관심 영역의 합집합 계산
    3. 정확하게 예측된 픽셀 수 계산
    4. Accuracy = 정확한 예측 / 전체 관심 영역
    
    Args:
        pred: 모델의 예측값 (0~1 사이의 확률)
        mask: 실제 마스크 (ground truth)
        threshold: 이진화를 위한 임계값 (기본값: 0.5)
    
    Returns:
        float: 정확도 (0~1 사이의 값, 1에 가까울수록 좋음)
    """
    # 예측값과 실제 마스크를 이진화
    pred = (pred > threshold).float().view(-1)
    mask = (mask > threshold).float().view(-1)
    
    true_region = (mask == 1)  # 실제 관심 영역
    pred_region = (pred == 1)  # 예측된 관심 영역
    
    # 실제 관심 영역과 예측된 관심 영역의 합집합
    total_region = (true_region | pred_region).float()
    total_pixels = total_region.sum()

    # 정확하게 예측된 픽셀 수
    correct_predictions = ((pred == mask) & true_region).float().sum() 
    
    accuracy = safe_divide(correct_predictions, total_pixels)
    return accuracy

def calc_precision_recall(pred, mask, threshold=0.5):
    """
    정밀도(Precision)와 재현율(Recall) 계산

    정밀도: 모델이 양성이라고 예측한 것 중 실제 양성의 비율
    재현율: 실제 양성 중 모델이 양성이라고 예측한 비율
    
    동작 과정:
    1. 예측값과 마스크를 임계값으로 이진화
    2. 진양성(TP), 예측 양성(TP+FP), 실제 양성(TP+FN) 계산
    3. Precision = TP / (TP + FP)
    4. Recall = TP / (TP + FN)

    Args:
        pred: 모델의 예측값 (0~1 사이의 확률)
        mask: 실제 마스크 (ground truth)
        threshold: 이진화를 위한 임계값 (기본값: 0.5)
    
    Returns:
        tuple: (정밀도, 재현율)
        - 정밀도: 예측한 양성 중 실제 양성의 비율 (0~1)
        - 재현율: 실제 양성 중 예측한 양성의 비율 (0~1)
    """
    # 예측값과 실제 마스크를 이진화
    pred = (pred > threshold).float().view(-1)
    mask = (mask > threshold).float().view(-1)
    
    # TP, 예측 양성(TP,FP), 실제 양성(TP,FN) 계산
    true_positives = (pred * mask).sum() # 모델이 양성으로 예측한 것 중 실제 양성인 픽셀 수 (TP)
    predicted_positives = pred.sum() # 모델이 양성으로 예측한 픽셀 수 (TP + FP)
    actual_positives = mask.sum() # 실제 양성인 픽셀 수 (TP + FN)
    
    # Precision, Recall 계산
    precision = safe_divide(true_positives, predicted_positives) # TP / (TP + FP)
    recall = safe_divide(true_positives, actual_positives) # TP / (TP + FN)
    
    return precision, recall