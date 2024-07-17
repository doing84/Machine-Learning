import os
from transformers import AutoModelForSequenceClassification, logging

logging.set_verbosity_error()  # Transformer 로깅 레벨을 오류로 설정

def test_checkpoint_path(checkpoint_path):
    # 경로를 절대 경로로 변환
    checkpoint_path = os.path.abspath(checkpoint_path)
    # 경로의 백슬래시를 슬래시로 교체
    checkpoint_path = checkpoint_path.replace('\\', '/')
    print("Processed checkpoint path:", checkpoint_path)
    
    if os.path.isdir(checkpoint_path):
        try:
            # 모델을 로드하여 경로가 올바른지 테스트
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=3)
            print("Model loaded successfully from checkpoint.")
        except Exception as e:
            print("Error loading model from checkpoint:", e)
    else:
        print("Provided path is not a valid directory.")

# 테스트할 체크포인트 경로 지정
test_checkpoint_path('./results/checkpoint-500')
