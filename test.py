import torch

print(f'torch.__version__: {torch.__version__}')

print(f'GPU 사용여부: {torch.cuda.is_available()}')
gpu_count = torch.cuda.device_count()
print(f'GPU count: {gpu_count}')
if gpu_count > 0:
    print(f'GPU name: {torch.cuda.get_device_name(0)}')

print("-" * 50)

print("CUDA available: ", torch.cuda.is_available())
print("CUDA device count: ", torch.cuda.device_count())
print("Current CUDA device: ", torch.cuda.current_device())
print("CUDA device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
print("cuDNN version:", torch.backends.cudnn.version())


import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import re
from bs4 import BeautifulSoup
import torch


# 가져온 모델 경로를 할당
checkpoint_dir = "./best_checkpoint"

# 모델과 토크나이저 적용(최적의 체크포인트 경로를 가져옴.)
def load_model_and_tokenizer(checkpoint_dir):
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir, num_labels=3) # 감정 분류 레이블 수에 맞게 조정(긍정, 중립, 부정)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer("./best_checkpoint")
sentiment_pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=512, truncation=True, device=0 if torch.cuda.is_available() else -1)

# 예제 텍스트 분석
texts = ["예제 테스트.", "예제 테스트 입니다."]
results = sentiment_pipeline(texts)
for result in results:
    print(f"Label: {result['label']}, Score: {result['score']}")