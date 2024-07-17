<<<<<<< HEAD
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
=======
from datasets import Dataset
from transformers import AutoTokenizer

# 예제 텍스트와 레이블
train_text = ["Hello, how are you?", "I am fine, thank you."]
train_label = [1, 0]

# 토크나이저 로드 및 텍스트 인코딩
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(train_text, truncation=True, padding=True)

# Dataset 객체 생성
train_dataset = Dataset.from_dict(
    {
        'text': train_text,
        'labels': train_label,
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask']
    }
)

# Dataset 객체 출력
print(train_encodings)
>>>>>>> d01a7ba370fbe14dce16311d794988f2b16a9f6a
