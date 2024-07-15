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
