import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset, load_metric


# 데이터 전처리
def train_prepare_data(file_path):
    # youtube_analyze_sentiment_naver로 전처리한 네이버 데이터 로드
    df = pd.read_csv("text_sentiment.csv")

    # 레이블 인코딩
    label_encoder = LabelEncoder()
    df['Sentiment'] = label_encoder.fit_transform(df['Sentiment'])

    # print(df['Sentiment'].unique())

    # 학습 / 테스트 데이터 분할
    train_text, test_text, train_label, test_label = train_test_split(
        df['Text'], df['Sentiment'], test_size=0.2, random_state=42
    )

    return train_text, test_text, train_label, test_label, label_encoder


# 데이터셋 생성
def create_dataset(train_text, test_text, train_label, test_label, tokenizer):
    train_encoding = tokenizer(train_text.tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(test_text.tolist(), truncation=True, padding=True)

    train_dataset = Dataset.from_dict(
        {
            'text': train_text.tolist(),
            'labels': train_label.tolist(),
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask']
        }
    )

    test_dataset  = Dataset.from_dict(
        {
            'text': test_text.tolist(),
            'labels': test_label.tolist(),
            'input_ids': test_encodings['input_ids'],
            'attention_mask': test_encodings['attention_mask']
        }
    )

    return train_dataset, test_dataset

# 모델 훈련
def train_model(train_dataset, test_dataset, model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 모델을 학습시 사용할 다양한 설정을 정의
    training_args = TrainingArguments(
        output_dir = './',
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=20,
        per_device_eval_batch_size=20,
        num_train_epochs=5,
        weight_decay=0.01,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # 짧은 샘플에 패딩 추가(길이 맞추기)

    # 모델의 학습 및 평가 과정을 관리
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    return trainer


# 모델 평가
def evaluate_model(trainer, test_dataset, label_encoder):
