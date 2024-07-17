import os
import pandas as pd
import torch
import numpy as np
import logging
import time
import shutil
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from transformers import TrainerCallback, TrainerState, TrainerControl
from datasets import Dataset, load_metric
from torch.utils.data import DataLoader

# 로깅 설정
logging.basicConfig(filename='batch_log.txt', level=logging.INFO)

# 데이터 전처리
def train_prepare_data(file_path, sample_frac=0.05):
    df = pd.read_csv(file_path)
    
    # 데이터 샘플링
    df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    # 레이블 인코딩
    label_encoder = LabelEncoder()
    df['Sentiment'] = label_encoder.fit_transform(df['Sentiment'])

    # 학습 / 테스트 데이터 분할
    train_text, test_text, train_label, test_label = train_test_split(
        df['Text'], df['Sentiment'], test_size=0.2, random_state=42
    )

    return train_text, test_text, train_label, test_label, label_encoder

# 데이터셋 생성
def create_dataset(train_text, test_text, train_label, test_label, tokenizer):
    train_encodings = tokenizer(train_text.tolist(), truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_text.tolist(), truncation=True, padding=True, max_length=512)

    train_dataset = Dataset.from_dict(
        {
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': train_label.tolist()
        }
    )

    test_dataset = Dataset.from_dict(
        {
            'input_ids': test_encodings['input_ids'],
            'attention_mask': test_encodings['attention_mask'],
            'labels': test_label.tolist()
        }
    )

    return train_dataset, test_dataset

# DataLoader 생성
def create_dataloader(dataset, batch_size, shuffle=False, collate_fn=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, collate_fn=collate_fn, num_workers=6)


class SaveEvalLossCallback(TrainerCallback):
    # 로그가 기록될 때 실행되는 콜백 함수
    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if 'eval_loss' in state.log_history[-1]:
            eval_loss = state.log_history[-1]['eval_loss']
            step = state.global_step
            
            checkpoint_dir = f"{args.output_dir}/checkpoint-{step}"
            
            with open('checkpoint_log.txt', 'a') as f:
                f.write(f"{checkpoint_dir},{eval_loss}\n")
            print(f"Checkpoint saved at {checkpoint_dir} with eval_loss {eval_loss}")

            best_checkpoint = get_best_checkpoint()
            if best_checkpoint:
                print(f"New best checkpoint found: {best_checkpoint}")
                # 최적의 체크포인트를 best_checkpoint 폴더로 복사
                best_checkpoint_dir = './best_checkpoint'
                try:
                    if os.path.exists(best_checkpoint_dir):
                        shutil.rmtree(best_checkpoint_dir)
                    # 체크포인트 디렉토리가 생성될 때까지 기다림
                    wait_for_directory(checkpoint_dir)
                    shutil.copytree(best_checkpoint, best_checkpoint_dir)
                except FileNotFoundError:
                    print(f"Checkpoint directory {best_checkpoint} does not exist. Skipping save.")
            else:
                print("No best checkpoint found.")


def wait_for_directory(directory, timeout=30):
    # 지정된 시간 동안 디렉토리가 생성될 때까지 기다림
    start_time = time.time()
    while not os.path.exists(directory):
        if time.time() - start_time > timeout:
            raise FileNotFoundError(f"Directory {directory} not found within {timeout} seconds.")
        time.sleep(1)


def get_best_checkpoint():
    best_checkpoint = None
    best_loss = float('inf')

    if not os.path.exists('checkpoint_log.txt'):
        return best_checkpoint

    with open('checkpoint_log.txt', 'r') as f:
        for line in f:
            checkpoint_dir, eval_loss = line.strip().split(',')
            eval_loss = float(eval_loss)
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_checkpoint = checkpoint_dir

    return best_checkpoint


# 모델 훈련
def train_model(train_dataset, test_dataset, model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=500,
        per_device_eval_batch_size=500,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=100,
        save_safetensors=False  # PyTorch 형식으로 저장
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[SaveEvalLossCallback()],
    )

    trainer.train()

    return trainer


# 모델 평가
def evaluate_model(trainer, test_dataloader, label_encoder):
    # 평가 지표 로드
    metric = load_metric("accuracy", trust_remote_code=True)
    device = trainer.model.device

    all_preds = []
    all_labels = []

    eval_start_time = time.time()

    for batch in test_dataloader:
        logging.info(f"Batch size: {len(batch['input_ids'])}")  # 배치 크기 로그 기록

        # 각 배치 항목을 GPU로 이동
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)

        with torch.no_grad():
            # 모델 예측
            outputs = trainer.model(**inputs)
            preds = outputs.logits.argmax(dim=-1)

        # 예측값과 실제값 저장
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    eval_end_time = time.time()  # 평가 종료 시간 기록
    eval_runtime = eval_end_time - eval_start_time  # 평가 시간 계산
    logging.info(f"Evaluation runtime: {eval_runtime}")  # 평가 시간 로그 기록

    # GPU 메모리 해제
    del inputs
    del labels
    del outputs
    torch.cuda.empty_cache()

    # 정확도 계산
    accuracy = metric.compute(predictions=all_preds, references=all_labels)
    return accuracy

# collate_fn 함수 정의
def collate_fn(batch):
    # 각 요소가 텐서인지 확인하고 텐서가 아니면 텐서로 변환
    def to_tensor(items):
        if isinstance(items[0], torch.Tensor):
            return torch.stack(items)
        else:
            return torch.tensor(items, dtype=torch.long)

    return {key: to_tensor([item[key] for item in batch]) for key in batch[0]}

def cross_validate(df, tokenizer, model_name, label_encoder, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []

    for i, (train_index, test_index) in enumerate(skf.split(df['Text'], df['Sentiment'])):
        train_text, test_text = df['Text'].iloc[train_index], df['Text'].iloc[test_index]
        train_label, test_label = df['Sentiment'].iloc[train_index], df['Sentiment'].iloc[test_index]

        train_dataset, test_dataset = create_dataset(train_text, test_text, train_label, test_label, tokenizer)
        train_dataloader = create_dataloader(train_dataset, batch_size=500, shuffle=True, collate_fn=collate_fn)
        test_dataloader = create_dataloader(test_dataset, batch_size=500, collate_fn=collate_fn)
        
        trainer = train_model(train_dataset, test_dataset, model_name)
        accuracy = evaluate_model(trainer, test_dataloader, label_encoder)
        accuracies.append(accuracy['accuracy'])

        # 두 번의 반복 후 루프 종료
        if i == 1:
            break

    return accuracies

def final_evaluation(train_text, train_label, test_text, test_label, tokenizer, checkpoint_path, label_encoder):
    train_dataset, test_dataset = create_dataset(train_text, test_text, train_label, test_label, tokenizer)
    train_dataloader = create_dataloader(train_dataset, batch_size=500, shuffle=True, collate_fn=collate_fn)
    test_dataloader = create_dataloader(test_dataset, batch_size=500, collate_fn=collate_fn)

    # 경로를 절대 경로로 변환하고 백슬래시를 슬래시로 교체
    checkpoint_path = os.path.abspath(checkpoint_path).replace('\\', '/')
    print(f"Using checkpoint path: {checkpoint_path}")  # 경로 확인을 위한 로깅 추가

    # 기존 모델 객체가 있다면 삭제
    if 'model' in locals():
        del model
        torch.cuda.empty_cache()
    
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=3)
    
    # 모델을 GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=500,
        per_device_eval_batch_size=500,
        num_train_epochs=1,
        weight_decay=0.01
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    accuracy = evaluate_model(trainer, test_dataloader, label_encoder)
    return accuracy
