import os
import pandas as pd
import re
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from model_train import get_best_checkpoint
import torch
from datasets import Dataset, load_metric
from tqdm import tqdm
import time

def analyze_youtube_comments(youtube_data_path, output_path, batch_size=150):
    start_time = time.time()

    # 최적의 체크포인트 경로를 가져옵니다.
    best_checkpoint = get_best_checkpoint()
    print("Best checkpoint:", best_checkpoint)
    
    # 최적의 체크포인트 절대 경로로 변환
    best_checkpoint = os.path.abspath(best_checkpoint).replace('\\', '/')
    print("Absolute path for the best checkpoint:", best_checkpoint)

    # 모델과 토크나이저 로드
    model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(best_checkpoint)

    # GPU 사용 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 유튜브 데이터 로드
    print("Loading YouTube data...")
    youtube_df = pd.read_csv(youtube_data_path)
    print("YouTube data loaded successfully.")

    # 텍스트 전처리 함수
    def preprocess_text(text):
        if text:
            if text.startswith(('http://', 'https://', 'file://')):
                return text
            text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'[^a-zA-Z가-힣\s]', '', text)
        text = text.lower()
        text = ' '.join(text.split())
        return text

    # 댓글을 개별적으로 분리
    def split_comments(text):
        if pd.isna(text):
            return []
        return text.split(' | ')

    # 전처리된 댓글을 위한 함수
    def preprocess_comments(comments):
        preprocessed_comments = []
        for comment in comments:
            preprocessed_comment = preprocess_text(comment)
            if len(preprocessed_comment) > 0:
                preprocessed_comments.append(preprocessed_comment)
        return preprocessed_comments

    # 각 댓글을 전처리하고 전처리된 댓글을 새로운 컬럼에 저장
    print("Preprocessing comments...")
    youtube_df['전처리댓글내용'] = youtube_df['각댓글내용'].apply(lambda x: " | ".join(preprocess_comments(split_comments(x))))
    print("Comments preprocessed successfully.")

    # 감정 분석을 수행할 전체 댓글 리스트 생성
    all_comments = youtube_df['전처리댓글내용'].apply(split_comments).sum()
    print(f"Total comments to analyze: {len(all_comments)}")

    # 댓글이 있는 경우에만 감정 분석 수행
    if len(all_comments) > 0:
        dataset = Dataset.from_dict({"text": all_comments})

        # 감정 분석 함수 정의
        def classify_batch(batch):
            inputs = tokenizer(batch['text'], truncation=True, padding=True, max_length=512, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            labels = predictions.cpu().numpy()
            return {"label": labels}

        # Dataset에 대한 감정 분석 수행
        print("Performing sentiment analysis...")
        results = dataset.map(classify_batch, batched=True, batch_size=batch_size)

        # 분석 결과를 댓글별로 나누기
        all_sentiments = []
        label_map = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}
        for label in results['label']:
            all_sentiments.append(label_map[label])

        # 분석된 결과를 원래 데이터프레임에 매핑
        sentiment_idx = 0
        for i, row in youtube_df.iterrows():
            if row['댓글수'] > 0:  # 댓글이 있는 경우에만 수행
                comments = split_comments(row['전처리댓글내용'])
                sentiments = all_sentiments[sentiment_idx:sentiment_idx + len(comments)]
                youtube_df.at[i, '개별감정평가'] = " | ".join(sentiments)
                sentiment_idx += len(comments)
            else:  # 댓글이 없는 경우 빈 문자열로 설정
                print(f"No comments for row {i}")
                youtube_df.at[i, '개별감정평가'] = ""
                youtube_df.at[i, '긍정수'] = 0
                youtube_df.at[i, '중립수'] = 0
                youtube_df.at[i, '부정수'] = 0
    else:
        # 감정 분석을 수행할 댓글이 없는 경우, 모든 감정 분석 결과를 초기화
        youtube_df['개별감정평가'] = ""
        youtube_df['긍정수'] = 0
        youtube_df['중립수'] = 0
        youtube_df['부정수'] = 0

    # 감정 분포 결과를 데이터프레임에 추가 (댓글이 있는 경우에만 수행)
    youtube_df['긍정수'] = youtube_df['개별감정평가'].apply(lambda x: x.split(' | ').count('POSITIVE') if x else 0)
    youtube_df['중립수'] = youtube_df['개별감정평가'].apply(lambda x: x.split(' | ').count('NEUTRAL') if x else 0)
    youtube_df['부정수'] = youtube_df['개별감정평가'].apply(lambda x: x.split(' | ').count('NEGATIVE') if x else 0)

    # 결과 저장
    output_path = os.path.abspath(output_path).replace('\\', '/')
    print("Saving results to CSV file...")
    youtube_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Data with sentiment saved to {output_path}")

    # 종료 시간 기록 및 총 걸린 시간 계산
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    youtube_data_path = "youtube_data.csv"
    output_path = "youtube_data_sentiment.csv"
    
    analyze_youtube_comments(youtube_data_path, output_path)
