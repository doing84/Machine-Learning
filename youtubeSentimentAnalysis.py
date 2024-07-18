import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import re
from bs4 import BeautifulSoup
from model_train import get_best_checkpoint  
import torch
from datasets import Dataset

def analyze_youtube_comments(youtube_data_path, output_path):
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
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=512, truncation=True, device=device)

    # 유튜브 데이터 로드
    youtube_df = pd.read_csv(youtube_data_path)

    # 텍스트 전처리 함수
    def preprocess_text(text):
        if text:
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
    youtube_df['전처리댓글내용'] = youtube_df['각댓글내용'].apply(lambda x: " | ".join(preprocess_comments(split_comments(x))))

    # 모든 댓글을 단일 리스트로 결합
    all_comments = youtube_df['전처리댓글내용'].apply(split_comments).sum()

    # 전처리된 댓글을 Dataset 객체로 변환
    dataset = Dataset.from_dict({"text": all_comments})

    # Dataset에 대한 감정 분석 수행
    results = sentiment_pipeline(dataset['text'])

    # 분석 결과를 댓글별로 나누기
    all_sentiments = []
    for result in results:
        if result['label'] == 'LABEL_0':
            all_sentiments.append('NEGATIVE')
        elif result['label'] == 'LABEL_1':
            all_sentiments.append('NEUTRAL')
        elif result['label'] == 'LABEL_2':
            all_sentiments.append('POSITIVE')

    # 분석된 결과를 원래 데이터프레임에 매핑
    sentiment_idx = 0
    for i, row in youtube_df.iterrows():
        comments = split_comments(row['전처리댓글내용'])
        if comments:  # 댓글이 있는 경우에만 수행
            sentiments = all_sentiments[sentiment_idx:sentiment_idx + len(comments)]
            youtube_df.at[i, '개별감정평가'] = " | ".join(sentiments)
            sentiment_idx += len(comments)
        else:  # 댓글이 없는 경우 빈 문자열로 설정
            youtube_df.at[i, '개별감정평가'] = ""

    # 감정 분포 결과를 데이터프레임에 추가
    youtube_df['긍정수'] = youtube_df['개별감정평가'].apply(lambda x: x.split(' | ').count('POSITIVE') if x else 0)
    youtube_df['중립수'] = youtube_df['개별감정평가'].apply(lambda x: x.split(' | ').count('NEUTRAL') if x else 0)
    youtube_df['부정수'] = youtube_df['개별감정평가'].apply(lambda x: x.split(' | ').count('NEGATIVE') if x else 0)

    # 결과 저장
    youtube_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Data with sentiment saved to {output_path}")

if __name__ == "__main__":
    youtube_data_path = "youtube_data.csv"
    output_path = "youtube_data_sentiment.csv"
    
    analyze_youtube_comments(youtube_data_path, output_path)
