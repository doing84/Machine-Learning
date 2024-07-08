import os
import pandas as pd
import re
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# 환경 변수 설정
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 한국어 감정 분석 모델 사용
model_name = "smilegate-ai/kor_unsmile"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=512, truncation=True)

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

# 각 댓글에 대해 감정 분석 수행
def analyze_sentiment(comments):
    sentiments = []
    for comment in comments:
        preprocessed_comment = preprocess_text(comment)
        if len(preprocessed_comment) > 0:
            result = sentiment_pipeline(preprocessed_comment)[0]['label']
            if result == 'LABEL_0':
                sentiments.append('NEGATIVE')
            elif result == 'LABEL_1':
                sentiments.append('NEUTRAL')
            elif result == 'LABEL_2':
                sentiments.append('POSITIVE')
    return sentiments

# 감정 분석 수행
def analyze_sentiments(df):
    df['전처리댓글내용'] = df['각댓글내용'].apply(lambda x: " | ".join([preprocess_text(comment) for comment in split_comments(x)]))
    df['개별감정평가'] = df['전처리댓글내용'].apply(lambda x: " | ".join(analyze_sentiment(split_comments(x))))
    df['긍정수'] = df['개별감정평가'].apply(lambda x: x.split(' | ').count('POSITIVE'))
    df['중립수'] = df['개별감정평가'].apply(lambda x: x.split(' | ').count('NEUTRAL'))
    df['부정수'] = df['개별감정평가'].apply(lambda x: x.split(' | ').count('NEGATIVE'))
    return df
