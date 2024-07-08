import pandas as pd
import re
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# CSV 파일 로드
df = pd.read_csv('youtube_data.csv')

# 감정 분석 파이프라인 로드 (다국어 감정 분석 모델 사용)
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=512, truncation=True, top_k=1)

# 텍스트 전처리 함수
def preprocess_text(text):
    # HTML 태그 제거
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+|www.\S+', '', text)
    # 특수 문자 및 숫자 제거
    text = re.sub(r'[^a-zA-Z가-힣\s]', '', text)
    # 소문자 변환
    text = text.lower()
    # 불필요한 공백 제거
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
        if len(preprocessed_comment) > 0:  # 빈 문자열이 아닌 경우에만 분석
            result = sentiment_pipeline(preprocessed_comment)[0][0]['label']
            if result == '1 star':
                sentiments.append('NEGATIVE')
            elif result == '2 stars':
                sentiments.append('NEGATIVE')
            elif result == '3 stars':
                sentiments.append('NEUTRAL')
            elif result == '4 stars':
                sentiments.append('POSITIVE')
            elif result == '5 stars':
                sentiments.append('POSITIVE')
    return sentiments

# 각 댓글을 전처리하고 전처리된 댓글을 새로운 컬럼에 저장
df['전처리댓글내용'] = df['각댓글내용'].apply(lambda x: " | ".join([preprocess_text(comment) for comment in split_comments(x)]))

# 전처리된 댓글에 대한 감정 분석 수행 및 결과 저장
df['개별감정평가'] = df['전처리댓글내용'].apply(lambda x: " | ".join(analyze_sentiment(split_comments(x))))

# 감정 분포 결과를 데이터프레임에 추가
df['긍정수'] = df['개별감정평가'].apply(lambda x: x.split(' | ').count('POSITIVE'))
df['중립수'] = df['개별감정평가'].apply(lambda x: x.split(' | ').count('NEUTRAL'))
df['부정수'] = df['개별감정평가'].apply(lambda x: x.split(' | ').count('NEGATIVE'))

# 결과 저장
df.to_csv('youtube_data_sentiment.csv', index=False, encoding='utf-8-sig')
print("Data with sentiment saved to youtube_data_sentiment.csv")
