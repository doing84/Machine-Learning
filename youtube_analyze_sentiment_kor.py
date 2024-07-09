import os
import pandas as pd
import re
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import logging

# 환경 변수 설정
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# transformers 로깅 설정
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

# CSV 파일 로드
logger.info("Loading CSV file...")
df = pd.read_csv('youtube_data.csv')
logger.info("CSV file loaded successfully.")

# 감정 분석 파이프라인 로드 (한국어 감정 분석 모델 사용)
logger.info("Loading sentiment analysis model and tokenizer...")
# model_name = "monologg/koelectra-base-v3-discriminator"
model_name = "beomi/KcELECTRA-base-v2022"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
sentiment_pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=512, truncation=True)
logger.info("Sentiment analysis model and tokenizer loaded successfully.")

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
            result = sentiment_pipeline(preprocessed_comment)[0]
            logger.info(f"Comment: {preprocessed_comment}, Sentiment: {result}")
            label = result['label']
            if label == 'LABEL_0':
                sentiments.append('NEGATIVE')
            elif label == 'LABEL_1':
                sentiments.append('NEUTRAL')
            elif label == 'LABEL_2':
                sentiments.append('POSITIVE')
    return sentiments

# 감정 분석 수행
def analyze_sentiments(df):
    total_comments = len(df)
    logger.info(f"Starting sentiment analysis for {total_comments} comments...")
    
    df['전처리댓글내용'] = df['각댓글내용'].apply(lambda x: " | ".join([preprocess_text(comment) for comment in split_comments(x)]))
    df['개별감정평가'] = df['전처리댓글내용'].apply(lambda x: " | ".join(analyze_sentiment(split_comments(x))))
    
    df['긍정수'] = df['개별감정평가'].apply(lambda x: x.split(' | ').count('POSITIVE'))
    df['중립수'] = df['개별감정평가'].apply(lambda x: x.split(' | ').count('NEUTRAL'))
    df['부정수'] = df['개별감정평가'].apply(lambda x: x.split(' | ').count('NEGATIVE'))
    
    logger.info("Sentiment analysis completed.")
    return df

if __name__ == "__main__":
    logger.info("Starting sentiment analysis process...")
    
    # 감정 분석 수행
    df = analyze_sentiments(df)
    
    # 결과 저장
    logger.info("Saving results to CSV file...")
    df.to_csv('youtube_data_sentiment.csv', index=False, encoding='utf-8-sig')
    logger.info("Data with sentiment saved to youtube_data_sentiment.csv.")
    print("Data with sentiment saved to youtube_data_sentiment.csv")
