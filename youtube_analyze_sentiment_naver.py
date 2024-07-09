print("youtube_analyze_sentiment_naver 모듈 로드됨")

import os
import pandas as pd
import json

# 여러 개 json 파일 하나로 합치기
def combine_data_files(data_path):
    all_data = []
    for filename in os.listdir(data_path):
        if filename.endswith(".json"):
            filepath = os.path.join(data_path, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
                all_data.extend(data) 
    return all_data


# 각 문장별 감정 점수 추출
def extract_aspect_sentiments(aspects):
    aspect_sentiments = []
    for aspect in aspects:
        sentiment_text = aspect['SentimentText']
        sentiment_polarity = int(aspect['SentimentPolarity'])
        aspect_sentiments.append({'SentimentText': sentiment_text, 'SentimentPolarity': sentiment_polarity})
    return aspect_sentiments


# 각 리뷰별 감정 점수 분리 
def split_review_sentiments(df):
    rows = []
    for idx, row in df.iterrows():
        for aspect in row['AspectSentiments']:
            rows.append({'Index': row['Index'], 'Text': aspect['SentimentText'], 'Sentiment': aspect['SentimentPolarity']})
    return pd.DataFrame(rows)


def check_and_remove_duplicates(df):
    # 중복 확인
    duplicated_df = df[df.duplicated(subset=['Index', 'Text'], keep=False)]
    print("중복된 리뷰 수: ", df.duplicated(subset=['Index', 'Text']).sum())
    print("중복된 리뷰 예시:\n", duplicated_df.head(20))  # 중복된 리뷰 예시 출력

    # 중복된 리뷰 제거
    df = df.drop_duplicates(subset=['Index', 'Text'])
    print("중복 제거 후 리뷰 수: ", len(df))
    return df