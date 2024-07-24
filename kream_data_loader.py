import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

class DataLoader:
    def __init__(self, product_file, youtube_file):
        self.product_file = product_file
        self.youtube_file = youtube_file

    # 데이터를 불러오는 함수
    def load_data(self):
        try:
            logging.info("Loading product data...")
            product_data = pd.read_excel(self.product_file, engine='openpyxl')  # 상품 데이터 불러오기
            logging.info("Loading YouTube data...")
            youtube_data = pd.read_csv(self.youtube_file)  # 유튜브 데이터 불러오기
            return product_data, youtube_data
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    # 데이터를 전처리하는 함수
    def preprocess_data(self, product_data, youtube_data):
        try:
            logging.info("Preprocessing data...")
            product_data['체결 일자'] = pd.to_datetime(product_data['체결 일자'], format='%y/%m/%d')  # 날짜 형식 변환
            youtube_data['게시일'] = pd.to_datetime(youtube_data['게시일'])

            # 년, 월 컬럼 생성
            product_data['년'] = product_data['체결 일자'].dt.year
            product_data['월'] = product_data['체결 일자'].dt.month
            youtube_data['년'] = youtube_data['게시일'].dt.year
            youtube_data['월'] = youtube_data['게시일'].dt.month

            # 유튜브 데이터의 날짜별 통계 데이터 생성
            youtube_stats = youtube_data.groupby(['년', '월']).agg({
                '조회수': 'sum',
                '좋아요수': 'sum',
                '댓글수': 'sum',
                '긍정수': 'sum',
                '중립수': 'sum',
                '부정수': 'sum'
            }).reset_index()

            # 상품 데이터와 유튜브 데이터를 '년', '월' 기준으로 합치기
            merged_data = pd.merge(product_data, youtube_stats, on=['년', '월'], how='left')
            merged_data.fillna(0, inplace=True)  # 결측치 0으로 대체
            return merged_data
        except Exception as e:
            logging.error(f"Error preprocessing data: {e}")
            raise
