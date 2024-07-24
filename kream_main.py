import logging
from kream_data_loader import DataLoader
from kream_model_trainer import ModelTrainer
from kream_predictor import Predictor
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

def main():
    start_time = time.time()  # 시작 시간 기록
    try:
        # 데이터 로딩 및 전처리
        data_loader = DataLoader('TIMBERLAND 6 IN.xlsx', 'youtube_data_sentiment.csv')
        product_data, youtube_data = data_loader.load_data()
        merged_data = data_loader.preprocess_data(product_data, youtube_data)
        
        # 데이터 시각화
        plt.figure(figsize=(10, 5))
        plt.plot(merged_data['년'].astype(str) + '-' + merged_data['월'].astype(str), merged_data['거래 금액'], marker='o', linestyle='-', color='blue')
        plt.title('Transaction Amount Over Time')
        plt.xlabel('Date')
        plt.ylabel('Transaction Amount')
        plt.grid(True)
        plt.show()

        # 상품명 추출
        product_name = merged_data['상품명'].iloc[0]

        # 피처와 타겟 분리
        features = ['조회수', '좋아요수', '댓글수', '긍정수', '중립수', '부정수']
        target = ['거래 금액']
        
        # 데이터 스케일링
        scaler_features = StandardScaler()
        scaler_target = StandardScaler()
        merged_data[features] = scaler_features.fit_transform(merged_data[features])
        merged_data[target] = scaler_target.fit_transform(merged_data[target])

        # 모델 학습
        model_trainer = ModelTrainer(merged_data)
        X_train, X_test, y_train, y_test = model_trainer.prepare_data()
        
        # 교차 검증
        model_trainer.cross_validate_model(X_train, y_train)
        
        # 모델 학습
        model = model_trainer.build_model()
        trained_model = model_trainer.train_model(model, X_train, y_train)

        # 예측 및 평가
        predictor = Predictor(trained_model, X_test, y_test)
        y_pred, mse = predictor.evaluate_model()

        # 미래 예측
        last_year = merged_data['년'].max()
        last_month = merged_data['월'][merged_data['년'] == last_year].max()
        future_years = [last_year + (last_month + i) // 12 for i in range(1, 25)]
        future_months = [(last_month + i) % 12 if (last_month + i) % 12 != 0 else 12 for i in range(1, 25)]
        future_dates = pd.DataFrame({'년': future_years, '월': future_months})
        
        future_data = future_dates.assign(
            조회수=np.mean(merged_data['조회수']),
            좋아요수=np.mean(merged_data['좋아요수']),
            댓글수=np.mean(merged_data['댓글수']),
            긍정수=np.mean(merged_data['긍정수']),
            중립수=np.mean(merged_data['중립수']),
            부정수=np.mean(merged_data['부정수'])
        )
        future_data[features] = scaler_features.transform(future_data[features])
        future_prices = predictor.predict_future(future_data[features])
        future_prices = scaler_target.inverse_transform(future_prices.reshape(-1, 1)).flatten()  # 스케일링 원복 후 1차원 배열로 변환
        
        # 예측 금액을 소수점 첫째 자리까지 포맷팅
        future_prices = np.round(future_prices, 1)

        # future_dates를 문자열 형식의 단일 열로 변환
        future_dates_str = future_dates['년'].astype(str) + '-' + future_dates['월'].astype(str)

        predictor.plot_predictions(future_dates_str, future_prices)
        predictor.save_predictions_to_csv(product_name, future_dates_str, future_prices)  # CSV 파일로 예측 결과 저장

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise
    finally:
        end_time = time.time()  # 끝나는 시간 기록
        elapsed_time = end_time - start_time  # 총 걸린 시간 계산
        logging.info(f"Execution started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        logging.info(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        logging.info(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    main()
