import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

class Predictor:
    def __init__(self, model, features, scaler):
        """
        모델, 피처 목록, 스케일러를 초기화합니다.
        """
        self.model = model
        self.features = features
        self.scaler = scaler

    def evaluate_model(self, X_test, y_test):
        """
        테스트 데이터를 사용하여 모델을 평가하고 예측 결과를 시각화합니다.

        Parameters:
        X_test (pd.DataFrame): 테스트 피처
        y_test (pd.Series): 테스트 타겟 변수
        """
        try:
            # 테스트 데이터 스케일링
            X_test_scaled = self.scaler.transform(X_test)
            # 예측 수행
            y_pred = self.model.predict(X_test_scaled)
            # 평균 제곱 오차 계산
            mse = np.mean((y_test - y_pred) ** 2)
            logging.info(f"Mean Squared Error: {mse}")

            # 실제 값 vs 예측 값 산점도 시각화
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.xlabel('Actual Prices')
            plt.ylabel('Predicted Prices')
            plt.title('Actual vs Predicted Prices')
            plt.show()

        except Exception as e:
            logging.error(f"모델 평가 중 오류 발생: {e}")
            raise

    def create_future_data(self, data, periods=365):
        """
        미래 데이터를 생성합니다.

        Parameters:
        data (pd.DataFrame): 기존 데이터
        periods (int): 예측할 기간 (일 단위)

        Returns:
        tuple: future_data, future_data_scaled
        """
        try:
            # 마지막 날짜 이후의 미래 날짜 생성
            last_date = data['date'].max()
            future_dates = pd.date_range(start=last_date, periods=periods + 1)[1:]
            future_data = pd.DataFrame(future_dates, columns=['date'])
            future_data['month'] = future_data['date'].dt.month
            future_data['day_of_week'] = future_data['date'].dt.dayofweek
            future_data['season'] = (future_data['month'] % 12 + 3) // 3

            # 피처별로 데이터 채우기
            for feature in self.features:
                if feature in future_data.columns:
                    continue
                if feature in data.columns:
                    future_data[feature] = np.random.choice(data[feature].values, size=periods, replace=True)
                else:
                    future_data[feature] = 0

            logging.info(f"미래 데이터:\n{future_data.head()}")

            # 미래 데이터 스케일링
            future_data_scaled = self.scaler.transform(future_data[self.features])
            return future_data, future_data_scaled

        except Exception as e:
            logging.error(f"미래 데이터 생성 중 오류 발생: {e}")
            raise

    def predict_future(self, data, output_file):
        """
        미래 데이터를 예측하고 결과를 CSV 파일로 저장합니다.

        Parameters:
        data (pd.DataFrame): 기존 데이터
        output_file (str): 예측 결과를 저장할 파일 경로
        """
        try:
            # 미래 데이터 생성
            future_data, future_data_scaled = self.create_future_data(data)
            # 미래 가격 예측
            future_data['predicted_price'] = self.model.predict(future_data_scaled)

            # 예측 결과를 CSV 파일로 저장
            output_data = future_data[['date', 'predicted_price']]
            output_data.to_csv(output_file, index=False, encoding='utf-8-sig')

            logging.info(f"예측 결과가 {output_file}에 저장되었습니다.")

        except Exception as e:
            logging.error(f"미래 가격 예측 중 오류 발생: {e}")
            raise
