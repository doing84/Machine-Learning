import numpy as np
import matplotlib.pyplot as plt
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

class Predictor:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    # 모델을 평가하는 함수
    def evaluate_model(self):
        try:
            logging.info("Evaluating the model...")
            y_pred = self.model.predict(self.X_test)  # 예측 수행
            if y_pred.ndim > 1:  # y_pred가 다차원 배열인 경우
                y_pred = y_pred.ravel()  # 1차원 배열로 변환
            mse = np.mean((self.y_test - y_pred) ** 2)  # MSE 계산
            logging.info(f'Mean Squared Error: {mse}')
            return y_pred, mse
        except Exception as e:
            logging.error(f"Error evaluating model: {e}")
            raise

    # 미래 데이터를 예측하는 함수
    def predict_future(self, future_data):
        try:
            logging.info("Predicting future prices...")
            future_prices = self.model.predict(future_data)  # 미래 데이터에 대한 예측 수행
            if future_prices.ndim > 1:  # future_prices가 다차원 배열인 경우
                future_prices = future_prices.ravel()  # 1차원 배열로 변환
            return future_prices
        except Exception as e:
            logging.error(f"Error predicting future prices: {e}")
            raise

    # 예측 결과를 시각화하는 함수
    def plot_predictions(self, future_dates, future_prices):
        try:
            logging.info("Plotting predictions...")
            plt.figure(figsize=(10, 5))
            plt.plot(future_dates, future_prices, marker='o', linestyle='-', color='blue')  # 예측 결과 시각화
            plt.title('Predicted Future Prices')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True)
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting predictions: {e}")
            raise

    # 예측 결과를 CSV 파일로 저장하는 함수
    def save_predictions_to_csv(self, product_name, future_dates, future_prices, file_name='predicted_prices.csv'):
        try:
            logging.info("Saving predictions to CSV...")
            future_data = pd.DataFrame({'Product_Name': product_name, 'Date': future_dates, 'Predicted_Price': future_prices})
            future_data.to_csv(file_name, index=False)  # CSV 파일로 저장
            logging.info(f"Predictions saved to {file_name}")
        except Exception as e:
            logging.error(f"Error saving predictions to CSV: {e}")
            raise
