import pandas as pd
import matplotlib.pyplot as plt

# 예측 결과 CSV 파일 로드
ridge_data = pd.read_csv('ridge_future_price_predictions.csv')
lasso_data = pd.read_csv('lasso_future_price_predictions.csv')

# 날짜 열을 datetime 형식으로 변환
ridge_data['date'] = pd.to_datetime(ridge_data['date'])
lasso_data['date'] = pd.to_datetime(lasso_data['date'])

# 월별로 그룹화하여 평균 예측 가격 계산
ridge_data['month'] = ridge_data['date'].dt.to_period('M')
lasso_data['month'] = lasso_data['date'].dt.to_period('M')

ridge_monthly_avg = ridge_data.groupby('month')['predicted_price'].mean().reset_index()
lasso_monthly_avg = lasso_data.groupby('month')['predicted_price'].mean().reset_index()

# 월별 평균 예측 가격 그래프
plt.figure(figsize=(14, 7))
plt.plot(ridge_monthly_avg['month'].astype(str), ridge_monthly_avg['predicted_price'], label='Ridge Predicted Price', color='blue', marker='o')
plt.xlabel('Month')
plt.ylabel('Predicted Price')
plt.title('Monthly Average Predicted Prices using Ridge Regression')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(lasso_monthly_avg['month'].astype(str), lasso_monthly_avg['predicted_price'], label='Lasso Predicted Price', color='green', marker='x')
plt.xlabel('Month')
plt.ylabel('Predicted Price')
plt.title('Monthly Average Predicted Prices using Lasso Regression')
plt.legend()
plt.grid(True)
plt.show()
