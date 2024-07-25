import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc
import platform

# 한글 폰트 설정
path = "C:/Windows/Fonts/malgun.ttf"
if platform.system() == "Darwin":
    rc("font", family="AppleGothic")
elif platform.system() == "Windows":
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc("font", family=font_name)
else:
    print("Unknown system")
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
file_path = 'future_price_predictions.csv'
data = pd.read_csv(file_path)

# 날짜 형식 변환
data['Date'] = pd.to_datetime(data['Date'])

# 계절, 월, 요일 컬럼 추가
data['Month'] = data['Date'].dt.month
data['DayOfWeek'] = data['Date'].dt.dayofweek

# 계절 설정
data['Season'] = data['Month'].apply(lambda x: '겨울' if x in [12, 1, 2] else (
    '봄' if x in [3, 4, 5] else (
    '여름' if x in [6, 7, 8] else (
    '가을' if x in [9, 10, 11] else 'Unknown'))))

# 계절별 데이터 분포 시각화
plt.figure(figsize=(10, 6))
sns.histplot(data, x='Predicted_Price', hue='Season', multiple='stack', palette='coolwarm')
plt.xlabel('Predicted Price')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Prices by Season')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
