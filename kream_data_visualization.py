import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from matplotlib import font_manager, rc
import logging

logging.basicConfig(level=logging.INFO)

# 한글 폰트 설정 함수
def set_korean_font():
    if platform.system() == "Darwin":
        rc("font", family="AppleGothic")
    elif platform.system() == "Windows":
        path = "C:/Windows/Fonts/malgun.ttf"
        font_name = font_manager.FontProperties(fname=path).get_name()
        rc("font", family=font_name)
    else:
        print("Unknown operating system. No Korean font settings applied.")
    plt.rcParams["axes.unicode_minus"] = False

# 히스토그램 그리기
def plot_histograms(data, columns, bins=30):
    try:
        logging.info("Plotting histograms...")
        data[columns].hist(bins=bins, figsize=(20, 15))
        plt.tight_layout()
        plt.show()
        logging.info("Histograms plotting completed")
    except Exception as e:
        logging.error(f"Error plotting histograms: {e}")

# 상관행렬 히트맵 그리기
def plot_correlation_matrix(data):
    try:
        logging.info("Plotting correlation matrix...")
        # date 열 제외
        data = data.select_dtypes(include=[np.number])
        corr = data.corr()
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
        plt.title('Correlation Matrix')
        plt.show()
        logging.info("Correlation matrix plotting completed")
    except Exception as e:
        logging.error(f"Error plotting correlation matrix: {e}")

# 산점도 행렬 그리기
def plot_scatter_matrix(data, columns):
    try:
        logging.info("Plotting scatter matrix...")
        sns.pairplot(data[columns])
        plt.show()
        logging.info("Scatter matrix plotting completed")
    except Exception as e:
        logging.error(f"Error plotting scatter matrix: {e}")

# 시계열 데이터 라인 그래프 그리기
def plot_time_series(data, date_column, value_column):
    try:
        logging.info("Plotting time series...")
        plt.figure(figsize=(14, 7))
        plt.plot(data[date_column], data[value_column])
        plt.xlabel('Date')
        plt.ylabel(value_column)
        plt.title(f'Time Series of {value_column}')
        plt.xticks(rotation=45)
        plt.show()
        logging.info("Time series plotting completed")
    except Exception as e:
        logging.error(f"Error plotting time series: {e}")

def main():
    set_korean_font()  # 한글 폰트 설정

    logging.info("Loading data...")
    data = pd.read_csv('result_data.csv')
    logging.info("Data loaded successfully")

    columns_to_plot = [
        '인스타_좋아요수', '인스타_댓글수', '블로그_긍정수', '블로그_전체수', '블로그_부정수', 
        '뉴스_긍정수', '뉴스_전체수', '뉴스_부정수', '검색트렌드', '기온', '강수량', '습도', 
        '풍속', '구름량', '날씨_코드', '월', '요일', '계절', '유튜브_조회수', '유튜브_좋아요수', 
        '유튜브_댓글수', '유튜브_긍정수', '유튜브_중립수', '유튜브_부정수', 'avg_price'
    ]

    # 히스토그램
    plot_histograms(data, columns_to_plot)

    # 상관행렬 히트맵
    plot_correlation_matrix(data)

    # 산점도 행렬
    plot_scatter_matrix(data, columns_to_plot)

    # 시계열 데이터 라인 그래프
    plot_time_series(data, 'date', 'avg_price')

if __name__ == "__main__":
    main()
