import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from a_data_loader import DataLoader
from a_model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)

def set_korean_font():
    """한글 폰트 설정 함수"""
    import platform
    from matplotlib import font_manager, rc
    if platform.system() == "Darwin":  # 맥OS
        rc("font", family="AppleGothic")
    elif platform.system() == "Windows":  # 윈도우
        path = "c:/Windows/Fonts/malgun.ttf"
        font_name = font_manager.FontProperties(fname=path).get_name()
        rc("font", family=font_name)
    else:  # 그 외 OS
        print("Unknown operating system. No Korean font settings applied.")
    plt.rcParams["axes.unicode_minus"] = False  # 마이너스 폰트 깨짐 방지

def plot_feature_importance(model, feature_names):
    try:
        logging.info("Plotting feature importance...")
        coef = model.coef_  # 'named_steps'가 아닌 모델 객체에서 직접 'coef_'를 사용
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': coef})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
        plt.xlabel('Coefficient Value')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()  # 가장 중요한 피처가 위에 오도록 설정
        plt.grid(True)
        plt.show()
        logging.info("Feature importance plotting completed")
    except Exception as e:
        logging.error(f"Error plotting feature importance: {e}")
        raise

def plot_correlation_matrix(data):
    try:
        logging.info("Plotting correlation matrix...")
        # 범주형 변수를 수치형으로 변환
        for column in data.select_dtypes(include=['category', 'object']).columns:
            data[column] = data[column].astype('category').cat.codes
        corr = data.corr()
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
        plt.title('Correlation Matrix')
        plt.show()
        logging.info("Correlation matrix plotting completed")
    except Exception as e:
        logging.error(f"Error plotting correlation matrix: {e}")
        raise

def plot_histograms(data):
    try:
        logging.info("Plotting histograms...")
        data.hist(bins=30, figsize=(20, 15), layout=(6, 5))
        plt.tight_layout()
        plt.show()
        logging.info("Histograms plotting completed")
    except Exception as e:
        logging.error(f"Error plotting histograms: {e}")
        raise

def plot_scatter_plots(data, target):
    try:
        logging.info("Plotting scatter plots...")
        features = data.columns.drop(target)
        plt.figure(figsize=(20, 15))
        for i, feature in enumerate(features):
            plt.subplot(6, 5, i + 1)
            plt.scatter(data[feature], data[target], alpha=0.5)
            plt.xlabel(feature)
            plt.ylabel(target)
        plt.tight_layout()
        plt.show()
        logging.info("Scatter plots plotting completed")
    except Exception as e:
        logging.error(f"Error plotting scatter plots: {e}")
        raise

def save_feature_importance_to_csv(model, features, model_name, file_name):
    importance = model.coef_  # 'named_steps'가 아닌 모델 객체에서 직접 'coef_'를 사용
    feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    feature_importance.to_csv(file_name, index=False, encoding='utf-8-sig')
    logging.info(f"{model_name} feature importance saved to {file_name}")

def main():
    try:
        set_korean_font()  # 한글 폰트 설정

        logging.info("Starting data loading and preprocessing for visualizations")
        data_loader = DataLoader('result_data.csv')  # 데이터 로드
        data = data_loader.load_data()
        processed_data = data_loader.preprocess_data(data)
        logging.info("Data loading and preprocessing completed")
        
        logging.info("Starting model training for feature importance")
        model_trainer = ModelTrainer(processed_data)  # 모델 트레이너 초기화
        X_train, X_test, y_train, y_test = model_trainer.prepare_data()  # 데이터 준비
        model, scaler = model_trainer.train_and_evaluate_model(X_train, y_train, model_type='ridge')  # 모델 빌드
        
        logging.info("Model training completed")

        plot_feature_importance(model, model_trainer.features)  # 피처 중요도 시각화 및 저장
        save_feature_importance_to_csv(model, model_trainer.features, "Ridge Regression", 'ridge_feature_importance.csv')

        plot_correlation_matrix(processed_data)  # 상관행렬 시각화
        plot_histograms(processed_data)  # 히스토그램 시각화
        plot_scatter_plots(processed_data, 'avg_price')  # 스캐터 플롯 시각화
    except Exception as e:
        logging.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
