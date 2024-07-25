import logging
import pandas as pd

from a_data_loader import DataLoader
from a_model_trainer import ModelTrainer
from a_predictor import Predictor

logging.basicConfig(level=logging.INFO)

def save_feature_importance_to_csv(model, features, model_name, file_name):
    """
    모델의 피처 중요도를 CSV 파일로 저장합니다.

    Parameters:
    model (sklearn model): 훈련된 모델
    features (list): 피처 이름 리스트
    model_name (str): 모델 이름
    file_name (str): 저장할 CSV 파일 이름
    """
    try:
        # 모델의 계수 추출
        importance = model.coef_
        # 피처 중요도 데이터프레임 생성
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
        # 중요도 순으로 정렬
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
        # CSV 파일로 저장
        feature_importance.to_csv(file_name, index=False, encoding='utf-8-sig')
        logging.info(f"{model_name} feature importance saved to {file_name}")
    except Exception as e:
        logging.error(f"Error saving feature importance: {e}")
        raise

def main():
    try:
        logging.info("Starting data loading and preprocessing")
        # 데이터 로드 및 전처리
        data_loader = DataLoader('result_data.csv')
        data = data_loader.load_data()
        processed_data = data_loader.preprocess_data(data)
        logging.info("Data loading and preprocessing completed")
        
        logging.info("Starting model training")
        # 모델 훈련 준비
        model_trainer = ModelTrainer(processed_data)
        X_train, X_test, y_train, y_test = model_trainer.prepare_data()
        
        # Ridge 회귀 모델 훈련
        ridge_model, scaler = model_trainer.train_and_evaluate_model(X_train, y_train, model_type='ridge')
        logging.info("Ridge regression model training completed")

        # Lasso 회귀 모델 훈련
        lasso_model, scaler = model_trainer.train_and_evaluate_model(X_train, y_train, model_type='lasso')
        logging.info("Lasso regression model training completed")

        logging.info("Starting model evaluation")
        # Ridge 모델 평가 및 미래 예측
        ridge_predictor = Predictor(ridge_model, model_trainer.features, scaler)
        ridge_predictor.evaluate_model(X_test, y_test)
        ridge_predictor.predict_future(processed_data, 'ridge_future_price_predictions.csv')
        
        # Lasso 모델 평가 및 미래 예측
        lasso_predictor = Predictor(lasso_model, model_trainer.features, scaler)
        lasso_predictor.evaluate_model(X_test, y_test)
        lasso_predictor.predict_future(processed_data, 'lasso_future_price_predictions.csv')
        
        logging.info("Model evaluation and prediction completed")

        # 피처 중요도 CSV로 저장
        save_feature_importance_to_csv(ridge_model, model_trainer.features, "Ridge Regression", 'ridge_feature_importance.csv')
        save_feature_importance_to_csv(lasso_model, model_trainer.features, "Lasso Regression", 'lasso_feature_importance.csv')

    except Exception as e:
        logging.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
