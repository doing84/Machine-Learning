from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import pandas as pd
import numpy as np

class ModelTrainer:
    def __init__(self, data):
        """
        데이터와 피처 목록을 초기화합니다.
        """
        self.data = data
        self.features = [
            '인스타_좋아요수', '인스타_댓글수', '블로그_긍정수', '블로그_전체수', '블로그_부정수', 
            '뉴스_긍정수', '뉴스_전체수', '뉴스_부정수', '검색트렌드', '기온', '강수량', '습도', 
            '풍속', '구름량', '날씨_코드', '월', '요일', '계절', '유튜브_조회수', '유튜브_좋아요수', 
            '유튜브_댓글수', '유튜브_긍정수', '유튜브_중립수', '유튜브_부정수'
        ]

    def prepare_data(self):
        """
        데이터를 학습용과 테스트용으로 나눕니다.
        """
        try:
            logging.info("학습 데이터를 준비 중...")
            X = self.data[self.features]  # 피처 선택
            y = self.data['avg_price']  # 타겟 변수 선택
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logging.info("데이터 준비 완료")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"데이터 준비 중 오류 발생: {e}")
            raise

    def train_and_evaluate_model(self, X_train, y_train, model_type='ridge'):
        """
        Ridge 또는 Lasso 회귀 모델을 학습하고 평가합니다.

        Parameters:
        X_train (pd.DataFrame): 학습 피처
        y_train (pd.Series): 학습 타겟 변수
        model_type (str): 사용할 회귀 모델 타입 ('ridge' 또는 'lasso')

        Returns:
        tuple: best_model, scaler
        """
        try:
            # 데이터 스케일링
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # 하이퍼파라미터 그리드 설정
            param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]}

            # 모델 타입 선택
            if model_type == 'ridge':
                logging.info("Ridge 회귀 모델 생성 중...")
                model = Ridge()
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
            elif model_type == 'lasso':
                logging.info("Lasso 회귀 모델 생성 중...")
                model = Lasso()
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')

            # 그리드 서치 수행
            grid_search.fit(X_train_scaled, y_train)

            # 최적의 파라미터와 모델 가져오기
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_

            logging.info(f"{model_type}의 최적 파라미터: {best_params}")
            print(f"{model_type}의 최적 파라미터: {best_params}")

            # 학습 데이터에 대한 모델 성능 평가
            y_pred = best_model.predict(X_train_scaled)
            mse = np.mean((y_train - y_pred) ** 2)
            logging.info(f"{model_type}의 평균 제곱 오차 (MSE): {mse}")

            return best_model, scaler

        except Exception as e:
            logging.error(f"모델 학습 및 평가 중 오류 발생: {e}")
            raise
