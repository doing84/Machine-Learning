from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

class ModelTrainer:
    def __init__(self, data):
        self.data = data
        self.features = [
            '인스타_좋아요수', '인스타_댓글수', '블로그_긍정수', '블로그_전체수', '블로그_부정수', 
            '뉴스_긍정수', '뉴스_전체수', '뉴스_부정수', '검색트렌드', '기온', '강수량', '습도', 
            '풍속', '구름량', '날씨_코드', '월', '요일', '계절', '유튜브_조회수', '유튜브_좋아요수', 
            '유튜브_댓글수', '유튜브_긍정수', '유튜브_부정수'
        ]

    def prepare_data(self):
        try:
            logging.info("Preparing data for training...")
            X = self.data[self.features]
            y = self.data['avg_price']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logging.info("Data preparation completed")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            raise

    def build_model(self):
        try:
            logging.info("Building the Gradient Boosting model...")
            model = GradientBoostingRegressor(random_state=42)
            logging.info("Model built successfully")
            return model
        except Exception as e:
            logging.error(f"Error building model: {e}")
            raise

    def cross_validate_model(self, X, y):
        try:
            logging.info("Cross-validating the model...")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5]
            }
            grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X, y)
            logging.info(f"Best parameters found: {grid_search.best_params_}")
            return grid_search.best_estimator_
        except Exception as e:
            logging.error(f"Error during cross-validation: {e}")
            raise

    def train_model(self, model, X_train, y_train):
        try:
            logging.info("Training the Gradient Boosting model...")
            model.fit(X_train, y_train)
            logging.info("Model training completed")
            return model
        except Exception as e:
            logging.error(f"Error training model: {e}")
            raise
