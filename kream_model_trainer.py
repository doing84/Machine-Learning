import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import logging
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

# GPU 메모리 사용 조정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # 동적 메모리 할당
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=14336)])  # 메모리 제한 설정 (예: 14GB)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        logging.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        logging.error(f"Error setting up GPU configuration: {e}")

class ModelTrainer:
    def __init__(self, data):
        self.data = data

    # 데이터를 학습과 테스트용으로 준비하는 함수
    def prepare_data(self):
        try:
            logging.info("Preparing data for training...")
            X = self.data[['조회수', '좋아요수', '댓글수', '긍정수', '중립수', '부정수']]  # 피처 선택
            y = self.data['거래 금액']  # 타겟 변수 선택
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 데이터 분할
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            raise

    # Keras 모델을 정의하는 함수
    def build_keras_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=6, activation='relu'))  # 입력층 및 은닉층 추가, input_dim 변경
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))  # 출력층 추가
        model.compile(optimizer='adam', loss='mean_squared_error')  # 모델 컴파일
        return model

    # 모델을 정의하는 함수
    def build_model(self):
        try:
            logging.info("Building the model...")
            with tf.device('/GPU:0'):  # GPU 사용을 명시
                model = self.build_keras_model()
            return model
        except Exception as e:
            logging.error(f"Error building model: {e}")
            raise

    # 교차 검증을 사용하는 함수
    def cross_validate_model(self, X, y):
        try:
            logging.info("Cross-validating the model...")
            keras_regressor = KerasRegressor(build_fn=self.build_keras_model, epochs=50, batch_size=120, verbose=0)
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            results = cross_val_score(keras_regressor, X, y, cv=kfold, scoring='neg_mean_squared_error')
            logging.info(f"Cross-validation MSE: {np.mean(results)}")
            return results
        except Exception as e:
            logging.error(f"Error during cross-validation: {e}")
            raise

    # 모델을 학습시키는 함수
    def train_model(self, model, X_train, y_train):
        try:
            logging.info("Training the model...")
            with tf.device('/GPU:0'):  # GPU 사용을 명시
                model.fit(X_train, y_train, epochs=50, batch_size=120, verbose=0, callbacks=[TqdmCallback()])
            return model
        except Exception as e:
            logging.error(f"Error training model: {e}")
            raise

class TqdmCallback(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epochs = kwargs.get('epochs', 50)
        self.batch_size = kwargs.get('batch_size', 120)
        self.samples = kwargs.get('samples', 1)
        self.steps_per_epoch = (self.samples + self.batch_size - 1) // self.batch_size
        self.epoch_progress_bar = None

    def on_train_begin(self, logs=None):
        self.epoch_progress_bar = tqdm(total=self.epochs, desc='Training Progress', unit='epoch')

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_progress_bar.update(1)

    def on_train_end(self, logs=None):
        self.epoch_progress_bar.close()
