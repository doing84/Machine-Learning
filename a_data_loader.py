import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        try:
            logging.info("Loading data...")
            data = pd.read_csv(self.file_path)
            logging.info("Data loaded successfully")
            return data
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def preprocess_data(self, data):
        try:
            logging.info("Preprocessing data...")
            # 날짜 형식 변환
            data['date'] = pd.to_datetime(data['date'])
            data['Month'] = data['date'].dt.month
            data['DayOfWeek'] = data['date'].dt.dayofweek
            data['Season'] = data['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else (
                'Spring' if x in [3, 4, 5] else (
                'Summer' if x in [6, 7, 8] else (
                'Fall' if x in [9, 10, 11] else 'Unknown'))))
            logging.info("Data preprocessing completed")
            return data
        except Exception as e:
            logging.error(f"Error preprocessing data: {e}")
            raise
