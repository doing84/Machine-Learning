import pandas as pd
import numpy as np
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
            data['date'] = pd.to_datetime(data['date'])
            data['month'] = data['date'].dt.month
            data['day_of_week'] = data['date'].dt.dayofweek
            data['season'] = (data['month'] % 12 + 3) // 3
            logging.info("Data preprocessing completed")
            return data
        except Exception as e:
            logging.error(f"Error preprocessing data: {e}")
            raise
