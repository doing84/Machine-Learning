import numpy as np
import matplotlib.pyplot as plt
import logging
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)

class Predictor:
    def __init__(self, model, X_test, y_test, features):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.features = features

    def evaluate_model(self):
        try:
            logging.info("Evaluating the model...")
            y_pred = self.model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            mre = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
            logging.info(f'Mean Squared Error (MSE): {mse}')
            logging.info(f'R² Score: {r2}')
            logging.info(f'Mean Relative Error (MRE): {mre}%')
            return y_pred, mse, r2, mre
        except Exception as e:
            logging.error(f"Error evaluating model: {e}")
            raise

    def plot_predictions(self, y_test, y_pred):
        try:
            logging.info("Plotting predictions...")
            plt.figure(figsize=(10, 5))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
            plt.xlabel('Actual Prices')
            plt.ylabel('Predicted Prices')
            plt.title('Actual vs Predicted Prices')
            plt.grid(True)
            plt.show()
            logging.info("Plotting completed")
        except Exception as e:
            logging.error(f"Error plotting predictions: {e}")
            raise

    def predict_future(self, future_data):
        try:
            logging.info("Predicting future prices...")
            future_prices = self.model.predict(future_data[self.features])
            logging.info("Future prediction completed")
            return future_prices
        except Exception as e:
            logging.error(f"Error predicting future prices: {e}")
            raise

    def create_future_data(self, start_date, end_date, mean_values, variance_values):
        try:
            logging.info("Creating future data...")
            future_dates = pd.date_range(start=start_date, end=end_date)
            future_data = pd.DataFrame(index=future_dates)
            future_data['월'] = future_data.index.month
            future_data['요일'] = future_data.index.weekday
            future_data['계절'] = ((future_data['월'] % 12 + 3) // 3)
            for feature in mean_values.keys():
                future_data[feature] = np.random.normal(mean_values[feature], variance_values[feature], len(future_data))
            logging.info("Future data created successfully")
            return future_data
        except Exception as e:
            logging.error(f"Error creating future data: {e}")
            raise

    def plot_future_predictions(self, future_dates, future_prices):
        try:
            logging.info("Plotting future predictions...")
            plt.figure(figsize=(10, 5))
            plt.plot(future_dates, future_prices, marker='o', linestyle='-', color='blue')
            plt.xlabel('Date')
            plt.ylabel('Predicted Price')
            plt.title('Future Price Predictions')
            plt.grid(True)
            plt.show()
            logging.info("Future predictions plotting completed")
        except Exception as e:
            logging.error(f"Error plotting future predictions: {e}")
            raise

    def save_predictions_to_csv(self, future_dates, future_prices, filename):
        try:
            logging.info("Saving predictions to CSV...")
            future_data = pd.DataFrame({'Date': future_dates, 'Predicted_Price': future_prices})
            future_data.to_csv(filename, index=False)
            logging.info(f"Predictions saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving predictions to CSV: {e}")
            raise
