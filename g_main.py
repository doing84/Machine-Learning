import logging
from g_data_loader import DataLoader
from g_model_trainer import ModelTrainer
from g_predictor import Predictor

logging.basicConfig(level=logging.INFO)

def main():
    try:
        logging.info("Starting data loading and preprocessing")
        data_loader = DataLoader('result_data.csv')
        data = data_loader.load_data()
        processed_data = data_loader.preprocess_data(data)
        logging.info("Data loading and preprocessing completed")
        
        logging.info("Starting model training")
        model_trainer = ModelTrainer(processed_data)
        X_train, X_test, y_train, y_test = model_trainer.prepare_data()
        
        best_model = model_trainer.cross_validate_model(X_train, y_train)
        
        trained_model = model_trainer.train_model(best_model, X_train, y_train)
        logging.info("Model training completed")

        logging.info("Starting model evaluation")
        predictor = Predictor(trained_model, X_test, y_test, model_trainer.features)
        y_pred, mse, r2, mre = predictor.evaluate_model()
        logging.info(f"Model evaluation completed with MSE: {mse}, R² Score: {r2}, MRE: {mre}%")

        predictor.plot_predictions(y_test, y_pred)

        # 미래 데이터 예측
        mean_values = processed_data[model_trainer.features].mean().to_dict()
        variance_values = processed_data[model_trainer.features].std().to_dict()
        future_data = predictor.create_future_data('2024-07-02', '2025-12-31', mean_values, variance_values)
        future_prices = predictor.predict_future(future_data)
        
        predictor.plot_future_predictions(future_data.index, future_prices)
        predictor.save_predictions_to_csv(future_data.index, future_prices, 'future_price_predictions.csv')

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == '__main__':
    main()
