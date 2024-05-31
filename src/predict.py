import pandas as pd
import joblib
import argparse
from utils.logger import logging

def load_model(model_path):
    try:
        model = joblib.load(model_path)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def load_preprocessor(preprocessor_path):
    try:
        preprocessor = joblib.load(preprocessor_path)
        logging.info(f"Preprocessor loaded from {preprocessor_path}")
        return preprocessor
    except Exception as e:
        logging.error(f"Error loading preprocessor: {e}")
        raise

def make_predictions(model, preprocessor, data_path, output_path):
    """
    Make predictions on new data using a loaded model and preprocessor, previously trained/fitted to the training data.
    
    Parameters
    ----------
    model : object
        The trained model.
    preprocessor : object
        The fitted preprocessor.
    data_path : str
        Path of the new data CSV file.
    output_path : str
        Path to save the predictions.
    """
    try:
        data = pd.read_csv(data_path)
        logging.info(f"Data loaded from {data_path}")
        
        features = preprocessor.transform(data)
        logging.info("Data preprocessed")
        
        predictions = model.predict(features)
        logging.info("Predictions made")
        
        prediction_df = pd.DataFrame(predictions, columns=["prediction"])
        prediction_df.to_csv(output_path, index=False)
        logging.info(f"Predictions saved to {output_path}")
        
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise

def main(model_path, preprocessor_path, data_path, output_path):
    try:
        model = load_model(model_path)
        preprocessor = load_preprocessor(preprocessor_path)
        make_predictions(model, preprocessor, data_path, output_path)
    except Exception as e:
        logging.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions on new data using a trained model and preprocessor.")
    parser.add_argument("model_path", type=str, help="Path to the trained model file.")
    parser.add_argument("preprocessor_path", type=str, help="Path to the preprocessor file.")
    parser.add_argument("data_path", type=str, help="Path to the new data CSV file.")
    parser.add_argument("output_path", type=str, help="Path to save the predictions.")

    args = parser.parse_args()
    
    main(args.model_path, args.preprocessor_path, args.data_path, args.output_path)
