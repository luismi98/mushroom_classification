import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import argparse
from utils.logger import logging

def load_data(train_path, test_path, target_column):
    """
    Load the preprocessed training and test data from CSV files and separate features and target.
    
    Parameters
    ----------
    train_path : str
        Path to the training data CSV file.
    test_path : str
        Path to the test data CSV file.
    target_column : str
        Name of the target column.
    
    Returns
    -------
    tuple, containing:
        - X_train : DataFrame
            Training features.
        - y_train : Series
            Training target.
        - X_test : DataFrame
            Test features.
        - y_test : Series
            Test target.
    """

    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise
    
    if target_column not in train_data.columns or target_column not in test_data.columns:
        logging.error(f"Target column `{target_column}` not found in the dataset.")
        raise ValueError(f"Target column `{target_column}` not found in the dataset.")
    
    X_train = train_data.drop(target_column, axis=1)
    y_train = train_data[target_column]
    X_test = test_data.drop(target_column, axis=1)
    y_test = test_data[target_column]
    
    logging.info("Data loaded successfully")
    
    return X_train, y_train, X_test, y_test

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_path="model.pkl", use_tuning=False):
    """
    Train a Random Forest model, optionally perform hyperparameter tuning, and evaluate its performance.
    
    Parameters
    ----------
    X_train : DataFrame
        Training features.
    y_train : Series
        Training target.
    X_test : DataFrame
        Test features.
    y_test : Series
        Test target.
    model_path : str, optional
        Path to save the trained model (default is "model.pkl").
    use_tuning : bool, optional
        Whether to perform hyperparameter tuning (default is False).
    
    Returns
    -------
    dict
        Evaluation metrics including accuracy, precision, recall, and f1 score.
    """

    try:
        if use_tuning:
            
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
            grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                                       param_grid=param_grid,
                                       cv=5,
                                       n_jobs=-1,
                                       scoring="accuracy")
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            logging.info(f"Best parameters found: {grid_search.best_params_}")
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        model.fit(X_train, y_train)
        
        joblib.dump(model, model_path)
        logging.info(f"Trained model saved to {model_path}")
        
        y_pred = model.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }
        
        logging.info(f"Model evaluation metrics: {metrics}")
        
        return metrics
    
    except Exception as e:
        logging.error(f"Error during model training and evaluation: {e}")
        raise

def main(train_path, test_path, target_column, model_path, use_tuning):
    try:
        X_train, y_train, X_test, y_test = load_data(train_path, test_path, target_column)
        metrics = train_and_evaluate_model(X_train, y_train, X_test, y_test, model_path, use_tuning)
        
        print(f"Model evaluation metrics:\n{metrics}")
    
    except Exception as e:
        logging.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classification model with optional hyperparameter tuning and evaluate its performance on the test set.")
    parser.add_argument("train_path", type=str, help="Path to the processed training data CSV file.")
    parser.add_argument("test_path", type=str, help="Path to the processed test data CSV file.")
    parser.add_argument("--target_column", type=str, default="target", help="Name of the target column.")
    parser.add_argument("--model_path", type=str, default="model.pkl", help="Path to save the trained model.")
    parser.add_argument("--use_tuning", action="store_true", help="Use hyperparameter tuning.")

    args = parser.parse_args()
    
    main(args.train_path, args.test_path, args.target_column, args.model_path, args.use_tuning)
