import pandas as pd
import joblib
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from utils.logger import logging
from utils.custom_transformers import MissingValuesFeatureRemover, CustomCategoricalEncoder

def preprocess_and_save_data(filepath, target_column="class", train_save_path="train_processed.csv", test_save_path="test_processed.csv", preprocessor_save_path="preprocessor.pkl", 
                             test_size=0.2, random_state=42, missing_value_remover_threshold=0.2):
    """
    Preprocess the data, split into training and test sets, and save the processed data and preprocessor.
    
    Parameters
    ----------
    filepath : str
        Path to the raw data CSV file.
    target_column : str, optional
        Name of the target column (default is "class").
    train_save_path : str, optional
        Path to save the processed training data (default is "train_processed.csv").
    test_save_path : str, optional
        Path to save the processed test data (default is "test_processed.csv").
    preprocessor_save_path : str, optional
        Path to save the preprocessor (default is "preprocessor.pkl").
    test_size : float, optional
        Proportion of the dataset to include in the test split (default is 0.2).
    random_state : int, optional
        Random state for reproducibility (default is 42).
    missing_value_remover_threshold : float, optional
        Fractional threshold for column removal. Columns with missing values below this fraction are removed (default is 0.2).
    """
    
    try:
        data = pd.read_csv(filepath)
        logging.info(f"Data loaded from {filepath}")
        
        if target_column not in data.columns:
            logging.error(f"Target column `{target_column}` not found in the dataset.")
            raise ValueError(f"Target column `{target_column}` not found in the dataset.")
        
        if data[target_column].dtype == "object":
            data[target_column] = (data[target_column] == "p").astype(int)
        logging.info(f"Converted target `{target_column}` to boolean, with 1 representing poisonous.")

        if data.duplicated().any():
            logging.info(f"Dropping `{data.duplicated().sum()}` duplicates.")
            data.drop_duplicates(inplace=True)

        features = data.drop(target_column, axis=1)
        target = data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)
        logging.info("Data split into training and test sets.")
        
        numerical_features = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = data.select_dtypes(include=["object"]).columns.tolist()
        
        if target_column in numerical_features:
            numerical_features.remove(target_column)
        if target_column in categorical_features:
            categorical_features.remove(target_column)

        num_pipeline = Pipeline(
            steps = [
                ("missing_features_remover", MissingValuesFeatureRemover(threshold=missing_value_remover_threshold)),
                ("median_imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        cat_pipeline = Pipeline(
            steps = [
                ("missing_features_remover", MissingValuesFeatureRemover(threshold=missing_value_remover_threshold)),
                ("mode_imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", CustomCategoricalEncoder()),
                ("scaler",StandardScaler(with_mean=False))
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("num_pipeline", num_pipeline, numerical_features),
                ("cat_pipeline", cat_pipeline, categorical_features)
            ],
            verbose_feature_names_out = False
        )

        preprocessor.set_output(transform="pandas")

        # Fit the preprocessor on the training data before transforming the train and test data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        logging.info("Data preprocessed")

        train_processed = pd.DataFrame(X_train_processed, columns=[f"feature_{i}" for i in range(X_train_processed.shape[1])])
        train_processed[target_column] = y_train.reset_index(drop=True)
        train_processed.to_csv(train_save_path, index=False)
        logging.info(f"Processed training data saved to {train_save_path}")
        
        test_processed = pd.DataFrame(X_test_processed, columns=[f"feature_{i}" for i in range(X_test_processed.shape[1])])
        test_processed[target_column] = y_test.reset_index(drop=True)
        test_processed.to_csv(test_save_path, index=False)
        logging.info(f"Processed test data saved to {test_save_path}")
        
        joblib.dump(preprocessor, preprocessor_save_path)
        logging.info(f"Preprocessor saved to {preprocessor_save_path}")

    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data, split into training and test sets, and save the processed data and preprocessor.")
    parser.add_argument("filepath", type=str, help="Path to the raw data CSV file.")
    parser.add_argument("--target_column", type=str, default="target", help="Name of the target column.")
    parser.add_argument("--train_save_path", type=str, default="train_processed.csv", help="Path to save the processed training data.")
    parser.add_argument("--test_save_path", type=str, default="test_processed.csv", help="Path to save the processed test data.")
    parser.add_argument("--preprocessor_save_path", type=str, default="preprocessor.pkl", help="Path to save the preprocessor.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument("--missing_value_remover_threshold", type=int, default=42, help="Fractional threshold for column removal. Columns with missing values below this fraction are removed.")

    args = parser.parse_args()
    
    preprocess_and_save_data(args.filepath, args.target_column, args.train_path, args.test_path, args.preprocessor_path, args.test_size, args.random_state, args.missing_value_remover_threshold)
