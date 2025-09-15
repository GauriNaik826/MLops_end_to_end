import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import yaml
from pathlib import Path
import sys
import os
# repo_root = .../MLops_end_to_end
repo_root = Path(__file__).resolve().parents[2]  # adjust if your depth differs
sys.path.insert(0, str(repo_root))
from src.logger import logging

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train the Logistic Regression model."""
    try:
        # Create a scikit-learn logistic regression classifier.
        # C=1: inverse of regularization strength (higher C = less regularization). Here it’s the default strength.
        # solver='liblinear': optimization algorithm suited for small/medium datasets; supports L1/L2 penalties; good for binary problems.
        # penalty='l2': use L2 (ridge) regularization to prevent overfitting.
        clf = LogisticRegression(C=1, solver='liblinear', penalty='l2')
        # Train the model: estimate coefficients using the provided training data.
        clf.fit(X_train, y_train)
        # Write an INFO log entry indicating training finished successfully.
        logging.info('Model training completed')
        return clf
    except Exception as e:
        logging.error('Error during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    # Type hints say file_path is a string and the function returns nothing (None).
    # Starts a try block to catch I/O errors.
    try:
        # Opens the target file in binary write mode ('wb').
        with open(file_path, 'wb') as file:
            # Serializes the model object to disk using pickle.dump.
            pickle.dump(model, file)
        # Logs an INFO message indicating where the model was saved.
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        # If anything goes wrong (e.g., folder doesn’t exist, permissions), 
        # log an ERROR and re-raise the exception so the caller knows the save failed.
        logging.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        # Loads the training dataset that was produced by feature engineering (Bag-of-Words).
        train_data = load_data('./data/processed/train_bow.csv')
        # load_data (defined elsewhere in your module) reads the CSV into a pandas DataFrame.
        # X_train: all columns except the last (BoW features) → converted to a NumPy array with .values.
        X_train = train_data.iloc[:, :-1].values
        # y_train: the last column (the label, typically 0/1) → also as a NumPy array.
        y_train = train_data.iloc[:, -1].values
        # Trains the classifier by calling train_model (defined above in the file).
        #  This initializes LogisticRegression with tuned hyperparameters, fits it on X_train, y_train, and returns the fitted estimator as clf.
        clf = train_model(X_train, y_train)

        os.makedirs("models",exist_ok=True)
        # Persists the trained model to models/model.pkl using the helper defined earlier.
        save_model(clf, 'models/model.pkl')
    except Exception as e:
        # If any step in main() fails (loading data, training, saving), log an ERROR and print a short message.
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

