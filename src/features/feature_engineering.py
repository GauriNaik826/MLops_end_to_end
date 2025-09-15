
from pathlib import Path
import sys
# repo_root = .../MLops_end_to_end
repo_root = Path(__file__).resolve().parents[2]  # adjust if your depth differs
sys.path.insert(0, str(repo_root))
# feature engineering
import numpy as np
import pandas as pd
import os, re
import nltk, string
# turns text into bag-of-words features
from sklearn.feature_extraction.text import CountVectorizer
import yaml
from src.logger import logging
# to save/load Python objects (e.g., the fitted vectorizer).
import pickle

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            logging.debug('Parameters retrieved from %s', params_path)
            return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    # Bad YAML syntax/content,
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    # Any other unexpected issue.
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        # replaces missing values with empty strings in place.
        # Why: text columns often have NaNs; downstream tokenizers/vectorizers expect strings. Empty string is a safe, 
        # neutral placeholder for text features.
        # (If you have numeric columns, you’d usually impute differently; this line applies to all columns, so be mindful.)
        df.fillna('', inplace=True)
        # high-level log for pipeline milestones (useful in runs).
        logging.info('Data loaded and NaNs filled from %s', file_path)
        return df
    # Specific CSV parse errors (bad delimiter/quotes/etc.) are logged, then re-raised.
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    # Catch-all for anything else, also logged and re-raised.
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

# Defines a helper that will featurize text with a Bag-of-Words (BoW) representation and return two DataFrames (train & test).
def apply_bow(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """Apply Count Vectorizer to the data."""
    try:
        logging.info("Applying BOW...")
        # Builds a CountVectorizer that will keep only the max_features most frequent tokens.
        vectorizer = CountVectorizer(max_features=max_features)

        # X is our feature and y is our target. X is our feature which we vectorize 
        # Splits features (text in review) and labels (sentiment) for both splits.
        X_train = train_data["review"].values
        y_train = train_data["sentiment"].values
        X_test  = test_data["review"].values
        y_test  = test_data["sentiment"].values

        # Learns the vocabulary on the training text and converts it to a sparse BoW matrix.
        X_train_bow = vectorizer.fit_transform(X_train)
        # (Use fit on train only to avoid data leakage.)
        # Converts test text using the already learned train vocabulary (no fitting here).
        X_test_bow  = vectorizer.transform(X_test)

        # Turns the (sparse) train matrix into a dense array and wraps it in a DataFrame, then appends the target as a label column.
        # Note: toarray() can be memory-heavy for large data; keeping it sparse is often better.
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df["label"] = y_train

        # Same for the test split.
        test_df  = pd.DataFrame(X_test_bow.toarray())
        test_df["label"] = y_test

        # Ensure the 'models/' directory exists
        os.makedirs('models', exist_ok=True)

        #Saves the fitted vectorizer to disk so inference (and any future runs) use the exact same vocabulary/token mapping.  
        pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))

        # Logs completion and returns the two DataFrames.
        logging.info('Bag of Words applied and data transformed')
        return train_df, test_df

    except Exception as e:
        logging.error('Error during Bag of Words transformation: %s', e)
        raise

# Defines a function save_data that takes a Pandas DataFrame (df) and a destination path (file_path).
def save_data(df: pd.DataFrame, file_path: str)-> None:
    
    try: 
        # os.path.dirname(file_path) extracts the folder part (e.g., 'data/processed' from 'data/processed/train.csv').
        dirpath =  os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # exist_ok=True avoids an error if the directory already exists.
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
            
        df.to_csv(file_path, index=False)
        # Writes the DataFrame to CSV at file_path.
        # index=False prevents Pandas from adding the DataFrame’s index as an extra column.
        logging.info("Data saved to %s",file_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try: 
        # Reads configuration from params.yaml (a single source of truth for knobs like max_features, paths, etc.).
        params = load_params('params.yaml')
        # Pulls the CountVectorizer limit (how many top tokens to keep) from the YAML.
        max_features = params['feature_engineering']['max_features']

        # max_features = 20
        # Loads the preprocessed splits (produced by your preprocessing step) from data/interim/… into DataFrames.
        # At this stage, text is cleaned (lowercased, no URLs, etc.) but not vectorized yet.
        train_data = load_data('./data/interim/train_processed.csv')
        test_data  = load_data('./data/interim/test_processed.csv')

        #Applies Bag-of-Words (CountVectorizer) to both splits:
        # Fits on train_data['review'] and transforms train/test.
        # Returns two DataFrames where columns are token counts and there’s a label column with 0/1.

        train_df, test_df = apply_bow(train_data, test_data, max_features)

        # Saves the engineered features to data/processed/… using a safe, cross-platform path join.
        save_data(train_df, os.path.join("./data", "processed", "train_bow.csv"))
        save_data(test_df,  os.path.join("./data", "processed", "test_bow.csv"))

        # These are the artifacts the model-training step will consume.

    except Exception as e:
        # Any exception bubbles here, gets logged as ERROR (for pipelines/alerts) and printed (helpful while running locally).
        logging.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()









    






