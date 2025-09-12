# put at the top of src/data/data_ingestion.py (or a small runner script)
from pathlib import Path
import sys

# repo_root = .../MLops_end_to_end
repo_root = Path(__file__).resolve().parents[2]  # adjust if your depth differs
sys.path.insert(0, str(repo_root))
import numpy as np
import pandas as pd
# tells pandas not to silently “downcast” dtypes during certain ops in future versions; helpful to surface dtype changes early (prevents subtle bugs).
pd.set_option('future.no_silent_downcasting', True)
import os
from sklearn.model_selection import train_test_split
import yaml
import logging
from src.logger import logging
# helper to connect to S3 (or S3-compatible) storage; 
# load/save data from a remote bucket the same way wwe do locally.
# from src.connections import s3_connection

# Load parameters from a YAML file
def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        # opens the YAML file at params_path
        # uses yaml.safe_load to parse it into a Python dict
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            # logs a DEBUG message so you can trace which config was read
            logging.debug('Parameters retrieved from %s', params_path)
            # returns the dict of parameters
            return params
    # Robust error handling:
    except FileNotFoundError:
        # FileNotFoundError if the path is wrong.
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        # YAMLError if the YAML is malformed.
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        # Catch-all to log any other unexpected errors.
        logging.error('Unexpected error: %s', e)
        raise
    # All re-raise to fail fast (and let a higher layer decide what to do), but with clear logs.


# Load a CSV (local or via S3, depending on how you call it)
def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        # logs at INFO level (this is a major step)
        logging.info('Data loaded from %s', data_url)
        return df
    # Specific error handling for common CSV issues (bad format,missing file) plus a catch all 
    except pd.errors.ParserError as e:
        logging.error('CSV parse error: %s', e)
        raise
    except FileNotFoundError:
        logging.error('File not found: %s', data_url)
        raise
    except Exception as e:
        logging.error('Unexpected error while loading data: %s', e)
        raise

# Function that takes a DataFrame and returns a cleaned one.
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        # df.drop(columns=['tweet_id'], inplace=True)   # (example of optional cleanup)
        # Mark the start of the step.
        logging.info('pre-processing...')
        # Keep only rows where sentiment is positive or negative; discard anything else (e.g., “neutral”).
        final_df = df[df['sentiment'].isin(['positive', 'negative'])]
        # Convert labels to numeric targets (positive→1, negative→0).
        final_df['sentiment'] = final_df['sentiment'].replace({'positive': 1, 'negative': 0})
        # Mark success.
        logging.info('Data preprocessing completed')
        # Return the cleaned data.
        return final_df
    # KeyError: If required columns (like sentiment) are missing, log and re-raise.
    except KeyError as e:
        logging.error('Missing column in the dataframe: %s', e)
        raise
    # Exception: Catch any other unexpected problem, log and re-raise.
    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    # this save data in the data/raw/ coming from data_ingestion.py
    # you provide the train and test data and path where u want to save the data
    try:
        # Function takes train/test DataFrames and a base directory; returns nothing (-> None).
        raw_data_path = os.path.join(data_path, 'raw')
        # raw_data_path = ...: Build <data_path>/raw as the folder to store raw splits.
        os.makedirs(raw_data_path, exist_ok=True)
        # Write train/test CSV files without the DataFrame index.
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logging.debug('Train and test data saved to %s', raw_data_path)
    # except: If saving fails (e.g., permission/path issues), log as ERROR and re-raise.
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise


    
def main():
    try:
        # Reads configuration from a YAML file (e.g., test split size) using the helper you wrote earlier.
        # So you don’t hard-code knobs in code—easier to reproduce and change via config.
        # params = load_params(params_path="params.yaml")

        # Pulls test_size from that YAML (typical structure below).
        # test_size = params['data_ingestion']['test_size']

        # The commented line is a fallback/example if you want to hard-code 0.2 (20%) during quick tests.
        test_size = 0.2 

        # Downloads & reads the CSV into a DataFrame using your load_data() (which logs and handles parser errors).
        df = load_data(data_url='notebooks/data.csv')


        # instead of HTTP, fetch the file from S3 via your s3_connection helper. You’d initialize a client, then pull "data.csv" from the bucket.
         # s3 = s3_connection.s3_operations("bucket-name", "accesskey", "secretkey")
        # df = s3.fetch_file_from_s3("data.csv")

        # Cleans the raw data with your earlier function: keeps only rows with sentiment in {positive, negative}, converts to {1, 0}, etc.
        final_df = preprocess_data(df)

        # Splits the cleaned data into train/test according to test_size.
        # random_state=42 makes the split reproducible run-to-run.
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        # Writes train.csv and test.csv into ./data/raw/ (your save_data() creates the folder if needed and logs where it wrote them).
        save_data(train_data, test_data, data_path='./data')


    except Exception as e:
        # Any exception bubbles to here: it’s logged once at ERROR and also printed to the console for quick visibility.
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()


    
    






