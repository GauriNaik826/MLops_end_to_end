import numpy as np 
import pandas as pd
from pathlib import Path
import sys
# repo_root = .../MLops_end_to_end
repo_root = Path(__file__).resolve().parents[2]  # adjust if your depth differs
sys.path.insert(0, str(repo_root))
import os 
import re
import nltk
import string
# English stop-word list (e.g., the, is, and).
from nltk.corpus import stopwords
# Lemmatizer that turns words into their base form (e.g., running → run).
from nltk.stem import WordNetLemmatizer
from src.logger import logging
nltk.download('wordnet')
nltk.download('stopwords')

# Defines a function that takes a Pandas DataFrame and the name of the text column (default 'text')
def preprocess_dataframe(df, col='text'):
    """Preprocess a DataFrame by applying text preprocessing to a specific column."""
    # Instantiate a lemmatizer once.
    lemmatizer = WordNetLemmatizer()
    # Build a set of stop words for fast O(1) membership checks.
    stop_words = set(stopwords.words("english"))

    def preprocess_text(text):
        """Helper function to preprocess a single text string."""

        # Deletes links like https://… or www.….
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Strips digits to reduce noise.
        text = ''.join(ch for ch in text if not ch.isdigit())
        # make text lower case 
        text = text.lower()
        # Replaces punctuation with spaces and collapses multiple spaces.
        text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        # Tokenize → remove stop words → lemmatize
        tokens = [w for w in text.split() if w not in stop_words]
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
        
        return ' '.join(tokens)

    df[col] = df[col].apply(preprocess_text)

    # Remove small sentences less than 3 words by marking them Nan
    # df[col] = df[col].apply(lambda x: np.nan if len(str(x).split()) < 3 else x)

    # Converts the target column to string (defensive), applies preprocess_text row-wise, 
    # logs completion, and returns the cleaned DataFrame.
    df = df.dropna(subset=[col])
    logging.info("Data preprocessing completed")
    return df



    
def main():
    try:
        # Reads the split CSVs produced by data_ingestion step (data/raw/), logs success.
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data  = pd.read_csv('./data/raw/test.csv')
        logging.info('Data loaded properly')
        # Cleans the review column in both train and test DataFrames using the function above.
        train_processed_data = preprocess_dataframe(train_data, 'review')
        test_processed_data  = preprocess_dataframe(test_data,  'review')

        # Ensures there’s a data/interim/ folder (standard “intermediate” layer in the data pipeline).
        data_path = os.path.join('./data', 'interim')
        os.makedirs(data_path, exist_ok=True)

        # Saves cleaned datasets to the interim layer (so downstream steps can consume them), logs the path.
        train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
        test_processed_data.to_csv( os.path.join(data_path, 'test_processed.csv'),  index=False)
        logging.info('Processed data saved to %s', data_path)


    except Exception as e:
        logging.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")




if __name__ == "__main__":
    main()






