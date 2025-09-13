# Loads the AWS SDK for Python. You’ll use it to talk to S3.
import boto3
import pandas as pd
# imports the standard logging module.
import logging
from src.logger import logging
# Lets you wrap a string as a file-like object. pd.read_csv can then read the S3 object as if it were a local file.
from io import StringIO


class s3_operations:
    def __init__(self, bucket_name, aws_access_key, aws_secret_key, region_name="us-east-1"):
        # Saves the target bucket_name
        self.bucket_name = bucket_name
        # Creates a low-level S3 client using the provided credentials and region.
        # this basically connects s3 bucket creates a cooonection with our code
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name
        )
        # Logs that the S3 ingestion utility is ready.
        logging.info("Data Ingestion from S3 bucket initialized")
        # In production, prefer AWS’s default credential chain (env vars / profiles / IAM roles) instead of hard-coding keys.

        
    # give it an S3 key (path inside the bucket), get back a DataFrame.    
    # before we use this fucntio our S3 bucket must have the data
    def fetch_file_from_s3(self, file_key):
        """
        Fetches a CSV file from the S3 bucket and returns it as a Pandas DataFrame.
        :param file_key: S3 key/path, e.g. "data/data.csv"
        :return: pandas.DataFrame
        """

        try:
            # Logs what it’s about to fetch.
            logging.info(f"Fetching file '{file_key}' from S3 bucket '{self.bucket_name}'...")
            # Calls GetObject to download the object (metadata + body stream).
            # this method id default s3 clinets object
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            # Reads the response body (bytes), decodes bytes → str with UTF-8, wraps it in StringIO (file-like), 
            # and hands it to pandas.read_csv to parse into a DataFrame.
            df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))
            logging.info(f"Successfully fetched and loaded '{file_key}' from S3 with {len(df)} records.")
            return df
        except Exception as e:
            # On any error, logs and re-raises so callers can handle it.
            logging.error("Error fetching file from S3: %s", e)
            raise




        

    
