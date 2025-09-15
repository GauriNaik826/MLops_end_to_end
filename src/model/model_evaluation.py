from pathlib import Path
import sys
# repo_root = .../MLops_end_to_end
repo_root = Path(__file__).resolve().parents[2]  # adjust if your depth differs
sys.path.insert(0, str(repo_root))
import numpy as np
import pandas as pd
# pickle to serialize/deserialize Python objects (e.g., a trained model)
import pickle
# json if you want to read/write JSON config or results.
import json
# Import the evaluation metrics you’ll compute for predictions.
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os
from src.logger import logging
from mlflow.models import infer_signature
from tempfile import TemporaryDirectory
from mlflow.models import infer_signature
import scipy
# Below code block is for p
# roduction use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")
# Read a personal access token (PAT) for DagsHub from CAPSTONE_TEST. Fail fast if it’s missing
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# MLflow uses basic auth when talking to a remote server. On DagsHub, both username & password are set to the token.
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Build the tracking URI that points MLflow at your repo’s tracking server on DagsHub: https://dagshub.com/<owner>/<repo>.mlflow.
dagshub_url = "https://dagshub.com"
repo_owner = "GauriNaik826"
repo_name = "MLops_end_to_end"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

# Below code block is for local use
# # -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/GauriNaik826/MLops_end_to_end.mlflow')
# dagshub.init(repo_owner='GauriNaik826', repo_name='MLops_end_to_end', mlflow=True)
# -------------------------------------------------------------------------------------

# loads the model we had trained and saved.
# Define a function that takes a path to a file (string). No return type hint given, because it could be any Python object (your model).
def load_model(file_path: str):
    """Load the trained model from a file."""
    # Start a try block so we can log and re-raise meaningful errors.
    try:
        # Open the file at file_path in binary read mode ('rb'). Use a context manager to auto-close it.
        with open(file_path, 'rb') as file:
            # Deserialize a Python object from the opened file using pickle. This reconstructs the trained model object.
            model = pickle.load(file)
        # Log a high-level info message that the model was loaded, including the path.
        logging.info('Model loaded from %s', file_path)
        # Return the loaded model to the caller.
        return model
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model: %s', e)
        raise

# loads the x test and y test we had saved 
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

# takes the unseen reviews and gives it to the model and comapres 
# the predicted labels with actual labels
def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        # Starts a try block to catch and log any runtime errors cleanly.
        # Uses the model to produce class predictions (0/1) for the test set.
        y_pred = clf.predict(X_test)
        # Gets predicted probabilities for the positive class (column 1).
        # Requires that clf implements predict_proba and that you’re doing binary classification with labels {0,1}.
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        # Computes accuracy = fraction of correct predictions.
        accuracy = accuracy_score(y_test, y_pred)
        # Computes precision for the positive class (by default pos_label=1) = TP / (TP + FP).
        precision = precision_score(y_test, y_pred)
        # Computes recall (a.k.a. sensitivity) for the positive class = TP / (TP + FN).
        recall = recall_score(y_test, y_pred)
        # Computes ROC AUC using the positive-class probabilities; 
        # measures ranking quality across all thresholds (0.5 threshold not fixed).
        # computes the AUC (Area Under the ROC Curve)
        auc = roc_auc_score(y_test, y_pred_proba)
        # Collects the four metrics into a dictionary.
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logging.info('Model evaluation metrics calculated')
        # returns the object dictionary of the metrics 
        return metrics_dict
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise
     
def main():
    # Tell ML flow to group all the runs from this script under experiment name specified 
    mlflow.set_experiment("my-dvc-pipeline")
    # Begin an MLflow tracking run (a single execution). 
    # The context manager auto-ends the run. run holds metadata like the run_id.
    with mlflow.start_run() as run:  # Start an MLflow run
        try:
            # Load your trained model (pickled) from disk into memory as clf.
            clf = load_model('./models/model.pkl')
            # Read the processed test dataset (after BOW features) into a pandas DataFrame.
            test_data = load_data('./data/processed/test_bow.csv')
            # X_test: all columns except the last (your BOW features).
            # y_test: the last column (target/class).
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values
            # Call your helper that runs the model on X_test, compares to y_test, and returns a dict of 
            # metrics the metrics dictinary is returned.
            metrics = evaluate_model(clf, X_test, y_test)
            # save those metrics locally as a JSON file for later inspection/versioning.
            save_metrics(metrics, 'reports/metrics.json')
     
            # Record each metric in the current MLflow run so they appear in the MLflow UI and can be compared across runs.
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # If the model follows the scikit-learn API, fetch its hyperparameters and 
            # log them to MLflow (e.g., C, solver, penalty). Parameters help explain/compare runs.
            # Log model parameters to MLflow
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            # Upload the trained model artifact to the run, under the artifact path model. 
            # MLflow also captures the flavor (sklearn) for later loading/serving.
            # Log model to MLflow pip install "mlflow==2.12.1"
            mlflow.sklearn.log_model(clf, "model")

            # Save model locally and upload as artifacts works on DagsHub
            # example = X_test[:5].toarray() if scipy.sparse.issparse(X_test) else X_test[:5]
            # signature = infer_signature(example, clf.predict(example))
            # TemporaryDirectory() creates a temporary folder on your local filesystem.
            # As soon as the with TemporaryDirectory() as tmpdir: block ends, Python automatically deletes the folder and everything inside it.
            # uploads the contents to MLflow’s artifact store (DagsHub in your case) before deletion.
            # with TemporaryDirectory() as tmpdir:
            #     local_dir = f"{tmpdir}/sk_model"
            #     mlflow.sklearn.save_model(
            #        clf,
            #        local_dir,
            #        input_example=example,
            #        signature=signature,
            #    )
            #    # This creates an artifacts/model/… tree in the run
            # mlflow.log_artifacts(local_dir, artifact_path="model")

            # local_dir = "./_tmp_model_artifacts/sk_model_eval"
            # os.makedirs(local_dir, exist_ok=True)
            # mlflow.sklearn.save_model(clf, local_dir, input_example=example, signature=signature)

            # mlflow.log_artifacts(local_dir, artifact_path="model")
            
            # Write a small JSON (your helper) with metadata about this run: 
            # the run_id and where the model artifact lives. Useful for downstream automation.
            # Save model info
            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            
            # Attach the metrics.json file as an artifact to the run, 
            # so it’s stored alongside the model and visible/downloadable in MLflow UI.
            # Log the metrics file to MLflow
            mlflow.log_artifact('reports/metrics.json')

        # If anything inside the run fails, log the error (for logs) and print it (for console visibility). 
        # The run will still be ended by the context manager.
        except Exception as e:
            logging.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")

if __name__ == '__main__':
    main()