# register model
from pathlib import Path
import sys
# repo_root = .../MLops_end_to_end
repo_root = Path(__file__).resolve().parents[2]  # adjust if your depth differs
sys.path.insert(0, str(repo_root))
import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()
# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "GauriNaik826"
repo_name = "MLops_end_to_end"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/GauriNaik826/MLops_end_to_end.mlflow')
# dagshub.init(repo_owner='GauriNaik826', repo_name='MLops_end_to_end', mlflow=True)
# -------------------------------------------------------------------------------------


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

# model_name → the registry name you want (e.g., "my_model").
# model_info → a dict you saved earlier in reports folder (typically contains the run_id and the relative model_path you logged).
def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        # Builds a model URI that tells MLflow where the model artifacts live.
        # Format runs:/<run_id>/<artifact_subpath> means “take the artifacts logged under this run, at this path”.
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        print("model_uri", model_uri)
        # Calls MLflow’s high-level API to register that artifact as a new version under model_name.
        # Returns a ModelVersion object (has fields like .version, .status, etc.).
        # so it gets different model versions we get after the reruns 
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Creates a lower-level MLflow client for registry operations not covered by the high-level function.
        # so eveytime we run the entire pipeline we get a new model everytime and we have decided to keep it in staging phase always
        # the best model we get from the staging area we send it to production. the test for these models in staging state to find the best model for 
        # production u see later
        # Transition the model to "Staging" stage 
        client = mlflow.tracking.MlflowClient()
        # Immediately moves the just-registered model version into the “Staging” stage (typical lifecycle: None → Staging → Production → Archived).
        # This is useful for CI/CD flows: new versions automatically land in Staging for testing.
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        # Writes a debug log confirming the version number and the stage transition.
        logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise

def main():
    try:
        # experiment_info.json consists of the model run_id and model_path.
        model_info_path = 'reports/experiment_info.json'
        # model_info → a dict you saved earlier (typically contains the run_id and the relative model_path you logged).
        # load_model_info should read that JSON and return a dict like: {"run_id": "<mlflow-run-id>", "model_path": "model"} 
        model_info = load_model_info(model_info_path)
        # model_name → the registry name you want (e.g., "my_model").
        # all the reruns we do, we get a new model, we can train_test split, we can change the max features 20->50,  
        model_name = "my_model"
        # Picks a registry name and calls the function above to register and stage it.
        register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main() 
