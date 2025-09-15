from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
# we have added classes and modules relevant to prometheus because inside this app only we will expose the metrics whihc 
# prometheus will consume and we build the dashboard on Grafana 
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(text):
    # combined all the text processing stages in one main function 
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)

    return text

# Below code block is for local use
# -------------------------------------------------------------------------------------
mlflow.set_tracking_uri('https://dagshub.com/GauriNaik826/MLops_end_to_end.mlflow')
dagshub.init(repo_owner='GauriNaik826', repo_name='MLops_end_to_end', mlflow=True)
# -------------------------------------------------------------------------------------

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "vikashdas770"
# repo_name = "YT-Capstone-Project"
# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# # -------------------------------------------------------------------------------------


# Initialize Flask app
# Creates the Flask web app object. You’ll use app to define routes like / and /predict.
app = Flask(__name__)

# from prometheus_client import CollectorRegistry

# Create a custom registry
# Makes a separate Prometheus metrics registry so only your app’s metrics are exposed (not the default Python/runtime metrics).
# CollectorRegistry() -> a container that holds all the metrics you want to expose.
registry = CollectorRegistry()

# Define your custom metrics using this registry and plot it using a dashboard called grafana
# You’ll increment it on each request to track how many times each route is called by HTTP method.
# number of times u can hit the endpoint 
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
# plot histogram for latency. Say in one min we send 4 requests how much time did it take for the response like 10ms, 15ms etc
# You’ll observe the request duration (in seconds) for each call. Prometheus will bucket these values to show 
# the latency distribution (e.g., how many fell into 0.1s, 0.5s, 1s, etc).
#  these values are plot in a histogram
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)

# Increment it after each prediction to track how many times each class is predicted. Over time windows (like 1 minute in Grafana), 
# you can see class distribution rates.
# so PREDICTION_COUNT tells on average in one min how many postive reviews or negative reviews do we get 
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

# ------------------------------------------------------------------------------------------
# This is model serving -> from the registery gets the latest version. 
# Then loads the latest version. 

# Model and vectorizer setup
model_name = "my_model"
# In our MLflow model registry, it fetches the latest version of the model in the 'Staging' stage.
def get_latest_model_version(model_name):
    # Create a low-level MLflow client for registry operations (querying versions/stages, etc.).
    client = mlflow.MlflowClient()
    # Ask the registry for the latest version of my_model that’s in the Staging stage.
    latest_version = client.get_latest_versions(model_name, stages=["Staging"])
    # If nothing is in Staging (first run, perhaps), fall back to the latest version in stage None 
    # (i.e., registered but not staged yet).
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    # Return the version number (e.g., "3"). If the model doesn’t exist at all, return None.
    return latest_version[0].version if latest_version else None


# here only get the latest version like say 3
model_version = get_latest_model_version(model_name)
# Resolve the model to a registry URI like models:/my_model/3, then log where you’ll load it from.
model_uri = f'models:/{model_name}/{model_version}'
print(f"Fetching model from: {model_uri}")
# Load the model from the registry into memory as a PyFunc using the uri.
model = mlflow.pyfunc.load_model(model_uri)
# Load your saved text vectorizer (CountVectorizer/TF-IDF) so you can transform raw text the same way as during training.
vectorizer = pickle.load(open('../models/vectorizer.pkl', 'rb'))

# Routes
# We also have made a home route 
# This route renders the index.html
@app.route("/")
def home():
    # Flask route for the homepage. Increment the request counter (Prometheus) for a GET on /.
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    # Start a timer, render the page (empty result at first).
    start_time = time.time()
    response = render_template("index.html", result=None)
    # Record the latency histogram for this request, then return the page.
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    # Prediction endpoint. Count the POST request and start timing it.
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()
    # Read the raw text from the HTML form input named "text".
    text = request.form["text"]
    # Clean text
    text = normalize_text(text)
    # Convert to features
    # it has the 20 or 30 columns we had during training 
    features = vectorizer.transform([text])
    # form the dataframe for these 20 or 30 columns
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    # Predict
    result = model.predict(features_df)
    prediction = result[0]

    # Increment prediction count metric
    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

    # Measure latency
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    # we render the results in the index.html page our prediction results
    return render_template("index.html", result=prediction)

@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose only custom Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    # app.run(debug=True) # for local use
    app.run(debug=True, host="0.0.0.0", port=3000)  # Accessible from outside Docker