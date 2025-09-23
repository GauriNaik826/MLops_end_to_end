# 🚀 End-to-End MLops Project

## 📌 Overview

This project is an **end-to-end MLops pipeline** for sentiment analysis that takes a machine learning model from **data ingestion → preprocessing → training → evaluation → tracking → deployment**.
It integrates **MLflow, DVC, S3, Docker, Flask, GitHub Actions (CI/CD)** and experiment tracking with **DagsHub**.

---

## 🏗️ Architecture

```
                ┌─────────────┐
                │   Raw Data  │
                └──────┬──────┘
                       │
                 Data Ingestion
                       │
                 Preprocessing
                       │
                  Model Training
                       │
         ┌─────────────┴─────────────┐
         │                           │
   MLflow Tracking             DVC + S3 Storage
         │                           │
   Metrics, Params              Data & Model Versioning
         │
   Model Registry (DagsHub MLflow)
         │
    Flask App (Serving)
         │
   Dockerized Application → CI/CD → (AWS/EKS deployment)
```

---

## ⚙️ Features Implemented

✅ **Experiment Tracking** with MLflow on **DagsHub**
✅ **Pipeline Automation** with DVC
✅ **Data Versioning** using DVC + AWS S3
✅ **Unit Tests** for both ML pipeline and Flask API
✅ **CI/CD** with GitHub Actions (automated pipeline + tests + model promotion)
✅ **Containerization** of Flask App using Docker
✅ **Environment Management** with `.env` file and GitHub Secrets
✅ **Monitoring** setup using Prometheus & Grafana (to be expanded for cloud deployment)

---

## 📂 Project Structure

```
MLops_end_to_end/
│── data/                # Raw & processed data (DVC tracked)
│── models/              # Saved models & vectorizers
│── notebooks/           # Jupyter experiments
│── reports/             # Metrics, experiment info
│── src/
│   ├── data_pipeline/   # Ingestion & preprocessing scripts
│   ├── model/           # Training, evaluation, registry
│   ├── logger/          # Logging utilities
│── flask_app/           # Flask web app
│── tests/               # Unit tests
│── .dvc/                # DVC cache & metadata
│── .github/workflows/   # GitHub Actions pipeline
│── requirements.txt     # Optimized dependencies
│── Dockerfile           # Docker configuration
│── dvc.yaml             # DVC pipeline definition
│── .env.example         # Example env file
│── README.md            # Project documentation
```

---

## 🚀 Getting Started

### 1️⃣ Clone Repository

```bash
git clone https://github.com/<your-username>/MLops_end_to_end.git
cd MLops_end_to_end
```

### 2️⃣ Setup Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Create a `.env` file:

```
CAPSTONE_TEST=your_dagshub_token
```

### 3️⃣ Run DVC Pipeline

```bash
dvc repro
```

### 4️⃣ Run Tests

```bash
pytest tests/
```

### 5️⃣ Run Flask App

```bash
python flask_app/app.py
```

or with Docker:

```bash
docker build -t capstone-app:latest .
docker run --env-file .env -p 8888:3000 capstone-app:latest
```

---

## 🔗 CI/CD with GitHub Actions

* Workflow runs automatically on every push.
* Steps:

  1. Install dependencies
  2. Run DVC pipeline
  3. Run tests
  4. Log metrics & artifacts to DagsHub MLflow
  5. Promote best model to **Production**

---

## 📊 Monitoring

* **Prometheus**: Collects request counts, latencies, prediction distribution.
* **Grafana**: Dashboards to visualize model performance in production.
  (Currently configured for local setup, AWS deployment planned later).

---

## 🛠️ Tech Stack

* **MLflow** → Experiment tracking & model registry
* **DVC** → Pipeline automation & data versioning
* **AWS S3** → Remote storage for DVC
* **DagsHub** → Remote MLflow tracking
* **Flask** → Model serving API
* **Docker** → Containerization
* **GitHub Actions** → CI/CD automation
* **Prometheus + Grafana** → Monitoring

