# PySpark Financial Anomaly Detection Pipeline

A production-grade, end-to-end data engineering and machine learning pipeline for detecting fraudulent financial transactions. This project processes over 5 million records using a Medallion Architecture (Bronze, Silver, Gold) powered by PySpark and Delta Lake.

## 🚀 Overview

The pipeline automates the entire lifecycle: from raw CSV ingestion and data quality validation to advanced behavioral feature engineering and ML model inference.

### Key Features
- **Medallion Architecture:** Clean separation of concerns using Bronze (raw), Silver (refined/split), and Gold (business metrics/predictions) layers.
- **Leakage-Free ML:** Chronological data splitting at the Silver layer to ensure the model is evaluated on future transactions.
- **Behavioral Features:** Advanced window-based calculations (velocity, moving averages, diversity) per account.
- **Data Quality:** Integrated **Great Expectations** validation gates.
- **ML Lifecycle:** Experiment tracking and model versioning via **MLflow**.
- **Scalable:** Built on **PySpark** and **Delta Lake** for handling millions of records.

---

## 🏗️ Architecture

1.  **Bronze Layer:** Ingests raw CSV with explicit schema enforcement and quarantines corrupt records.
2.  **Silver Layer:**
    *   Extracts temporal features (hour, day, weekend).
    *   Validates data quality (GX).
    *   Performs **Chronological Splitting** (Train before 2023-10-20, Test after).
3.  **Features Layer:** Computes account-level behavioral windows (1h velocity, 24h spend deviation, 6h merchant diversity).
4.  **Gold Layer:**
    *   Generates daily and hourly business aggregations.
    *   Stores final model predictions and confidence scores.

---

## 🛠️ Tech Stack
- **Engine:** PySpark (3.5.0)
- **Storage:** Delta Lake (3.2.0)
- **ML Tracking:** MLflow
- **Validation:** Great Expectations
- **Runtime:** OpenJDK 17 + Python 3.10
- **Environment:** Conda

---

## 📥 Getting Started

### 1. Prerequisites
- **Java:** OpenJDK 17 is required for PySpark.
- **Kaggle API:** Ensure `~/.kaggle/kaggle.json` is configured to download the dataset.

### 2. Environment Setup
```bash
# Create and activate environment
conda create -n financial-anomaly python=3.10 -y
conda activate financial-anomaly

# Install dependencies
pip install -r requirements.txt
pip install pyspark==3.5.0 delta-spark==3.2.0 mlflow great-expectations loguru kaggle
```

### 3. Data Ingestion
Download the dataset from Kaggle:
```bash
kaggle datasets download -d aryan208/financial-transactions-dataset-for-fraud-detection -p data/raw --unzip
```

---

## ⚙️ Execution

### The "One-Command" Run
To execute the entire pipeline (Ingest -> Silver -> Features -> Gold -> Predict):
```bash
python run_pipeline.py
```

### Individual Components
*   **Train Model:** `python src/models/train.py` (includes Cross-Validation & MLflow logging).
*   **View MLflow UI:** `mlflow ui` (access at http://localhost:5000).

---

## 📁 Directory Structure
```text
├── config/             # Spark & Delta configurations
├── data/               # Local Delta Lake storage (Git ignored)
│   ├── raw/            # Original CSV
│   ├── bronze/         # Enforced schema records
│   ├── silver/         # Refined & Chronologically split sets
│   └── gold/           # Aggregates & Predictions
├── notebooks/          # EDA and Visualization
├── src/
│   ├── data/           # ETL logic (Ingest, Silver, Gold)
│   ├── features/       # Window-based engineering
│   ├── models/         # Training and Inference scripts
│   └── utils/          # Data quality (GX) & helpers
├── run_pipeline.py     # Orchestrator script
└── pyproject.toml      # Dependency management
```

---

## 📊 Results
The pipeline outputs a final prediction table in `data/gold/fraud_predictions` containing:
- `transaction_id`
- `prediction` (0 or 1)
- `probability` (Model confidence score)
- `label` (Actual ground truth for validation)
