"""
main_pipeline.py
Automated ETL → EDA → ML → FastAPI launch pipeline
Author: Whitney Moss
"""

# ==========================================================
# 🧩 1. Import Libraries
# ==========================================================
import os
import sys
import json
import time
import logging
import subprocess
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# ==========================================================
# 🧠 2. Configurations and Keys
# ==========================================================
API_KEYS = {
    "BEA": "C7AB3527-B334-4396-B1C9-837FAF4FB814",
    "ALPHA_VANTAGE": "2M31LG9FZQGZX2PJ"
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ==========================================================
# 🛠️ 3. Extract Phase – API Data Retrieval
# ==========================================================

def get_bea_data(table_name="T10105", frequency="A"):
    """Fetch data from BEA API"""
    logging.info("Fetching BEA data...")
    url = "https://apps.bea.gov/api/data"
    params = {
        "UserID": API_KEYS["BEA"],
        "method": "GetData",
        "datasetname": "NIPA",
        "TableName": table_name,
        "Frequency": frequency,
        "ResultFormat": "JSON"
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()["BEAAPI"]["Results"]["Data"]
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(DATA_DIR, "bea_data.csv"), index=False)
    logging.info(f"BEA data saved: {len(df)} records")
    return df

def get_alpha_vantage(symbol="MSFT", function="TIME_SERIES_DAILY"):
    """Fetch stock data from Alpha Vantage"""
    logging.info(f"Fetching Alpha Vantage data for {symbol}...")
    url = "https://www.alphavantage.co/query"
    params = {
        "function": function,
        "symbol": symbol,
        "apikey": API_KEYS["ALPHA_VANTAGE"],
        "datatype": "json"
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    key = list(data.keys())[1]
    df = pd.DataFrame(data[key]).T
    df = df.rename(columns=lambda x: x.split(". ")[-1])
    df.to_csv(os.path.join(DATA_DIR, f"{symbol}_stock.csv"))
    return df

# ==========================================================
# 🔍 4. Transform Phase – Data Cleaning and Type Detection
# ==========================================================

def transform_data(df):
    logging.info("Transforming data...")
    df = df.apply(pd.to_numeric, errors="ignore")
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)
    df.dropna(axis=1, how="all", inplace=True)
    return df

# ==========================================================
# 📊 5. EDA – Statistical Summaries and Visualization
# ==========================================================

def exploratory_data_analysis(df, label="EDA_Analysis"):
    logging.info("Running EDA...")
    desc = df.describe(include="all")
    desc.to_csv(os.path.join(DATA_DIR, f"{label}_summary.csv"))

    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, f"{label}_heatmap.png"))
    plt.close()

    # Z-scores for outlier detection
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number]), nan_policy='omit'))
    outliers = (z_scores > 3).sum().sum()
    logging.info(f"Detected {outliers} potential outliers")
    return desc

# ==========================================================
# 🤖 6. ML Modeling – Supervised and Unsupervised
# ==========================================================

def machine_learning_models(df):
    logging.info("Running machine learning models...")

    # Example: Linear Regression (Supervised)
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if len(numeric_df.columns) > 1:
        X = numeric_df.iloc[:, :-1]
        y = numeric_df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        logging.info(f"Linear Regression R2={r2:.3f}, MSE={mse:.3f}")

    # Example: KMeans Clustering (Unsupervised)
    if len(numeric_df.columns) >= 2:
        km = KMeans(n_clusters=3, random_state=42)
        km.fit(numeric_df)
        df["Cluster"] = km.labels_
        logging.info(f"KMeans clustering complete. Clusters assigned.")
    return df

# ==========================================================
# 🚀 7. Launch FastAPI Backend
# ==========================================================

def launch_backend():
    logging.info("Launching FastAPI backend...")
    server = subprocess.Popen(["uvicorn", "backend.app:app", "--reload", "--port", "8000"])
    print("\n✅ FastAPI running at: http://127.0.0.1:8000")
    print("Press CTRL+C to stop.")
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logging.info("Shutting down backend...")
        server.terminate()

# ==========================================================
# 🧭 8. Main Pipeline Execution
# ==========================================================

def main():
    logging.info("Starting full ETL → EDA → ML pipeline...")

    bea_df = get_bea_data()
    bea_df = transform_data(bea_df)
    exploratory_data_analysis(bea_df, label="BEA")

    stock_df = get_alpha_vantage("AAPL")
    stock_df = transform_data(stock_df)
    exploratory_data_analysis(stock_df, label="AAPL")

    ml_df = machine_learning_models(bea_df.select_dtypes(include=[np.number]))
    ml_df.to_csv(os.path.join(DATA_DIR, "bea_ml_results.csv"), index=False)

    launch_backend()

if __name__ == "__main__":
    main()

