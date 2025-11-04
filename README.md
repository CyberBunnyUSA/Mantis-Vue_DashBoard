# Mantis-Vue_DashBoard
WTO ETL + ML + FastAPI Dashboard
================================

Overview
--------
This project contains an automated pipeline that:
- Extracts WTO timeseries data
- Transforms/cleans the data
- Runs EDA and generates reports (CSV + PNG)
- Fits ML models (regression, classification, clustering)
- Produces a buy/hold/sell predictive signal
- Serves a small FastAPI dashboard for integration with Mantis Vue Dashboard

Files of interest
- main_pipeline.py         # Full ETL → EDA → ML pipeline and FastAPI app (example)
- backend/requirements.txt # Python dependencies
- Dockerfile               # Container configuration
- reports/                 # Generated CSVs, images, ML outputs (created at runtime)

Requirements
------------
- Docker (optional)
- Python 3.11 (if running locally)
- The required Python packages in backend/requirements.txt

Environment
-----------
No secret is required for the included WTO example endpoint, but if you change APIs (Alpha Vantage, BEA, etc.) add your keys as environment variables and update the pipeline to read them.

Quickstart — Local (venv)
-------------------------
1. Create and activate a venv:
   python3.11 -m venv venv
   source venv/bin/activate

2. Install dependencies:
   pip install -r backend/requirements.txt

3. Run the pipeline & API (development):
   python main_pipeline.py
   - The script runs the pipeline and launches the FastAPI app on port 8000 by default.

Quickstart — Docker
-------------------
Build image:
  docker build -t wto-dashboard:latest .

Run container:
  docker run --rm -p 8000:8000 -v $(pwd)/reports:/app/reports wto-dashboard:latest

API endpoints (examples)
------------------------
- GET /                       — Simple HTML page with links
- GET /run_pipeline           — Run full pipeline (extract → transform → EDA → ML → prediction)
- GET /latest_report          — List files under the reports/ folder

Notes
-----
- The pipeline writes outputs to ./reports/ by default. Mount or persist that directory when running in containers if you want to keep outputs.
- For production deployments, replace uvicorn CMD with a process manager (gunicorn + uvicorn workers) and tune worker count, timeouts, logging, and resource limits.
- Pin or update package versions as desired. The pinned versions are chosen to be reasonably recent and compatible as of mid-2024.

Contributing / Extending
------------------------
- Swap WTO extraction with any other data source — keep transforms and EDA intact, and ensure the data returns as a DataFrame.
- Add CORS rules and authentication if you intend to expose the API to a public frontend.
