# backend/app.py
import os, json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent
DATA_DIR = (ROOT / "data" / "dashboard_output")  # point to your ETL output
CHARTS_DIR = DATA_DIR / "charts"

app = FastAPI(title="Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve static chart HTML
if CHARTS_DIR.exists():
    app.mount("/static/charts", StaticFiles(directory=str(CHARTS_DIR)), name="charts")

def load_json_safe(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

@app.get("/api/kpis")
def get_kpis():
    # expect a kpis.json created by ETL
    p = DATA_DIR / "kpis.json"
    data = load_json_safe(p)
    if data is None:
        raise HTTPException(status_code=404, detail="KPIs not found")
    return data

@app.get("/api/sector/{sector}")
def get_sector(sector: str):
    p = DATA_DIR / f"sector_{sector}.json"
    data = load_json_safe(p)
    if data is None:
        raise HTTPException(status_code=404, detail="Sector data not found")
    return data

@app.get("/api/company/{ticker}")
def get_company(ticker: str):
    p = DATA_DIR / f"company_{ticker.upper()}.json"
    data = load_json_safe(p)
    if data is None:
        raise HTTPException(status_code=404, detail="Company not found")
    return data

@app.get("/api/chart/html/{name}")
def chart_html(name: str):
    p = CHARTS_DIR / f"{name}.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Chart not found")
    return p.read_text(encoding="utf-8")

# Optionally: return Vega-Lite spec JSON
@app.get("/api/chart/vega/{name}")
def chart_vega(name: str):
    p = DATA_DIR / "vega_specs" / f"{name}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Vega spec not found")
    return json.loads(p.read_text(encoding="utf-8"))

