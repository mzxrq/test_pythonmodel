# main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
import os

app = FastAPI(title="Stock Fraud Detection API")

# --- Allow frontend access ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Paths for market models ---
MODEL_PATHS = {
    "US": "US_model.pkl",
    "JP": "JP_model.pkl",
    "TH": "TH_model.pkl"
}

# --- Example symbols for training per market ---
MARKET_SYMBOLS = {
    "US": ["AAPL", "MSFT"],
    "JP": ["7203.T", "9020.T"],
    "TH": ["CPALL.BK", "PTT.BK"]
}

# --- Helper: Train IsolationForest for given symbols ---
def train_model(symbols, model_path):
    dfs = []

    for symbol in symbols:
        # Auto-adjust interval & period by market
        if symbol.endswith(".T") or symbol.endswith(".BK"):
            interval = "1d"
            period = "1y"
        else:
            interval = "1h"
            period = "30d"

        df = yf.download(symbol, interval=interval, period=period)
        if df.empty:
            print(f"⚠️ No data for {symbol} with interval={interval} and period={period}")
            continue

        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join([str(c) for c in col if c]) for col in df.columns.values]
        else:
            df.columns = df.columns.map(str)

        close_col = next((c for c in df.columns if "Close" in c), None)
        volume_col = next((c for c in df.columns if "Volume" in c), None)
        if not close_col or not volume_col:
            print(f"⚠️ Missing Close or Volume for {symbol}")
            continue

        df["return"] = df[close_col].pct_change()
        rolling_window = min(60, max(5, len(df)//2))
        df["volume_z"] = (df[volume_col] - df[volume_col].rolling(rolling_window).mean()) / df[volume_col].rolling(rolling_window).std()
        df = df.dropna()

        features = df[["return", "volume_z"]].replace([np.inf, -np.inf], np.nan).dropna()
        if not features.empty:
            dfs.append(features)

    if not dfs:
        raise ValueError("No valid data to train model!")

    train_features = pd.concat(dfs)
    model = IsolationForest(contamination=0.02, random_state=42)
    model.fit(train_features)
    joblib.dump(model, model_path)
    print(f"✅ Model trained and saved to {model_path}")
    return model

# --- Load or train models per market ---
market_models = {}
for market, symbols in MARKET_SYMBOLS.items():
    if os.path.exists(MODEL_PATHS[market]):
        market_models[market] = joblib.load(MODEL_PATHS[market])
        print(f"✅ {market} model loaded")
    else:
        market_models[market] = train_model(symbols, model_path=MODEL_PATHS[market])
        print(f"✅ {market} model trained")

# --- Determine market by symbol ---
def get_market(symbol: str):
    if symbol.endswith(".T"):
        return "JP"
    elif symbol.endswith(".BK"):
        return "TH"
    else:
        return "US"

# --- API Root ---
@app.get("/")
def home():
    return {"message": "Welcome to the Fraud Detection API 🚀", "usage": "/detect?symbol=AAPL"}

# --- Fraud detection endpoint ---
@app.get("/detect")
def detect(symbol: str = Query(...)):
    try:
        market = get_market(symbol)
        model = market_models[market]

        # Interval & period per market
        interval = "1d" if market in ["JP", "TH"] else "1h"
        period = "1y" if market in ["JP", "TH"] else "30d"

        df = yf.download(symbol, interval=interval, period=period)
        if df.empty:
            return {"error": f"No data for {symbol}"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join([str(c) for c in col if c]) for col in df.columns.values]
        else:
            df.columns = df.columns.map(str)

        close_col = next((c for c in df.columns if "Close" in c), None)
        volume_col = next((c for c in df.columns if "Volume" in c), None)
        if not close_col or not volume_col:
            return {"error": f"Missing Close or Volume columns"}

        df["return"] = df[close_col].pct_change()
        rolling_window = min(60, max(5, len(df)//2))
        df["volume_z"] = (df[volume_col] - df[volume_col].rolling(rolling_window).mean()) / df[volume_col].rolling(rolling_window).std()
        df = df.dropna()

        features = df[["return", "volume_z"]].replace([np.inf, -np.inf], np.nan).dropna()
        if features.empty:
            return {"error": "Not enough valid features"}

        # Predictions
        preds = model.predict(features)
        scores = model.decision_function(features)
        # Normalize score to 0-1 for frontend transparency (higher = more suspicious)
        norm_scores = 1 - ((scores - scores.min()) / (scores.max() - scores.min() + 1e-9))

        df = df.iloc[-len(features):].copy()
        df["anomaly"] = preds == -1
        df["anomaly_score"] = norm_scores

        anomalies = df[df["anomaly"]].tail(10).copy()
        anomalies.reset_index(inplace=True)
        anomalies_json = [
            {str(k): (float(v) if isinstance(v, (np.floating, np.integer)) else str(v)) for k, v in row.items()}
            for _, row in anomalies.iterrows()
        ]

        return {
            "symbol": symbol,
            "market": market,
            "interval": interval,
            "total_records": int(len(df)),
            "anomaly_count": int(df["anomaly"].sum()),
            "recent_anomalies": anomalies_json
        }

    except Exception as e:
        return {"error": str(e)}
