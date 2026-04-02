"""
Shared utilities for the stock market prediction project.

This module keeps the inference logic in one place so both the Streamlit app
and the optional Flask API use the same data preparation steps.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "stock_prediction_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = BASE_DIR / "feature_names.txt"
MODEL_INFO_PATH = BASE_DIR / "model_info.pkl"

DEFAULT_TRAIN_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "JPM",
    "V",
    "JNJ",
]
REQUIRED_PRICE_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def normalize_downloaded_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten Yahoo Finance columns into a predictable single-level schema."""
    cleaned = df.copy()

    if isinstance(cleaned.columns, pd.MultiIndex):
        cleaned.columns = [str(col[0]).strip() for col in cleaned.columns.to_flat_index()]
    else:
        cleaned.columns = [str(col).strip() for col in cleaned.columns]

    return cleaned.rename(columns={"Adj Close": "Adj_Close", "Date_": "Date"})


def download_stock_data(ticker: str, lookback_days: int = 365) -> pd.DataFrame:
    """Download historical price data for one ticker."""
    symbol = ticker.strip().upper()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    stock_data = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
    )

    if stock_data.empty:
        raise ValueError(f"Could not fetch data for ticker '{symbol}'.")

    stock_data = normalize_downloaded_columns(stock_data.reset_index())
    missing_columns = [col for col in REQUIRED_PRICE_COLUMNS if col not in stock_data.columns]
    if missing_columns:
        raise ValueError(
            f"Downloaded data for '{symbol}' is missing columns: {', '.join(missing_columns)}"
        )

    stock_data["Ticker"] = symbol
    return stock_data


def build_stock_universe(tickers: list[str], lookback_days: int = 365 * 3) -> pd.DataFrame:
    """Download and combine historical price data for multiple tickers."""
    frames = [download_stock_data(ticker, lookback_days=lookback_days) for ticker in tickers]
    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    return combined.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, deduplicate, and fill missing price values."""
    data = normalize_downloaded_columns(df).copy()
    data["Date"] = pd.to_datetime(data["Date"])

    if "Ticker" not in data.columns:
        raise ValueError("The input data must contain a 'Ticker' column.")

    data = data.sort_values(["Ticker", "Date"]).drop_duplicates(subset=["Ticker", "Date"]).reset_index(drop=True)

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    data[numeric_cols] = data.groupby("Ticker")[numeric_cols].transform(lambda s: s.ffill().bfill())
    return data


def _calculate_indicators_for_ticker(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the same technical indicators used during training."""
    data = frame.sort_values("Date").copy()

    data["Daily_Return"] = data["Close"].pct_change() * 100
    data["MA_5"] = data["Close"].rolling(window=5).mean()
    data["MA_10"] = data["Close"].rolling(window=10).mean()
    data["MA_20"] = data["Close"].rolling(window=20).mean()
    data["MA_50"] = data["Close"].rolling(window=50).mean()
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))
    data.loc[(loss == 0) & (gain > 0), "RSI"] = 100
    data.loc[(loss == 0) & (gain == 0), "RSI"] = 50

    data["BB_Middle"] = data["Close"].rolling(window=20).mean()
    bb_std = data["Close"].rolling(window=20).std()
    data["BB_Upper"] = data["BB_Middle"] + (bb_std * 2)
    data["BB_Lower"] = data["BB_Middle"] - (bb_std * 2)
    data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / data["BB_Middle"]

    data["Volume_MA_10"] = data["Volume"].rolling(window=10).mean()
    data["Volume_Ratio"] = data["Volume"] / data["Volume_MA_10"]
    data["Momentum_5"] = data["Close"] - data["Close"].shift(5)
    data["Momentum_10"] = data["Close"] - data["Close"].shift(10)
    data["Volatility_10"] = data["Daily_Return"].rolling(window=10).std()
    data["Volatility_20"] = data["Daily_Return"].rolling(window=20).std()
    data["HL_Spread"] = (data["High"] - data["Low"]) / data["Close"] * 100
    data["Price_Range"] = (data["High"] - data["Low"]) / data["Open"] * 100
    data["Gap_Open"] = (data["Open"] - data["Close"].shift(1)) / data["Close"].shift(1) * 100
    data["Day_Of_Week"] = data["Date"].dt.day_name()

    return data.replace([np.inf, -np.inf], np.nan)


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply technical indicator creation ticker by ticker."""
    cleaned = clean_stock_data(df)
    frames = [_calculate_indicators_for_ticker(frame) for _, frame in cleaned.groupby("Ticker", sort=False)]
    return pd.concat(frames, ignore_index=True).sort_values(["Date", "Ticker"]).reset_index(drop=True)


def encode_categorical_features(df: pd.DataFrame, categorical_cols: list[str] | None = None) -> pd.DataFrame:
    """One-hot encode categorical columns used by the model."""
    categorical_cols = categorical_cols or ["Ticker", "Day_Of_Week"]
    present_cols = [col for col in categorical_cols if col in df.columns]
    if not present_cols:
        return df.copy()
    return pd.get_dummies(df, columns=present_cols, drop_first=False, dtype=int)


def load_artifacts() -> tuple[object, StandardScaler, list[str], dict[str, object]]:
    """Load the saved model, scaler, feature names, and model metadata."""
    required_paths = [MODEL_PATH, SCALER_PATH]
    missing_files = [path.name for path in required_paths if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(missing_files) + ". Run the notebook through Step 7 first."
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    feature_names: list[str] = []

    if MODEL_INFO_PATH.exists():
        model_info = joblib.load(MODEL_INFO_PATH)
        if not isinstance(model_info, dict):
            model_info = {}
    else:
        model_info = {}

    if model_info.get("features"):
        feature_names = list(model_info["features"])
    elif FEATURE_NAMES_PATH.exists():
        feature_names = [
            line.strip()
            for line in FEATURE_NAMES_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif hasattr(scaler, "feature_names_in_"):
        feature_names = list(scaler.feature_names_in_)
    else:
        raise FileNotFoundError("Feature names are missing. Re-run the notebook to regenerate the saved artifacts.")

    return model, scaler, feature_names, model_info


def prepare_inference_dataset(
    ticker: str,
    feature_names: list[str],
    lookback_days: int = 365,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare the latest rows for live prediction."""
    raw_data = download_stock_data(ticker, lookback_days=lookback_days)
    feature_data = calculate_technical_indicators(raw_data).dropna().reset_index(drop=True)
    if feature_data.empty:
        raise ValueError("Not enough recent rows to calculate technical indicators.")

    encoded = encode_categorical_features(feature_data)
    model_frame = encoded.reindex(columns=feature_names).ffill().bfill().fillna(0.0)
    return feature_data, model_frame


def predict_with_confidence(
    model: object,
    scaler: StandardScaler,
    feature_frame: pd.DataFrame,
) -> tuple[int, np.ndarray]:
    """Predict the class and return down/up probabilities."""
    scaled = scaler.transform(feature_frame.astype(float))
    prediction = int(model.predict(scaled)[0])

    if hasattr(model, "predict_proba"):
        raw_prob = model.predict_proba(scaled)[0]
        class_index = {int(label): idx for idx, label in enumerate(model.classes_)}
        probability_down = float(raw_prob[class_index.get(0, 0)])
        probability_up = float(raw_prob[class_index.get(1, len(raw_prob) - 1)])
        return prediction, np.array([probability_down, probability_up])

    return prediction, np.array([1.0, 0.0]) if prediction == 0 else np.array([0.0, 1.0])


def run_prediction(
    ticker: str,
    model: object,
    scaler: StandardScaler,
    feature_names: list[str],
) -> dict[str, object]:
    """Prepare live features and score the latest available row."""
    try:
        feature_data, model_frame = prepare_inference_dataset(ticker, feature_names, lookback_days=365)
        latest_features = model_frame.iloc[[-1]]
        prediction, probabilities = predict_with_confidence(model, scaler, latest_features)

        return {
            "prediction": prediction,
            "probabilities": probabilities,
            "feature_data": feature_data,
            "model_frame": model_frame,
        }
    except Exception as exc:
        return {"error": str(exc)}
