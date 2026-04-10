# Stock Market Prediction - Technical Architecture

**Version:** 1.0  
**Date:** April 2026

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Training Pipeline](#training-pipeline)
3. [Inference Pipeline](#inference-pipeline)
4. [Data Schema](#data-schema)
5. [Model Specifications](#model-specifications)
6. [Code Flow & Functions](#code-flow--functions)
7. [Deployment Topology](#deployment-topology)
8. [Database Schema](#database-schema)
9. [API Specifications](#api-specifications)
10. [Performance Characteristics](#performance-characteristics)

---

## System Architecture

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        STOCK MARKET PREDICTION SYSTEM                    │
└─────────────────────────────────────────────────────────────────────────┘

                           ┌─────────────────────┐
                           │   Yahoo Finance API │
                           │   (yfinance lib)    │
                           └──────────┬──────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
         ┌──────────▼──────────┐          ┌────────────▼─────────┐
         │  Training Pipeline  │          │  Inference Pipeline  │
         │  (Jupyter Notebook) │          │  (Streamlit App)     │
         └──────────┬──────────┘          └────────────┬─────────┘
                    │                                   │
         ┌──────────▼──────────────────────────────────▼────────┐
         │        Artifact Storage (File System)                │
         │                                                      │
         │  ├─ stock_prediction_model.pkl                       │
         │  ├─ stock_prediction_model_weekly.pkl               │
         │  ├─ scaler.pkl & scaler_weekly.pkl                  │
         │  ├─ model_info.pkl & model_info_weekly.pkl          │
         │  ├─ feature_names.txt & feature_names_weekly.txt    │
         │  └─ cleaned_stock_data.csv                          │
         └───────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data Source** | Yahoo Finance API | Market data |
| **Data Fetching** | `yfinance` | Python wrapper for API |
| **Data Processing** | `pandas`, `numpy` | DataFrame operations |
| **ML Training** | `scikit-learn` | Algorithm implementation |
| **Visualization (Training)** | `matplotlib`, `seaborn` | Static charts in notebook |
| **Model Serialization** | `joblib` | Saving/loading models |
| **Web Framework** | `Streamlit` | Interactive UI |
| **Visualization (Web)** | `plotly` | Interactive charts |
| **Development** | `jupyter` | Notebook IDE |

---

## Training Pipeline

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              TRAINING PIPELINE (Stock_Market_Prediction.ipynb)
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─ Step 1: DATA COLLECTION ──────────────────────────┐    │
│  │ Input: Stock tickers (AAPL, MSFT, ...)            │    │
│  │ Output: Raw OHLCV data (450 days per ticker)      │    │
│  │ Source: Yahoo Finance API via yfinance            │    │
│  │ Shape: [ticker_df1, ticker_df2, ...]              │    │
│  └────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│  ┌─ Step 2: DATA NORMALIZATION ───────────────────────┐    │
│  │ Task: Standardize column names across tickers     │    │
│  │ Handle: MultiIndex columns from multi-ticker DL   │    │
│  │ Output: Consistent column naming schema           │    │
│  └────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│  ┌─ Step 3: DATA CLEANING ────────────────────────────┐    │
│  │ Missing Values: ffill → bfill → drop              │    │
│  │ Outliers: IQR method (1.5 × IQR boundaries)       │    │
│  │ Output: cleaned_stock_data.csv                    │    │
│  └────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│  ┌─ Step 4: FEATURE ENGINEERING ──────────────────────┐    │
│  │ Create 13 technical indicators:                    │    │
│  │ • RSI, MACD, Bollinger Bands                       │    │
│  │ • Moving Averages, Momentum                        │    │
│  │ • Volatility, Volume indicators                    │    │
│  │ Output: X [features], y [binary target]           │    │
│  └────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│  ┌─ Step 5: FEATURE SELECTION ────────────────────────┐    │
│  │ Correlation Analysis: Remove cor > 0.95           │    │
│  │ Variance Analysis: Remove zero-variance           │    │
│  │ Importance Ranking: Random Forest feature imp     │    │
│  │ Output: 13 selected features                      │    │
│  └────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│  ┌─ Step 6: DATA SPLITTING ───────────────────────────┐    │
│  │ Method: Time-based split (80/20)                   │    │
│  │ Split Date: 80% of historical dates                │    │
│  │ X_train, y_train, X_test, y_test                   │    │
│  │ Prevents: Look-ahead bias                          │    │
│  └────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│  ┌─ Step 7: FEATURE SCALING ──────────────────────────┐    │
│  │ Scaler: StandardScaler (μ=0, σ=1)                 │    │
│  │ Fit on: Training data only                         │    │
│  │ Apply to: Test data                                │    │
│  │ Save: scaler.pkl for production                    │    │
│  └────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│  ┌─ Step 8: MODEL TRAINING ───────────────────────────┐    │
│  │ Split validation:                                  │    │
│  │  - Baseline: Logistic Regression                   │    │
│  │  - Candidate 1: Random Forest                      │    │
│  │  - Candidate 2: Gradient Boosting                  │    │
│  │  - Candidate 3: SVM (RBF kernel)                   │    │
│  │ Evaluate: F1-score, Accuracy, AUC-ROC            │    │
│  │ Best models selected for hyperparameter tuning     │    │
│  └────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│  ┌─ Step 9: HYPERPARAMETER TUNING ────────────────────┐    │
│  │ Method: GridSearchCV with TimeSeriesSplit (5)      │    │
│  │ Search space (Random Forest):                      │    │
│  │  - n_estimators: [50, 100, 200]                    │    │
│  │  - max_depth: [5, 10, 15, 20]                      │    │
│  │  - min_samples_split: [5, 10, 20]                  │    │
│  │  - min_samples_leaf: [2, 5, 10]                    │    │
│  │ Objective: Maximize F1-score                       │    │
│  │ Output: Best hyperparameters                       │    │
│  └────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│  ┌─ Step 10: FINAL EVALUATION ────────────────────────┐    │
│  │ Test set performance:                              │    │
│  │ • Accuracy, Precision, Recall, F1                 │    │
│  │ • AUC-ROC, Confusion Matrix                        │    │
│  │ • Cross-validation scores (mean ± std)             │    │
│  │ Output: model_info.pkl (metadata)                  │    │
│  └────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│  ┌─ Step 11: MODEL SERIALIZATION ─────────────────────┐    │
│  │ Save artifacts:                                    │    │
│  │ • stock_prediction_model.pkl (trained classifier) │    │
│  │ • scaler.pkl (fitted StandardScaler)              │    │
│  │ • model_info.pkl (dict with metadata)             │    │
│  │ • feature_names.txt (feature list)                │    │
│  │ Format: joblib (pickle-based, sklearn-optimized)  │    │
│  │ Location: Project root directory                   │    │
│  └────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│       ┌──────────────────────────────────────────┐          │
│       │    ARTIFACTS READY FOR PRODUCTION        │          │
│       │  (Used by Streamlit app for inference)   │          │
│       └──────────────────────────────────────────┘          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Parameters

| Component | Value | Rationale |
|-----------|-------|-----------|
| **Historical Data** | 450 days | Sufficient for indicators (BBands needs 20, MACD needs 26) |
| **Train/Test Split** | 80/20 | Standard split with time-based ordering |
| **Cross-Validation** | TimeSeriesSplit(n_splits=5) | Prevents future-looking leakage |
| **Feature Scaling** | StandardScaler | Suitable for most ML algorithms |
| **Primary Metric** | F1-score | Balanced for class imbalance |
| **Random State** | 42 | Reproducibility |
| **N_jobs** | -1 | Use all cores for parallel processing |

---

## Inference Pipeline

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│         INFERENCE PIPELINE (app.py - Streamlit)             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  USER INTERACTION LAYER                                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Sidebar: Select horizon (1d / 5d)                │  │
│  │ 2. Sidebar: Select ticker (dropdown or custom)      │  │
│  │ 3. Main: Click "Predict Next Day/Week" button       │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ARTIFACT LOADING (Cached - runs once)                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ @st.cache_resource                                   │  │
│  │ def load_artifacts(horizon='1d'):                    │  │
│  │   • Load stock_prediction_model.pkl                  │  │
│  │   • Load scaler.pkl                                  │  │
│  │   • Load model_info.pkl (metadata)                   │  │
│  │   • Load feature_names.txt                           │  │
│  │   Return: (model, scaler, features, info)           │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  DATA FETCHING (Live)                                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ def latest_feature_row(ticker, features):           │  │
│  │   • Download last 450 days from Yahoo Finance       │  │
│  │   • Normalize column names                           │  │
│  │   • Check for empty/insufficient data                │  │
│  │   Return: (latest_row_df, full_data_df)             │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  FEATURE COMPUTATION (Same as training)                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Calculate technical indicators:                      │  │
│  │ • Daily_Return, Gap_Open, HL_Spread                 │  │
│  │ • Momentum (5, 10 days)                              │  │
│  │ • MA (20, 50 days)                                   │  │
│  │ • Volatility (10, 20 days)                           │  │
│  │ • RSI (14-period), MACD (12,26,9)                   │  │
│  │ • Bollinger Bands (20-period, ±2σ)                 │  │
│  │ Output: DataFrame with all indicators               │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  DATA PREPARATION                                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ • Drop NaN rows                                       │  │
│  │ • Select latest row (most recent trading day)       │  │
│  │ • Reindex to match training features (ffill/bfill)  │  │
│  │ • Extract last row as prediction input X             │  │
│  │ Shape: (1, 13) - one sample, 13 features            │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  FEATURE SCALING (Using saved scaler)                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ X_scaled = scaler.transform(X)                        │  │
│  │ • Apply same transformations as training            │  │
│  │ • Ensures consistent scaling                         │  │
│  │ • Shape: (1, 13) unchanged                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  PREDICTION (Model inference)                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ IF model has predict_proba():                        │  │
│  │   probs = model.predict_proba(X_scaled)             │  │
│  │   pred = 1 if prob_up >= threshold else 0           │  │
│  │ ELSE:                                                │  │
│  │   pred = model.predict(X_scaled)                    │  │
│  │   prob_up = 1.0 if pred==1 else 0.0                │  │
│  │ Output: pred (0/1), prob_up, prob_down              │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  OUTPUT RENDERING                                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Prediction display:                               │  │
│  │    st.success("Prediction: UP/DOWN")                 │  │
│  │ 2. Probabilities:                                    │  │
│  │    st.write(f"P(UP) = {prob_up*100:.2f}%")          │  │
│  │    st.write(f"P(DOWN) = {prob_down*100:.2f}%")      │  │
│  │ 3. Decision threshold:                               │  │
│  │    st.caption(f"Threshold: {threshold:.2f}")        │  │
│  │ 4. Charts (via render_charts):                       │  │
│  │    • Price candlestick + MA(20) + MA(50)            │  │
│  │    • RSI indicator (with 30/70 levels)              │  │
│  │    • MACD + Signal line                              │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  USER OUTPUT                                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Direction (UP/DOWN)                                  │  │
│  │ Probability scores                                   │  │
│  │ Technical analysis charts                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Latency Breakdown

| Component | Typical Time | Notes |
|-----------|------|-------|
| Artifact loading | <100ms | Cached after first load |
| Data download | 1-2s | Network dependent |
| Feature calculation | 500-800ms | Numpy vectorized |
| Model prediction | 10-50ms | Inference only |
| Chart rendering | 500-1000ms | Plotly rendering |
| **Total** | **3-5s** | User perceives as responsive |

---

## Data Schema

### Training Data Schema

```python
# Raw data from Yahoo Finance
DataFrame(index=Date, columns=[
    'Open': float64,
    'High': float64,
    'Low': float64,
    'Close': float64,
    'Adj_Close': float64,
    'Volume': int64
])
# Shape: (~7500 rows, 7 columns) after concatenating all tickers

# After feature engineering
DataFrame(index=Date, columns=[
    # Raw OHLCV
    'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume',
    
    # Computed indicators (13 features)
    'Gap_Open': float64,
    'Volume_Ratio': float64,
    'RSI': float64,
    'Daily_Return': float64,
    'Volatility_20': float64,
    'Momentum_5': float64,
    'Volatility_10': float64,
    'Momentum_10': float64,
    'HL_Spread': float64,
    'BB_Width': float64,
    'MACD_Signal': float64,
    'Volume_MA_10': float64,
    'BB_Lower': float64,
    
    # Target
    'Target': int64 (0 or 1)
])
# Shape: (~7500 rows, 20 columns)
```

### Cleaned Data Output (CSV)

```
Date,Open,High,Low,Close,Adj_Close,Volume,Daily_Return,...
2021-01-04,126.360001,128.760002,125.080002,126.360001,...
2021-01-05,128.520004,131.490002,127.360001,130.029999,...
...
```

### Feature Names Storage

**File:** `feature_names.txt`
```
Gap_Open
Volume_Ratio
RSI
Daily_Return
Volatility_20
Momentum_5
Volatility_10
Momentum_10
HL_Spread
BB_Width
MACD_Signal
Volume_MA_10
BB_Lower
```

### Model Metadata Storage

**File:** `model_info.pkl` (Python dict)
```python
{
    "model_name": "RandomForestClassifier",
    "accuracy": 0.542,
    "f1_score": 0.516,
    "precision": 0.503,
    "recall": 0.531,
    "auc_roc": 0.582,
    "decision_threshold": 0.5,
    "training_tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ"],
    "n_samples_train": 6000,
    "n_samples_test": 1500,
    "features": ["Gap_Open", "Volume_Ratio", ..., "BB_Lower"]
}
```

---

## Model Specifications

### 1-Day Prediction Model

**Serialized:** `stock_prediction_model.pkl`

```python
sklearn.ensemble.RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
```

**Training characteristics:**
- Input shape: (6000, 13)
- Classes: {0, 1} (DOWN, UP)
- Output: Class labels (0/1) + Probabilities

**Expected performance:**
- Accuracy: 52-55%
- F1-score: 51-53%
- AUC-ROC: 0.55-0.60

### 5-Day (Weekly) Prediction Model

**Serialized:** `stock_prediction_model_weekly.pkl`

```python
sklearn.ensemble.GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.9,
    random_state=42,
    verbose=0
)
```

**Training characteristics:**
- Input shape: (6000, 13)
- Classes: {0, 1} (DOWN, UP)
- Output: Class labels (0/1) + Probabilities

**Expected performance:**
- Accuracy: 50-53%
- F1-score: 49-52%
- AUC-ROC: 0.52-0.58

---

## Code Flow & Functions

### Main Entry Points

#### Training Pipeline (Jupyter)

```python
# Cell execution order:
1. Import libraries
2. Define constants & paths
3. Data Collection (yfinance download)
4. Data Normalization
5. Data Cleaning
6. EDA & Visualizations
7. Feature Engineering
8. Feature Selection
9. Train/Test Split (time-based)
10. Feature Scaling
11. Model Training (multiple algorithms)
12. Hyperparameter Tuning (GridSearchCV)
13. Model Evaluation
14. Save Artifacts (joblib)
```

#### Inference Pipeline (Streamlit)

```python
# app.py execution:

def main():
    st.title("Stock Direction Predictor")
    
    # 1. Load user inputs from sidebar
    horizon = st.sidebar.radio("Prediction horizon", ...)
    ticker = st.sidebar.selectbox("Choose ticker", ...)
    custom_ticker = st.sidebar.text_input("Or type custom ticker", "")
    
    # 2. Load artifacts (cached)
    model, scaler, features, model_info = load_artifacts(horizon)
    
    # 3. Display model info
    st.sidebar.write(f"Model: {model_info.get('model_name')}")
    st.sidebar.write(f"Accuracy: {model_info.get('accuracy')*100:.1f}%")
    
    # 4. Prediction button
    if st.button("Predict Next Day/Week"):
        try:
            # 5. Fetch latest data
            x_live, chart_data = latest_feature_row(ticker, features)
            
            # 6. Scale features
            x_scaled = scaler.transform(x_live.astype(float))
            
            # 7. Make prediction
            threshold = float(model_info.get("decision_threshold", 0.5))
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x_scaled)[0]
                # ... extract probabilities
            else:
                pred = int(model.predict(x_scaled)[0])
            
            # 8. Display results
            st.success(f"Prediction: {'UP' if pred==1 else 'DOWN'}")
            st.write(f"P(UP): {up_prob*100:.2f}%")
            st.write(f"P(DOWN): {down_prob*100:.2f}%")
            
            # 9. Render charts
            render_charts(chart_data, ticker)
            
        except Exception as e:
            st.error(str(e))

if __name__ == "__main__":
    main()
```

### Core Functions

#### `load_artifacts(horizon: str)`

```python
"""Load pre-trained models and scalers."""
Inputs:
  - horizon: "1d" or "5d"
Returns:
  - model: Fitted classifier (RandomForest or GradientBoosting)
  - scaler: Fitted StandardScaler
  - features: List of 13 feature names
  - model_info: Dict with metadata
```

#### `latest_feature_row(ticker: str, features: list) -> Tuple`

```python
"""Fetch latest data and compute features."""
Inputs:
  - ticker: Stock symbol (e.g., "AAPL")
  - features: List of feature names
Process:
  1. Download 450 days historical data
  2. Normalize column names
  3. Compute all 13 technical indicators
  4. Drop NaN rows
  5. Extract latest row
  6. Reindex to match features (ffill/bfill)
Returns:
  - x_live: DataFrame(shape=(1,13)) - latest features
  - chart_data: DataFrame with full history for charts
Errors:
  - ValueError if ticker invalid or insufficient data
```

#### `render_charts(data: pd.DataFrame, ticker: str)`

```python
"""Display technical analysis charts using Plotly."""
Inputs:
  - data: DataFrame with OHLCV + indicators
  - ticker: Stock symbol (for title)
Renders:
  - Candlestick + MA(20) + MA(50)
  - RSI with 30/70 levels
  - MACD + Signal line
```

---

## Deployment Topology

### Local Deployment (Current)

```
┌─────────────────────────────────────┐
│      USER'S LOCAL MACHINE           │
├─────────────────────────────────────┤
│                                      │
│  ┌──────────────────────────────┐   │
│  │  Web Browser                  │   │
│  │  http://localhost:8501        │   │
│  └────────────┬─────────────────┘   │
│               │                      │
│  ┌────────────▼─────────────────┐   │
│  │  Streamlit Server Process     │   │
│  │  (Python runtime)             │   │
│  │  ├─ app.py execution          │   │
│  │  ├─ Artifact loading (.pkl)   │   │
│  │  ├─ Data fetching (yfinance)  │   │
│  │  └─ Prediction inference      │   │
│  └────────────┬─────────────────┘   │
│               │                      │
│  ┌────────────▼─────────────────┐   │
│  │  File System                  │   │
│  │  ├─ *.pkl files               │   │
│  │  ├─ *.txt files               │   │
│  │  └─ *.csv files               │   │
│  └───────────────────────────────┘   │
│               │                      │
│  ┌────────────▼─────────────────┐   │
│  │  Network (Internet)           │   │
│  │  ├─ Yahoo Finance API         │   │
│  │  └─ yfinance library calls    │   │
│  └───────────────────────────────┘   │
│                                      │
└─────────────────────────────────────┘
```

**Process:**
1. User opens browser: `localhost:8501`
2. Streamlit renders UI from app.py
3. User inputs (ticker, horizon, prediction request)
4. App loads cached artifacts
5. App fetches live data from Yahoo Finance
6. App computes features & makes prediction
7. Results displayed in browser

**Constraints:**
- Single-user access
- Dependent on internet connection
- Limited to local machine IP
- No persistent logging

### Scalability Considerations (Future)

For production deployment:
- Deploy on cloud (AWS, GCP, Azure)
- Use containerization (Docker)
- Implement API server (Flask, FastAPI)
- Add database for logging
- Implement model versioning & caching

---

## Database Schema

### Current Implementation: File-Based Storage

```
project_root/
├── cleaned_stock_data.csv
│   └─ Columnar data format (CSV)
│      Structure: Date, Open, High, Low, Close, Adj_Close, Volume, Indicators...
│      Rows: ~7500 (3 years of data concatenated)
│      Purpose: Input dataset reference & Power BI import
│
├── stock_prediction_model.pkl
│   └─ Binary serialized object
│      Type: sklearn.ensemble.RandomForestClassifier
│      Size: ~5-10 MB
│      Purpose: 1-day prediction model
│
├── scaler.pkl
│   └─ Binary serialized object
│      Type: sklearn.preprocessing.StandardScaler
│      Size: <1 MB (just fitted parameters)
│      Purpose: Feature normalization (training & inference)
│
├── model_info.pkl
│   └─ Binary serialized dictionary
│      Content: Metadata (model_name, accuracy, f1_score, etc.)
│      Size: <1 MB
│      Purpose: Model metadata for display & thresholds
│
├── feature_names.txt
│   └─ Plain text file (one feature per line)
│      Rows: 13
│      Purpose: Feature order specification for model input
│
└─ [Weekly variants with "_weekly" suffix]
   └─ Same structure for 5-day prediction models
```

### Alternative: Database-Backed (Future)

```sql
-- If migrating to SQL database:

CREATE TABLE models (
    id INT PRIMARY KEY,
    name VARCHAR(50),
    horizon VARCHAR(5),  -- '1d' or '5d'
    accuracy FLOAT,
    f1_score FLOAT,
    created_at TIMESTAMP,
    model_path VARCHAR(255),
    scaler_path VARCHAR(255)
);

CREATE TABLE features (
    id INT PRIMARY KEY,
    model_id INT,
    feature_name VARCHAR(100),
    feature_index INT,
    importance FLOAT,
    FOREIGN KEY (model_id) REFERENCES models(id)
);

CREATE TABLE predictions (
    id INT PRIMARY KEY,
    model_id INT,
    ticker VARCHAR(10),
    prediction_date DATE,
    predicted_direction INT,  -- 0 or 1
    probability_up FLOAT,
    probability_down FLOAT,
    actual_direction INT,  -- After verification
    FOREIGN KEY (model_id) REFERENCES models(id)
);
```

---

## API Specifications

### Current: Streamlit-Based UI (No explicit API)

### Future: RESTful API Specification

```
Base URL: http://localhost:8000/api/v1

Endpoint 1: Get Prediction
────────────────────────────────────
POST /predict

Request body:
{
    "ticker": "AAPL",
    "horizon": "1d"  // or "5d"
}

Response (200 OK):
{
    "ticker": "AAPL",
    "horizon": "1d",
    "prediction": 1,  // 0=DOWN, 1=UP
    "probability_up": 0.62,
    "probability_down": 0.38,
    "decision_threshold": 0.5,
    "timestamp": "2026-04-10T11:30:00Z",
    "model_info": {
        "model_name": "RandomForestClassifier",
        "accuracy": 0.542,
        "f1_score": 0.516
    }
}

Error (400 Bad Request):
{
    "error": "Invalid ticker symbol",
    "details": "Ticker not found on Yahoo Finance"
}

────────────────────────────────────

Endpoint 2: Get Model Info
────────────────────────────────────
GET /models/{horizon}

Response (200 OK):
{
    "horizon": "1d",
    "model_name": "RandomForestClassifier",
    "training_tickers": ["AAPL", "MSFT", ...],
    "accuracy": 0.542,
    "f1_score": 0.516,
    "auc_roc": 0.582,
    "n_features": 13,
    "features": ["Gap_Open", "Volume_Ratio", ...]
}

────────────────────────────────────

Endpoint 3: Get Historical Predictions
────────────────────────────────────
GET /predictions?ticker=AAPL&limit=100

Response (200 OK):
{
    "ticker": "AAPL",
    "predictions": [
        {
            "date": "2026-04-09",
            "prediction": 1,
            "probability_up": 0.58,
            "actual": 1  // After market close
        },
        ...
    ]
}
```

---

## Performance Characteristics

### Model Performance

**1-Day Model:**
- Accuracy: 52-55%
- Precision: 50-55%
- Recall: 50-55%
- F1-Score: 51-53%
- AUC-ROC: 0.55-0.60
- Feature count: 13

**5-Day Model:**
- Accuracy: 50-53%
- Precision: 49-52%
- Recall: 49-52%
- F1-Score: 49-52%
- AUC-ROC: 0.52-0.58
- Feature count: 13

### Computational Performance

| Operation | Time | Memory | Notes |
|-----------|------|--------|-------|
| Model load | 50-100ms | 50-80MB | Cached after first load |
| Data fetch | 1-2s | 20-30MB | Network-dependent |
| Feature calc | 500-800ms | 10-20MB | Vectorized numpy |
| Prediction | 10-50ms | <1MB | Trivial |
| Chart render | 500-1000ms | 5-10MB | Browser-side |
| **Total** | **3-5s** | **100-150MB** | User-perceived time |

### Scalability Analysis

| Metric | Current | Limiting Factor | Solution |
|--------|---------|-----------------|----------|
| Concurrent users | 1 | Streamlit session model | Multi-session HTTP API |
| Predictions/minute | ~12 | Data download latency | Caching, batching |
| Supported tickers | Any | API rate limits | Rate limiting, queue |
| Historical lookback | 450 days | Memory | Windowing, caching |
| Model update freq | Manual | Development effort | CI/CD automation |

---

## Summary

**Architecture Type:** Traditional ML pipeline with local deployment

**Key Characteristics:**
- Stateless inference (no session state)
- Cached artifact loading
- Real-time data fetching
- Interactive visualization
- Single-user deployment

**Trade-offs:**
- ✓ Simple, easy to understand
- ✓ No external dependencies (except yfinance API)
- ✓ Fast inference (<50ms)
- ✗ Limited concurrency
- ✗ Manual model updates
- ✗ No logging/audit trail

**Recommended improvements:** See deployment section for scalability considerations.

---

**Document Version:** 1.0  
**Last Updated:** April 2026
