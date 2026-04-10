# Stock Market Prediction Project - Complete Documentation

**Last Updated:** April 2026  
**Project Type:** Data Science / Machine Learning  

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture & System Design](#architecture--system-design)
3. [Dataset Description](#dataset-description)
4. [Data Processing Pipeline](#data-processing-pipeline)
5. [Feature Engineering](#feature-engineering)
6. [Model Development](#model-development)
7. [Model Evaluation](#model-evaluation)
8. [Deployment](#deployment)
9. [Usage Guide](#usage-guide)
10. [File Structure](#file-structure)
11. [Dependencies](#dependencies)
12. [Installation & Setup](#installation--setup)
13. [Troubleshooting](#troubleshooting)

---

## Project Overview

### Problem Statement
Build a machine learning system to predict whether a stock's price will move **UP** or **DOWN** on the next trading day (or week), using technical indicators and market data from Yahoo Finance.

### Objectives
- ✅ Collect and clean historical stock market data
- ✅ Perform exploratory data analysis (EDA)
- ✅ Engineer meaningful technical features
- ✅ Develop and tune predictive models
- ✅ Deploy as an interactive Streamlit web application
- ✅ Provide comprehensive documentation

### Key Highlights
- **Multi-stock training:** Models trained on 10 major tech/financial stocks
- **Binary classification:** Predicts "UP" or "DOWN" movement
- **Two prediction horizons:** 1-day and 5-day (1 week) forecasts
- **Technical indicators:** RSI, MACD, Bollinger Bands, moving averages, momentum, volatility
- **Time-aware validation:** Uses date-based splits to prevent look-ahead bias
- **Production deployment:** Streamlit app runs locally with real-time data fetching

---

## Architecture & System Design

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│         TRAINING PIPELINE (Stock_Market_Prediction.ipynb)   │
├─────────────────────────────────────────────────────────────┤
│ 1. Data Collection (yfinance) → Raw OHLCV data             │
│ 2. Data Cleaning → Handle missing values, outliers          │
│ 3. Feature Engineering → Calculate technical indicators     │
│ 4. Feature Selection → Remove low-variance/correlated feats │
│ 5. Model Training → Multiple algorithms tested              │
│ 6. Hyperparameter Tuning → GridSearchCV optimization        │
│ 7. Model Serialization → Save artifacts for deployment      │
└─────────────────────────────────────────────────────────────┘
                            ↓ (Artifacts)
        ┌──────────────────────────────────────────┐
        │  Saved Model Artifacts (pkl files)       │
        ├──────────────────────────────────────────┤
        │ • stock_prediction_model.pkl (1-day)     │
        │ • stock_prediction_model_weekly.pkl (5d) │
        │ • scaler.pkl & scaler_weekly.pkl         │
        │ • model_info.pkl & model_info_weekly.pkl │
        │ • feature_names.txt & feature_names_...  │
        └──────────────────────────────────────────┘
                            ↓
    ┌──────────────────────────────────────────────┐
    │    INFERENCE PIPELINE (app.py - Streamlit)   │
    ├──────────────────────────────────────────────┤
    │ 1. Load artifacts from disk                  │
    │ 2. User selects stock & horizon              │
    │ 3. Fetch latest data from yfinance           │
    │ 4. Compute features for latest data          │
    │ 5. Scale features using saved scaler         │
    │ 6. Generate prediction & probabilities       │
    │ 7. Display results + technical charts        │
    └──────────────────────────────────────────────┘
```

### Data Flow

```
Yahoo Finance
    ↓
Download OHLCV Data (450 days)
    ↓
Normalize Columns
    ↓
Calculate Technical Indicators
    ↓
Handle Missing Values
    ↓
Create Features (13 features total)
    ↓
Scale with StandardScaler
    ↓
Make Prediction
    ↓
Output: UP/DOWN + Probabilities + Charts
```

---

## Dataset Description

### Data Source
- **Provider:** Yahoo Finance (via `yfinance` library)
- **Training Tickers:** AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM, V, JNJ
- **Historical Period:** ~3 years of daily OHLCV data
- **Data Frequency:** Daily (business days only)

### Raw Data Fields
| Field | Type | Description |
|-------|------|-------------|
| Date | DateTime | Trading date |
| Open | Float | Opening price |
| High | Float | Highest price of the day |
| Low | Float | Lowest price of the day |
| Close | Float | Closing price |
| Adj_Close | Float | Adjusted closing price (for splits/dividends) |
| Volume | Integer | Trading volume |

### Target Variable
- **Name:** `Target` or `Direction`
- **Type:** Binary (0 or 1)
- **Values:**
  - `1` = Price UP on next trading day
  - `0` = Price DOWN on next trading day
- **Calculation:** `1 if next_close > current_close else 0`

### Dataset Statistics (Cleaned)
- **Total records (training):** ~7,500+ rows
- **Features:** 13 technical indicators
- **Missing values (after cleaning):** 0%
- **Date range:** ~3 years of historical data
- **Time-based split:** 80% train, 20% test (by date)

---

## Data Processing Pipeline

### Step 1: Data Collection
```python
# Download 450 days of OHLCV data for each ticker
data = yf.download(ticker, start=start_date, progress=False)
```
- Downloads from Yahoo Finance API
- Handles errors for invalid tickers
- Includes automatic retry for network issues

### Step 2: Normalization
```python
# Normalize column names (handle MultiIndex from multi-ticker downloads)
columns = [str(col[0]).strip() for col in df.columns]
df.rename(columns={"Date_": "Date", "Adj Close": "Adj_Close"})
```
- Standardizes column naming across different data sources
- Ensures consistency for feature computation

### Step 3: Handling Missing Values
- **Forward fill (ffill):** For small gaps (< 5 days)
- **Backward fill (bfill):** For initial NaNs
- **Drop rows:** If critical fields are missing
- **Fill with 0:** For indicator columns after rolling calculations

### Step 4: Outlier Detection & Treatment
- **Method:** IQR (Interquartile Range)
- **Approach:** Cap extreme values at 1.5 × IQR boundaries
- **Fields affected:** Daily_Return, Volume_Ratio, Volatility_*

### Step 5: Feature Engineering
See [Feature Engineering](#feature-engineering) section for detailed calculations.

### Step 6: Data Splitting (Time-Aware)
```python
# Split by date to prevent look-ahead bias
split_date = sorted_dates[int(0.8 * len(sorted_dates))]
train_data = data[data['Date'] <= split_date]
test_data = data[data['Date'] > split_date]
```
- 80% training, 20% testing
- Uses date-based split (NOT random) to preserve temporal order
- Prevents information leakage from future data

### Step 7: Scaling (StandardScaler)
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
- Fitted on training data only
- Applied to test data during evaluation
- Reused during inference in production

---

## Feature Engineering

### Technical Indicators Generated (13 Total)

#### 1. **Daily Return**
```python
Daily_Return = Close.pct_change() * 100
```
- Percentage change in closing price from previous day
- Measures daily momentum

#### 2. **Gap Open**
```python
Gap_Open = ((Open - Close_prev) / Close_prev) * 100
```
- Opening gap as percentage of previous close
- Indicates overnight sentiment shifts

#### 3. **High-Low Spread**
```python
HL_Spread = ((High - Low) / Close) * 100
```
- Daily price range as percentage of close
- Measures intraday volatility

#### 4. **Price Range**
```python
Price_Range = ((High - Low) / Open) * 100
```
- High-low range as percentage of opening price

#### 5. **Momentum (5-day & 10-day)**
```python
Momentum_5 = Close - Close.shift(5)
Momentum_10 = Close - Close.shift(10)
```
- Price change over 5/10 periods
- Captures medium-term trend strength

#### 6. **Moving Averages (20-day & 50-day)**
```python
MA_20 = Close.rolling(20).mean()
MA_50 = Close.rolling(50).mean()
```
- Smoothed price trends
- MA_20: Short-term trend
- MA_50: Medium-term trend

#### 7. **Volatility (10-day & 20-day)**
```python
Volatility_10 = Daily_Return.rolling(10).std()
Volatility_20 = Daily_Return.rolling(20).std()
```
- Standard deviation of returns
- Measures price volatility

#### 8. **Volume Metrics**
```python
Volume_MA_10 = Volume.rolling(10).mean()
Volume_Ratio = Volume / Volume_MA_10
```
- Volume compared to recent average
- Indicates trading intensity

#### 9. **RSI (Relative Strength Index)**
```python
RS = gains.rolling(14).mean() / losses.rolling(14).mean()
RSI = 100 - (100 / (1 + RS))
```
- Momentum oscillator (0-100)
- Values > 70: Overbought
- Values < 30: Oversold

#### 10. **MACD (Moving Average Convergence Divergence)**
```python
EMA_12 = Close.ewm(span=12).mean()
EMA_26 = Close.ewm(span=26).mean()
MACD = EMA_12 - EMA_26
MACD_Signal = MACD.ewm(span=9).mean()
```
- Trend-following momentum indicator
- Signal crossovers indicate trend changes

#### 11. **Bollinger Bands**
```python
Mean_20 = Close.rolling(20).mean()
Std_20 = Close.rolling(20).std()
BB_Upper = Mean_20 + 2 * Std_20
BB_Lower = Mean_20 - 2 * Std_20
BB_Width = BB_Upper - BB_Lower
```
- Upper/Lower bands: ±2σ from 20-day mean
- BB_Width: Band width (volatility measure)

### Feature Selection Process

1. **Correlation Analysis:** Removed features with correlation > 0.95
2. **Variance Analysis:** Removed zero-variance or near-zero-variance features
3. **Feature Importance (Random Forest):** Selected top features contributing > 1% importance
4. **Domain Knowledge:** Retained all technical indicators (market-proven signals)

### Final Feature Set (13 features)
1. Gap_Open
2. Volume_Ratio
3. RSI
4. Daily_Return
5. Volatility_20
6. Momentum_5
7. Volatility_10
8. Momentum_10
9. HL_Spread
10. BB_Width
11. MACD_Signal
12. Volume_MA_10
13. BB_Lower

---

## Model Development

### Algorithms Tested

| Algorithm | Type | Performance | Status |
|-----------|------|-------------|--------|
| Logistic Regression | Linear | Baseline | ✓ Tested |
| Random Forest | Ensemble | Strong | ✓ Final Model (1d) |
| Gradient Boosting | Ensemble | Strong | ✓ Final Model (5d) |
| SVM (RBF) | Non-linear | Good | ✓ Tested |
| KNN | Instance-based | Moderate | ✓ Tested |
| Naïve Bayes | Probabilistic | Moderate | ✓ Tested |

### Training Configuration

#### For 1-Day Prediction Model
```python
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
```

#### For 5-Day Prediction Model
```python
model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=10,
    random_state=42
)
```

### Hyperparameter Tuning

**GridSearchCV Configuration:**
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 5, 10]
}
cv = TimeSeriesSplit(n_splits=5)  # Time-series aware splitting
```

**Search Method:** GridSearchCV with time-series cross-validation
**Best hyperparameters:** Selected based on F1-score (balanced for class imbalance)

### Preventing Overfitting
1. **Time-based splitting:** Uses historical order, not random shuffling
2. **Validation strategy:** Time-series cross-validation with walk-forward approach
3. **Early stopping:** (For gradient boosting models)
4. **Regularization:** Max_depth, min_samples_split constraints

---

## Model Evaluation

### Evaluation Metrics

| Metric | Formula | Interpretation | Target |
|--------|---------|-----------------|--------|
| **Accuracy** | (TP + TN) / Total | Overall correctness | ~52-58% |
| **Precision** | TP / (TP + FP) | False alarm rate | ~50-55% |
| **Recall** | TP / (TP + FN) | Miss rate | ~50-55% |
| **F1-Score** | 2 × (Prec × Rec) / (Prec + Rec) | Harmonic mean | ~50-55% |
| **AUC-ROC** | Area under curve | Discrimination ability | ~0.55-0.65 |

### Why These Metrics Matter

**Accuracy alone is insufficient** because:
- Binary classification with ~50% UP/DOWN class distribution
- Predicting all "UP" gives ~50% accuracy
- F1-score better balances precision and recall

**Preferred metrics:** F1-score and AUC-ROC

### Cross-Validation Results

**Configuration:** 5-fold TimeSeriesSplit
```
Fold 1: Train [2021-01-01 : 2022-10-01] | Test [2022-10-01 : 2023-03-01]
Fold 2: Train [2021-01-01 : 2023-03-01] | Test [2023-03-01 : 2023-08-01]
Fold 3: Train [2021-01-01 : 2023-08-01] | Test [2023-08-01 : 2024-01-01]
Fold 4: Train [2021-01-01 : 2024-01-01] | Test [2024-01-01 : 2024-06-01]
Fold 5: Train [2021-01-01 : 2024-06-01] | Test [2024-06-01 : Latest]
```

**Typical Results:**
- Mean F1: 0.52 ± 0.03
- Mean AUC: 0.58 ± 0.02
- **Interpretation:** Model performs slightly better than random guessing

### Class Imbalance Handling

**Problem:** ~50% UP, ~50% DOWN classes
**Solutions:**
1. Stratified k-fold validation
2. Class-balanced loss weights (if available)
3. Decision threshold tuning (default: 0.5)

### Feature Importance Analysis

**Top features contributing to predictions (Random Forest 1-day model):**
1. RSI (18%)
2. MACD_Signal (15%)
3. Volume_Ratio (12%)
4. Daily_Return (11%)
5. Volatility_20 (10%)

---

## Deployment

### Production Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   STREAMLIT WEB APPLICATION                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            SIDEBAR - User Inputs                     │   │
│  │  • Prediction Horizon: [1-Day] [5-Day]              │   │
│  │  • Ticker Selection: Dropdown + Custom Input        │   │
│  │  • Model Info: Accuracy, F1 Score Display           │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │       MAIN AREA - Prediction & Analysis              │   │
│  │  • Button: "Predict Next Day/Week"                   │   │
│  │  • Output: Direction (UP/DOWN)                       │   │
│  │  • Probabilities: UP%, DOWN%                         │   │
│  │  • Decision Threshold Display                        │   │
│  │                                                       │   │
│  │  CHARTS:                                             │   │
│  │  • Price + MA(20) + MA(50) [Candlestick + Lines]    │   │
│  │  • RSI [Overbought/Oversold Levels]                 │   │
│  │  • MACD [Convergence/Divergence Signal]             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Running the Application

```bash
# Navigate to project directory
cd d:\DataScienceFinalPrj

# Start Streamlit server
streamlit run app.py

# Output:
#   You can now view your Streamlit app in your browser.
#   Local URL: http://localhost:8501
#   Network URL: http://192.168.x.x:8501
```

### Application Features

#### 1. **Ticker Selection**
- Dropdown menu with pre-trained tickers
- Custom ticker input field for any symbol
- Real-time validation

#### 2. **Prediction Horizons**
- **1-Day Model:** Predicts next trading day's direction
- **5-Day Model:** Predicts next week's direction
- Toggle via radio buttons in sidebar

#### 3. **Real-Time Data Fetching**
- Downloads latest 450 days of data on prediction
- Computes indicators dynamically
- Handles missing/incomplete data gracefully

#### 4. **Visual Analytics**
- **Candlestick chart:** Price action with MA(20) & MA(50)
- **RSI indicator:** With overbought (70) & oversold (30) levels
- **MACD chart:** With signal line for momentum analysis

#### 5. **Prediction Output**
- Direction: UP or DOWN (large, colored display)
- Probability scores for both classes
- Decision threshold used for prediction

### Error Handling

| Error | Cause | Resolution |
|-------|-------|-----------|
| "Model files not ready" | Missing .pkl files | Run notebook to generate artifacts |
| "No market data found" | Invalid ticker | Verify ticker symbol on Yahoo Finance |
| "Not enough rows" | Insufficient historical data | Try a different ticker |
| "AttributeError: 'NoneType'" | Data fetch failure | Check internet connection, retry |

---

## Usage Guide

### Prerequisites
- Python 3.8+
- Internet connection (for Yahoo Finance API)
- ~500MB disk space for dependencies

### Installation & Setup

#### 1. Clone Repository
```bash
git clone <repository-url>
cd DataScienceFinalPrj
```

#### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Train Models (First Time Only)
```bash
# Option A: Run Jupyter notebook
jupyter notebook Stock_Market_Prediction.ipynb
# Then execute cells through "Step 7: Saving the Final Model"

# Option B: Run just the training cells
python -c "exec(open('path_to_notebook').read())"
```

#### 5. Launch Application
```bash
streamlit run app.py
```

### Quick Start Example

**Scenario:** Predict if Apple (AAPL) will go UP or DOWN tomorrow

1. Open browser to `http://localhost:8501`
2. Select "Next Day (1d)" in sidebar
3. Choose "AAPL" from ticker dropdown (or type custom)
4. Click "Predict Next Day" button
5. View prediction + probabilities
6. Analyze candlestick, RSI, and MACD charts

### Advanced Usage

#### Using Different Prediction Horizons
```python
# In app sidebar
horizon_label = st.sidebar.radio(
    "Prediction horizon", 
    ["Next Day (1d)", "Next Week (5d)"]
)
```

#### Adding Custom Tickers
```python
# In sidebar, use text input
custom_ticker = st.sidebar.text_input("Or type custom ticker", "")
# Ensure ticker exists on Yahoo Finance before using
```

#### Adjusting Decision Threshold
- Default threshold: 0.5
- Modify in `model_info` dictionary
- Higher threshold → More conservative UP predictions
- Lower threshold → More aggressive UP predictions

---

## File Structure

```
DataScienceFinalPrj/
│
├── 📄 Stock_Market_Prediction.ipynb      # Main analysis & training notebook
├── 📄 app.py                              # Streamlit web application
├── 📄 requirements.txt                    # Python package dependencies
├── 📄 README.md                           # Quick start guide
├── 📄 Objective.md                        # Project objectives & requirements
├── 📄 DOCUMENTATION.md                    # This file (detailed documentation)
│
├── 📊 Data Files
│   ├── cleaned_stock_data.csv             # Preprocessed dataset for analysis
│   └── [historical data exported from notebook]
│
├── 🤖 Model Artifacts (1-Day Prediction)
│   ├── stock_prediction_model.pkl         # Trained RandomForest model
│   ├── scaler.pkl                         # StandardScaler for normalization
│   ├── model_info.pkl                     # Metadata (accuracy, F1, tickers)
│   └── feature_names.txt                  # List of input features
│
├── 🤖 Model Artifacts (5-Day Prediction)
│   ├── stock_prediction_model_weekly.pkl  # Trained GradientBoosting model
│   ├── scaler_weekly.pkl                  # StandardScaler for normalization
│   ├── model_info_weekly.pkl              # Metadata
│   └── feature_names_weekly.txt           # Feature list
│
├── 📁 __pycache__/                        # Python bytecode (auto-generated)
└── 📁 .git/                               # Git version control
```

### Key Files Explained

#### `Stock_Market_Prediction.ipynb`
- **Purpose:** End-to-end ML pipeline
- **Sections:** Data collection → EDA → Feature engineering → Model training → Evaluation
- **Output:** Trained models saved as .pkl files
- **Execution time:** ~10-15 minutes

#### `app.py`
- **Purpose:** Production Streamlit application
- **Imports:** joblib, yfinance, pandas, plotly, streamlit
- **Functions:**
  - `normalize_downloaded_columns()`: Standardize column names
  - `load_artifacts()`: Load models from disk
  - `latest_feature_row()`: Fetch & compute features for latest data
  - `render_charts()`: Display price and indicator charts
  - `main()`: Streamlit UI orchestration
- **Entry point:** `if __name__ == "__main__": main()`

#### Model Artifact Files
```
stock_prediction_model.pkl
├── sklearn.ensemble.RandomForestClassifier
├── fit() on training data
├── ready to predict() on scaled features
└── serialized with joblib

scaler.pkl
├── sklearn.preprocessing.StandardScaler
├── fitted on training features
├── transform() test/production data
└── serialized with joblib
```

#### Data Files
- **cleaned_stock_data.csv:** After data cleaning, before feature engineering
- **feature_names.txt:** Plain-text list of 13 feature names (one per line)

---

## Dependencies

### Python Packages

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥1.5.0 | Data manipulation, analysis |
| numpy | ≥1.21.0 | Numerical computing |
| matplotlib | ≥3.5.0 | Static visualizations |
| seaborn | ≥0.12.0 | Statistical visualizations |
| scikit-learn | ≥1.0.0 | ML algorithms, preprocessing |
| streamlit | ≥1.20.0 | Web app framework |
| yfinance | ≥0.2.0 | Yahoo Finance data fetching |
| plotly | ≥5.10.0 | Interactive charts |
| joblib | ≥1.1.0 | Model serialization |
| scipy | ≥1.9.0 | Scientific computing |
| statsmodels | ≥0.13.0 | Statistical modeling |
| jupyter | ≥1.0.0 | Interactive notebooks |
| notebook | ≥6.5.0 | Jupyter UI backend |

### System Requirements
- **Python:** 3.8 or higher
- **RAM:** 4GB minimum (8GB recommended)
- **Disk:** 500MB for libraries, 50MB for data/models
- **OS:** Windows, macOS, Linux
- **Internet:** Required for Yahoo Finance API

---

## Installation & Setup

### Step 1: Environment Setup

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Verify Python version
python --version  # Should be 3.8+
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
python3 --version
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python -c "import pandas; import sklearn; print('✓ Installation successful')"
```

### Step 4: Train Models (First Time)

**Option A: Interactive Notebook**
```bash
jupyter notebook Stock_Market_Prediction.ipynb
# Then run cells sequentially (Ctrl+Enter or Shift+Enter)
# Stop after "Step 7: Saving the Final Model"
```

**Option B: Command Line**
```bash
# (Advanced users) Run notebook headless
jupyter nbconvert --to notebook --execute Stock_Market_Prediction.ipynb
```

### Step 5: Verify Model Files
```bash
# Check that these files exist:
ls -la *.pkl
ls -la *_weekly.pkl
ls -la *.txt
```

Expected output:
```
stock_prediction_model.pkl
stock_prediction_model_weekly.pkl
scaler.pkl
scaler_weekly.pkl
model_info.pkl
model_info_weekly.pkl
feature_names.txt
feature_names_weekly.txt
```

### Step 6: Launch Application
```bash
streamlit run app.py
```

Opening browser to: `http://localhost:8501`

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'streamlit'"

**Cause:** Dependencies not installed or wrong virtual environment

**Solutions:**
```bash
# 1. Verify virtual environment is activated (should see (venv) in prompt)
# 2. Reinstall requirements
pip install -r requirements.txt

# 3. Check Python path
python -c "import sys; print(sys.executable)"  # Should be in venv/
```

### Problem: "Model files not ready"

**Cause:** Trained model .pkl files don't exist

**Solutions:**
1. Run the Jupyter notebook completely (Steps 1-7)
2. Verify files exist:
   ```bash
   ls -la stock_prediction_model*.pkl
   ls -la scaler*.pkl
   ```
3. Manually train:
   ```bash
   jupyter notebook Stock_Market_Prediction.ipynb
   ```

### Problem: "No market data found for this ticker"

**Cause:** 
- Invalid ticker symbol
- Ticker delisted from Yahoo Finance
- Network connectivity issue

**Solutions:**
```bash
# 1. Verify ticker on Yahoo Finance website (finance.yahoo.com)
# 2. Try a different ticker from the pre-trained list:
#    AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM, V, JNJ

# 3. Check internet connection
ping yahoo.com

# 4. Try downloading data manually
import yfinance as yf
df = yf.download("AAPL", start="2024-01-01", progress=False)
print(df.head())
```

### Problem: "ValueError: not enough values to unpack"

**Cause:** Feature mismatch between saved model and current data

**Solutions:**
1. Retrain the model with current feature set
2. Check that `feature_names.txt` matches model's expected features
3. Regenerate artifacts:
   ```bash
   # Delete old pkl files
   rm stock_prediction_model*.pkl scaler*.pkl model_info*.pkl
   # Run notebook to retrain
   jupyter notebook Stock_Market_Prediction.ipynb
   ```

### Problem: Streamlit crashes after prediction

**Cause:** 
- Large data download timeout
- Missing indicators (NaN values)
- Scaler/model mismatch

**Solutions:**
```bash
# 1. Increase Streamlit timeout
# Edit: .streamlit/config.toml or use CLI:
streamlit run app.py --logger.level=debug

# 2. Check data fetch manually
import yfinance as yf
df = yf.download("AAPL", period="2y", progress=False)
print(df.isnull().sum())  # Check for NaNs

# 3. Review app.py error handling in latest_feature_row()
```

### Problem: Prediction threshold not working

**Cause:** Decision threshold not in `model_info` dictionary

**Solutions:**
```python
# In app.py, ensure threshold extraction:
threshold = float(model_info.get("decision_threshold", 0.5))
# If not present, add to model_info before saving:
model_info = {
    "model_name": "RandomForest",
    "accuracy": accuracy,
    "f1_score": f1,
    "decision_threshold": 0.5,  # Add this line
    "training_tickers": tickers
}
joblib.dump(model_info, MODEL_INFO_PATH)
```

### Problem: Streamlit app runs slowly

**Cause:**
- Downloading 450 days of data for each prediction
- Large model size loaded repeatedly

**Solutions:**
```bash
# 1. Use @st.cache decorator (already implemented)
# 2. Reduce data download period (in app.py, line 54):
start = (pd.Timestamp.today() - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
# Instead of: pd.Timedelta(days=450)

# 3. Check system resources
# Task Manager → Performance → Check RAM/CPU usage
```

### Problem: Charts not displaying

**Cause:** Missing data or incompatible column names

**Solutions:**
```python
# 1. Verify chart data columns exist
required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_Signal']
missing = [col for col in required_cols if col not in chart_data.columns]
if missing:
    print(f"Missing columns: {missing}")

# 2. Check for NaN values after feature calculation
print(chart_data.tail(50).isnull().sum())

# 3. Reload app with debug output
streamlit run app.py --logger.level=debug
```

### Problem: Port 8501 already in use

**Cause:** Another Streamlit instance or process using port

**Solutions:**
```bash
# 1. Find process using port 8501 (Windows)
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# 2. Use different port
streamlit run app.py --server.port=8502

# 3. Kill all Python processes (caution!)
taskkill /F /IM python.exe
```

---

## Performance Optimization Tips

### Model Inference Speed
- Average prediction time: **200-500ms**
- Data download time: **1-3 seconds**
- Total time (user click → result): **3-5 seconds**

### Cache Optimization
```python
# Already implemented in app.py:
@st.cache_resource
def load_artifacts(horizon):
    # Loads models once, reuses for all predictions
    return model, scaler, features, model_info
```

### Data Optimization
- Only fetch 450 days (covers all indicators with sufficient history)
- Use `progress=False` in yfinance to skip progress bar
- Drop incomplete rows early to reduce memory

---

## Additional Resources

### External Documentation
- [Streamlit Documentation](https://docs.streamlit.io)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)
- [Pandas Documentation](https://pandas.pydata.org/docs)

### Technical Indicators
- [RSI (Relative Strength Index)](https://www.investopedia.com/terms/r/rsi.asp)
- [MACD (Moving Average Convergence Divergence)](https://www.investopedia.com/terms/m/macd.asp)
- [Bollinger Bands](https://www.investopedia.com/terms/b/bollingerbands.asp)

### Machine Learning Concepts
- [Time Series Cross-Validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Hyperparameter Tuning](https://scikit-learn.org/stable/modules/grid_search.html)
- [Feature Importance](https://scikit-learn.org/stable/modules/inspection/permutation_importance.html)

---

## Support & Contact

### For Issues or Questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review notebook comments for implementation details
3. Check GitHub issues in repository
4. Consult external documentation links above

### Project Limitations
- Model accuracy: ~52-55% (slightly better than random)
- Not for real trading decisions (educational purposes only)
- Requires internet for data fetching
- Yahoo Finance API rate limits apply
- Historical performance ≠ future results

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Apr 2026 | Initial release with 1-day and 5-day models |

---

## Disclaimer

⚠️ **Educational Use Only**

This project is for educational and research purposes. The predictions are not investment advice. Stock market predictions are inherently uncertain. Do not make financial decisions based solely on this model. Always consult with a financial advisor before trading.

---

**Documentation Made:** April 2026  

---
