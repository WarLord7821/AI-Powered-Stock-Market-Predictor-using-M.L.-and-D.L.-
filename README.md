# 📈 Hybrid Stock Analysis & Prediction System

A data-driven investment tool that combines **Extreme Gradient Boosting (XGBoost)** for price forecasting and **Google Gemini API** for automated financial narrative generation. This project bridges the gap between raw numerical predictions and actionable investment insights.

## 🚀 Key Features

* **Quantitative Forecasting**: Uses XGBoost to predict next-day closing prices or price direction (Up/Down) based on technical indicators.
* **AI-Powered Insights**: Integrates **Gemini 2.0 Flash** to analyze news sentiment, interpret model confidence, and generate human-readable "Buy/Hold/Sell" summaries.
* **Feature Engineering**: Includes a robust pipeline for RSI, MACD, Moving Averages (SMA/EMA), and Bollinger Bands.
* **Backtesting Engine**: Evaluates model performance using RMSE, MAE, and Sharpe Ratio metrics.

---

## 🛠️ Tech Stack

| Component | Technology |
| --- | --- |
| **Language** | Python 3.9+ |
| **Model (Numerical)** | XGBoost (Regressor/Classifier) |
| **Model (LLM)** | Google Gemini API (`google-generativeai`) |
| **Data Fetching** | `yfinance` or Alpha Vantage |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Visualization** | Plotly / Matplotlib |

---

## 📂 Project Structure

```bash
├── data/               # Local datasets (CSV files)
├── models/             # Saved XGBoost models (.json or .pkl)
├── notebooks/          # Exploratory Data Analysis (EDA) & Prototyping
├── src/
│   ├── data_loader.py  # Fetches and cleans market data
│   ├── features.py     # Technical indicator engineering
│   ├── train.py        # XGBoost training & hyperparameter tuning
│   ├── gemini_agent.py # Gemini API integration & prompt logic
│   └── main.py         # The main execution pipeline
├── .env                # API keys (GEMINI_API_KEY)
├── requirements.txt    # Project dependencies
└── README.md

```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/stock-analysis-xgboost-gemini.git
cd stock-analysis-xgboost-gemini

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Configure API Keys

Create a `.env` file in the root directory and add your Google AI Studio API key:

```env
GEMINI_API_KEY=your_actual_api_key_here

```

---

## 📊 How It Works

### Step 1: Data & Technicals

The system fetches historical data and calculates features like:

* **Momentum**: RSI (14), MACD.
* **Volatility**: Average True Range (ATR), Bollinger Bands.
* **Trend**: 50-day and 200-day Moving Averages.

### Step 2: XGBoost Training

The model is trained using **TimeSeriesSplit** cross-validation to prevent data leakage.


### Step 3: Gemini Synthesis

The prediction ( for tomorrow) is sent to Gemini alongside recent news headers. Gemini provides a context-aware report:

> "While the XGBoost model predicts a bullish move based on the golden cross, recent earnings reports suggest supply chain headwinds. Recommendation: Moderate Buy."

---

## 📝 Example Usage

Run the full pipeline for a specific ticker:

```bash
python src/main.py --ticker NVDA --days 365

```

---

## ⚠️ Disclaimer

*This software is for educational purposes only. Do not use it as the sole basis for financial decisions. Trading stocks involves significant risk.*

---
