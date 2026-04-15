import os
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import joblib
from groq import Groq

# NLP Imports
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure NLTK lexicon is downloaded
nltk.download('vader_lexicon', quiet=True)

app = Flask(__name__)
# Enable CORS so your HTML file can talk to this server
CORS(app) 

# --- CONFIGURATION ---
# It is best practice to use environment variables for API keys. 
# We fallback to the key from your notebook for ease of testing.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize Groq Client
try:
    groq_client = Groq()
    LLM_AVAILABLE = True
except Exception as e:
    print(f"Warning: Groq API not available. {e}")
    LLM_AVAILABLE = False

# --- LOAD ML ASSETS ---
MODEL_PATH = 'model/stock_lstm.h5'
SCALER_PATH = 'model/scaler.pkl'
SEQUENCE_LENGTH = 30

try:
    lstm_model = load_model(MODEL_PATH)
    data_scaler = joblib.load(SCALER_PATH)
    print("✅ Model and Scaler loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning: Could not load model/scaler. Ensure they exist in the 'model/' folder. Error: {e}")


def get_live_sentiment(ticker):
    """Fetches recent news from yfinance and calculates Vader sentiment."""
    ticker_obj = yf.Ticker(ticker)
    news_data = ticker_obj.news
    sia = SentimentIntensityAnalyzer()
    
    if not news_data:
        return 0.0

    scores = []
    for article in news_data[:5]:
        title = article.get('title', '')
        score = sia.polarity_scores(title)['compound']
        scores.append(score)
        
    return sum(scores) / len(scores) if scores else 0.0

def fetch_and_prepare_data(ticker):
    """Fetches enough historical data to compute indicators and return the last 30 days."""
    # Fetch 6 months of data to safely calculate 50-day MA and 14-day RSI
    df = yf.download(ticker, period="6mo", interval="1d", progress=False)
    
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")

    # Flatten multi-index if yfinance returns one
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.reset_index(inplace=True)
    df.sort_values(by='Date', ascending=True, inplace=True)
    df.dropna(subset=['Close'], inplace=True)

    # Calculate Indicators
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Return'] = df['Close'].pct_change()
    
    # Add Sentiment
    live_sentiment = get_live_sentiment(ticker)
    df['Sentiment'] = 0.0 
    df.iloc[-1, df.columns.get_loc('Sentiment')] = live_sentiment

    # Drop NaNs created by rolling windows
    df.dropna(inplace=True)
    
    # Extract the last 30 days of features
    features = ['Close', 'MA_10', 'MA_50', 'RSI', 'Return', 'Sentiment']
    latest_data = df[features].tail(SEQUENCE_LENGTH)
    
    if len(latest_data) < SEQUENCE_LENGTH:
        raise ValueError("Not enough historical data to form a 30-day sequence.")
        
    current_price = df['Close'].values[-1]
    
    return latest_data, current_price, live_sentiment

def get_llm_reasoning(ticker, signal, advice, current_price, shares):
    """Generates human-readable context using Groq Llama 3."""
    
    # Custom professional message for API limits or disabled state
    fallback_message = (
        "Due to credit limitations and the risk of public use of the API, "
        "we have temporarily halted human-explainable text. "
        "We are actively working on replacing it with an internal XAI (Explainable AI) model."
    )

    if not LLM_AVAILABLE:
        return fallback_message
        
    prompt = f"""
    You are a senior financial analyst. I am looking at the stock '{ticker}'.
    1. Deep Learning Prediction: Our LSTM model predicts a '{signal}' 5-day trend.
    2. User Situation: The user currently owns {shares} shares.
    3. Algorithm Recommendation: '{advice}'.
    4. Current Price: {current_price:.2f}.

    Please provide a short explanation (3-4 bullet points) for this recommendation.
    Focus on:
    - Why this advice suits the user's current portfolio situation.
    - General sector performance or recent news typical for this company.
    - A disclaimer that this is AI generated based on mathematical momentum.
    """
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        # Replaces the raw 401 error with your custom XAI message
        return fallback_message


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # 1. Parse incoming JSON
        data = request.json
        ticker = data.get('ticker', '').upper().strip()
        shares = float(data.get('shares', 0))
        
        if not ticker:
            return jsonify({"success": False, "error": "Ticker symbol is required."}), 400

        # 2. Fetch and Prepare Data
        latest_data, current_price, sentiment = fetch_and_prepare_data(ticker)
        
        # 3. Scale and Reshape for LSTM
        scaled_features = data_scaler.transform(latest_data)
        tensor_input = np.array([scaled_features]) # Shape: (1, 30, 6)
        
        # 4. Model Inference
        buy_prob = float(lstm_model.predict(tensor_input, verbose=0)[0][0])
        
        # 5. Logic Engine
        if buy_prob > 0.55:
            market_signal = "BULLISH (Uptrend)"
            action = "BUY"
        elif buy_prob < 0.45:
            market_signal = "BEARISH (Downtrend)"
            action = "SELL"
        else:
            market_signal = "NEUTRAL"
            action = "HOLD"

        if action == "SELL" and shares <= 0:
            action = "AVOID"

        # 6. Groq LLM Context
        reasoning = get_llm_reasoning(ticker, market_signal, action, current_price, shares)
        
        # 7. Send Response back to HTML
        return jsonify({
            "success": True,
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "sentiment_score": round(sentiment, 2),
            "signal": market_signal,
            "probability": f"{buy_prob:.2f}",
            "advice": action,
            "reasoning": reasoning
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # Update to use the environment's port, default to 5000 for local testing
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting AI Stock Predictor API on Port {port}...")
    # Bind to 0.0.0.0 so Render can route external traffic to the app
    app.run(host='0.0.0.0', port=port, debug=False)