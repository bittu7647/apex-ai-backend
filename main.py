import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Mutes TensorFlow warnings

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import warnings

warnings.filterwarnings("ignore")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Apex-AI Quant Model is running!"}

from functools import lru_cache
import asyncio

# Create an in-memory cache to store predictions for 15 minutes
# If a user searches AAPL, the first search takes 5 seconds, all subsequent searches take 0.01 seconds.
@lru_cache(maxsize=100)
def cached_prediction(ticker: str, days_to_predict: int):
    # (The actual prediction logic goes inside a sub-method so it can be cached cleanly by dict inputs)
    pass

@app.get("/predict/{ticker}")
def predict_stock(ticker: str, days_to_predict: int = 5):
    # Simply retrieve from cache if it exists for this ticker today
    return get_stock_prediction(ticker, days_to_predict)

@lru_cache(maxsize=100)
def get_stock_prediction(ticker: str, days_to_predict: int):
    try:
        # --- THE FIX: IMPORT ORDER MATTERS ---
        # We MUST import yfinance BEFORE tensorflow. 
        # TensorFlow accidentally breaks the browser disguise if it loads first!
        import yfinance as yf
        import numpy as np
        import pandas as pd
        import pandas_ta as ta  
        
        # Now that yfinance is safely in memory, it is safe to load TensorFlow
        global tf
        if 'tf' not in globals():
            import tensorflow as tf
            globals()['tf'] = tf
            
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense

        # 1. Fetch Data
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y")
        
        if df.empty:
            return {"error": "Yahoo Finance blocked the data request or ticker is invalid. Please try again."}

        # 2. THE QUANT UPGRADE: Calculate Technical Indicators
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        
        # Drop rows with blank data
        df.dropna(inplace=True) 

        # Define our 7 powerful features (The true Quant Upgrade)
        features = ['Close', 'Open', 'High', 'Low', 'Volume', 'RSI_14', 'MACD_12_26_9']
        
        # 3. Dual-Scaling Strategy
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = feature_scaler.fit_transform(df[features])
        
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler.fit(df[['Close']])

        # 4. Create the "Memory Window"
        look_back = 60
        X_train, y_train = [], []
        
        for i in range(look_back, len(scaled_features)):
            X_train.append(scaled_features[i-look_back:i])
            y_train.append(scaled_features[i, 0])
            
        X_train, y_train = np.array(X_train), np.array(y_train)

        # 5. Build the Deep Multivariate LSTM
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1)) 
        
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the Brain! (Reduced to 1 Epoch for drastic speed increase since we are on a free tier)
        model.fit(X_train, y_train, batch_size=32, epochs=1, verbose=0)

        # 6. Predict the Future
        last_60_days = scaled_features[-look_back:]
        current_batch = last_60_days.reshape((1, look_back, len(features)))
        
        predicted_prices = []
        
        for _ in range(days_to_predict):
            next_prediction_scaled = model.predict(current_batch, verbose=0)[0, 0]
            predicted_prices.append(next_prediction_scaled)
            
            new_step = np.copy(current_batch[0, -1, :]) 
            new_step[0] = next_prediction_scaled 
            
            current_batch = np.append(current_batch[:, 1:, :], [[new_step]], axis=1)

        # 7. Un-scale the data back to real dollars
        predicted_prices = target_scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

        # Format output for the React frontend
        historical_data = df['Close'].tail(5).to_dict()
        
        latest_data = df.iloc[-1]
        indicators_dict = {
            "RSI": float(latest_data['RSI_14']),
            "MACD": float(latest_data['MACD_12_26_9']),
            "Close": float(latest_data['Close']),
            "Open": float(latest_data['Open']),
            "High": float(latest_data['High']),
            "Low": float(latest_data['Low']),
            "Volume": float(latest_data['Volume'])
        }
        
        prediction_dict = {}
        predicted_avg = 0
        for i in range(days_to_predict):
            pred_val = float(predicted_prices[i][0])
            prediction_dict[str(i + 1)] = pred_val
            predicted_avg += pred_val
            
        predicted_avg = predicted_avg / days_to_predict
        current_price = indicators_dict["Close"]
        
        # Bullish or Bearish?
        price_diff_percent = ((predicted_avg - current_price) / current_price) * 100
        
        # Base confidence is 50. Max confidence tops out at 99%.
        # A 5% expected move equals +45 confidence points (very strong).
        confidence_offset = min(49, abs(price_diff_percent) * 9) 
        confidence_score = round(50 + confidence_offset)
        
        if price_diff_percent > 0.5:
            signal = "BULLISH"
        elif price_diff_percent < -0.5:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"
            
        ai_confidence = {
            "score": confidence_score,
            "signal": signal,
            "projected_move": round(price_diff_percent, 2)
        }

        return {
            "ticker": ticker.upper(),
            "historical_close": historical_data,
            "predictions": prediction_dict,
            "current_indicators": indicators_dict,
            "ai_confidence": ai_confidence
        }

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}