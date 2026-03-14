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

@app.get("/predict/{ticker}")
def predict_stock(ticker: str, days_to_predict: int = 5):
    try:
        # --- THE FIX: LAZY LOADING ---
        # We only load these heavy libraries when a user actually requests a prediction!
        import yfinance as yf
        import numpy as np
        import pandas as pd
        import pandas_ta as ta  
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense

        # 1. Fetch Data (yfinance handles the bot bypass automatically)
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y")
        
        if df.empty:
            return {"error": "Yahoo Finance blocked the data request or ticker is invalid. Please try again."}

        # 2. THE QUANT UPGRADE: Calculate Technical Indicators
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        
        # Drop rows with blank data (MACD takes 26 days to calculate its first point)
        df.dropna(inplace=True) 

        # Define our 4 powerful features
        features = ['Close', 'Volume', 'RSI_14', 'MACD_12_26_9']
        
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

        # Train the Brain! 
        model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=0)

        # 6. Predict the Future
        last_60_days = scaled_features[-look_back:]
        current_batch = last_60_days.reshape((1, look_back, len(features)))
        
        predicted_prices = []
        
        for _ in range(days_to_predict):
            # Predict the next close price
            next_prediction_scaled = model.predict(current_batch, verbose=0)[0, 0]
            predicted_prices.append(next_prediction_scaled)
            
            # Create the next day's data block
            new_step = np.copy(current_batch[0, -1, :]) 
            new_step[0] = next_prediction_scaled 
            
            # Slide the window forward
            current_batch = np.append(current_batch[:, 1:, :], [[new_step]], axis=1)

        # 7. Un-scale the data back to real dollars
        predicted_prices = target_scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

        # Format output for the React frontend
        historical_data = df['Close'].tail(5).to_dict()
        
        prediction_dict = {}
        for i in range(days_to_predict):
            prediction_dict[str(i + 1)] = float(predicted_prices[i][0])

        return {
            "ticker": ticker.upper(),
            "historical_close": historical_data,
            "predictions": prediction_dict
        }

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}