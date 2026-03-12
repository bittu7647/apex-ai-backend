from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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
    return {"message": "LSTM Stock Prediction API is running!"}

@app.get("/predict/{ticker}")
def predict_stock(ticker: str, days_to_predict: int = 5):
    try:
        # 1. Fetch 2 years of data (Neural Networks need a lot of history)
        stock_data = yf.download(ticker, period="2y", interval="1d")
        
        if stock_data.empty:
            return {"error": "No data found for this ticker."}

        df = stock_data[['Close']].dropna()
        dataset = df.values

        # 2. Scale the data between 0 and 1 (Crucial for LSTM)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # 3. Create the "Memory Window" (Look at the last 60 days to predict the next 1 day)
        look_back = 60
        X_train, y_train = [], []
        
        for i in range(look_back, len(scaled_data)):
            X_train.append(scaled_data[i-look_back:i, 0])
            y_train.append(scaled_data[i, 0])
            
        X_train, y_train = np.array(X_train), np.array(y_train)
        
        # Reshape data for LSTM [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # 4. Build the LSTM Neural Network
        model = Sequential()
        model.add(LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
        model.add(Dense(1)) # Output layer
        
        model.compile(optimizer='adam', loss='mean_squared_error')

        # 5. Train the Brain! (Epochs=5 to keep it fast for your prototype)
        model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=0)

        # 6. Predict the Future
        # Get the last 60 days to start the prediction chain
        last_60_days = scaled_data[-look_back:]
        current_batch = last_60_days.reshape((1, look_back, 1))
        
        predicted_prices = []
        
        for _ in range(days_to_predict):
            # Predict the next day
            next_prediction = model.predict(current_batch)
            predicted_prices.append(next_prediction[0, 0])
            
            # Slide the window forward: remove the oldest day, add the new prediction
            current_batch = np.append(current_batch[:, 1:, :], [[next_prediction[0]]], axis=1)

        # 7. Un-scale the data back to real dollar amounts
        predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

        # Format output to match exactly what your React app expects
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
        return {"error": str(e)}