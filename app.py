import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras.models import load_model
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load pre-trained model and scaler
model = load_model('lstm_stock_model.keras')
scaler = joblib.load('scaler.save')

# App title
st.title("Stock Closing Price Predictor using LSTM")

# Inputs
ticker = st.text_input("Enter Stock Ticker Symbol:", value='AAPL')
start_date = st.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))

# Predict button
if st.button("Predict"):
    st.info("Fetching data from Yahoo Finance...")
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty or len(df) < 110:
        st.error("Not enough data for prediction. Ensure at least 110 days of data.")
    else:
        df = df.reset_index()
        close_actual = df[['Date', 'Close']].copy()

        # Select features and scale
        data = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        scaled_data = scaler.transform(data)
        scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

        # Create sequences from last portion of data
        def create_sequences(X, y, time_steps=10):
            Xs, ys = [], []
            for i in range(len(X) - time_steps):
                Xs.append(X[i:(i + time_steps)])
                ys.append(y[i + time_steps])
            return np.array(Xs), np.array(ys)

        time_steps = 10
        X_all, y_all = create_sequences(scaled_df.drop('Close', axis=1).values, scaled_df['Close'].values)

        # Predict
        y_pred_scaled = model.predict(X_all)

        # Inverse transform both actual and predicted prices
        def inverse_close(preds, X_features):
            reconstructed = np.concatenate((X_features, preds), axis=1)
            return scaler.inverse_transform(reconstructed)[:, -1]

        X_last_features = X_all[:, -1, :]
        predicted_close = inverse_close(y_pred_scaled, X_last_features)
        actual_close = inverse_close(y_all.reshape(-1, 1), X_last_features)

        # Plot actual vs predicted
        st.subheader("📊 Actual vs Predicted Closing Prices (Recent Days)")
        plt.figure(figsize=(12, 6))
        plt.plot(actual_close, label="Actual Close")
        plt.plot(predicted_close, label="Predicted Close")
        plt.title(f"{ticker} - Actual vs Predicted Closing Prices")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend()
        plt.grid()
        st.pyplot(plt)

        # Latest predicted closing price
        st.success(f"📌 Latest Predicted Closing Price: ${predicted_close[-1]:.2f}")
