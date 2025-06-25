#!/usr/bin/env python
# coding: utf-8

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os

# Load AAPL data
df = yf.download('AAPL', start='2015-01-01', end='2024-12-31', auto_adjust=True)
df.reset_index(inplace=True)

# Univariate LSTM Data Preparation
data = df[['Date', 'Close']]
data.set_index('Date', inplace=True)

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data, 60)
X = X.reshape(X.shape[0], X.shape[1], 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.load_weights('model_weights.h5')

predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
r2 = r2_score(actual_prices, predicted_prices) * 100

# Technical Indicators: MA20 and RSI
df['MA20'] = df['Close'].rolling(window=20).mean()
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# Multivariate LSTM Data Preparation
df_clean = df[['Close', 'MA20', 'RSI']].dropna()
scaler2 = MinMaxScaler()
scaled_data2 = scaler2.fit_transform(df_clean)

def create_multifeature_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

X2, y2 = create_multifeature_dataset(scaled_data2, 60)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, shuffle=False)

model2 = Sequential()
model2.add(LSTM(50, return_sequences=True, input_shape=(X_train2.shape[1], X_train2.shape[2])))
model2.add(LSTM(50))
model2.add(Dense(1))
model2.compile(optimizer='adam', loss='mean_squared_error')
model2.load_weights('model_weights2.h5')

predicted2 = model2.predict(X_test2)
predicted_full = np.concatenate([predicted2, np.zeros((predicted2.shape[0], 2))], axis=1)
actual_full = np.concatenate([y_test2.reshape(-1, 1), np.zeros((y_test2.shape[0], 2))], axis=1)
predicted_prices2 = scaler2.inverse_transform(predicted_full)[:, 0]
actual_prices2 = scaler2.inverse_transform(actual_full)[:, 0]
r2_2 = r2_score(actual_prices2, predicted_prices2) * 100

# Streamlit Dashboard
st.title("\U0001F4CA AAPL Stock Forecasting Dashboard")
st.subheader("\U0001F4C8 AAPL Closing Price")
st.line_chart(df.set_index("Date")["Close"])

st.subheader("\U0001F4CA Closing Price with MA20")
fig1, ax1 = plt.subplots()
ax1.plot(df['Date'], df['Close'], label='Close')
ax1.plot(df['Date'], df['MA20'], label='MA20')
ax1.legend()
st.pyplot(fig1)

st.subheader("\U0001F4CA RSI (Relative Strength Index)")
fig2, ax2 = plt.subplots()
ax2.plot(df['Date'], df['RSI'], label='RSI', color='purple')
ax2.axhline(70, linestyle='--', color='red')
ax2.axhline(30, linestyle='--', color='green')
ax2.legend()
st.pyplot(fig2)

st.header("\U0001F9E0 Univariate LSTM (Close Price Only)")
if st.button("\u25B6 Show Univariate Predictions"):
    fig3, ax3 = plt.subplots()
    ax3.plot(actual_prices, label="Actual")
    ax3.plot(predicted_prices, label="Predicted")
    ax3.set_title("Univariate: Actual vs Predicted")
    ax3.legend()
    st.pyplot(fig3)
    st.metric("Univariate R² Score", f"{r2:.2f}%")

st.header("\U0001F9E0 Multivariate LSTM (Close + MA20 + RSI)")
if st.button("\u25B6 Show Multivariate Predictions"):
    fig4, ax4 = plt.subplots()
    ax4.plot(actual_prices2, label="Actual")
    ax4.plot(predicted_prices2, label="Predicted")
    ax4.set_title("Multivariate: Actual vs Predicted")
    ax4.legend()
    st.pyplot(fig4)
    st.metric("Multivariate R² Score", f"{r2_2:.2f}%")
