# AIML_Internship_Project
# Stock Price Prediction
The main goal of this project was to analyze historical stock data and predict future stock prices using deep learning models.

---

## Workflow & Explanation

### 1. Data Collection
- Used `yfinance` to download stock data (e.g., AAPL) from **2015 to 2024**.
- Fetched fields like `Open`, `High`, `Low`, `Close`, `Volume`, and `Date`.

### 2. Data Preprocessing
- Resampled/cleaned the data and set `Date` as index.
- Scaled the data using `MinMaxScaler` to normalize input features.
- Converted the time series data into sequences suitable for LSTM.


### 3. Model Building - LSTM
- Defined a sequential LSTM model using Keras.
- Input shape: sequences of stock prices for `n` days to predict the next day's price.
- Compiled with `mean squared error (MSE)` loss and `Adam` optimizer.

### 4. Model Training & Evaluation
- Split data into training and testing sets (e.g., 80-20).
- Trained the model over multiple epochs.
- Evaluated using RMSE and plotted **actual vs predicted** prices.

### 5. Forecasting
- Predicted future stock prices for a given number of days.
- Plotted future trends along with historical data for comparison.

### 6. Dashboard
- Displayed the dashboard using streamlit.

---


## Results

- The model was able to closely follow the actual price trend. 
- Test predictions plotted alongside actual prices showed good accuracy and trend capture.

---
