import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ----------------- DATA LOADING -----------------
@st.cache_data
def load_stock_data(tickers, start="2015-01-01", end=datetime.today().strftime("%Y-%m-%d")):
    data = yf.download(tickers, start=start, end=end)
    close_prices = data['Close']  # Fix: Use 'Close' instead of 'Adj Close'
    return close_prices

tickers = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
stock_data = load_stock_data(tickers)

# ----------------- FEATURE ENGINEERING -----------------
returns = stock_data.pct_change().dropna()
moving_avg = stock_data.rolling(window=20).mean()

# ----------------- APP SECTIONS -----------------
tab1, tab2 = st.tabs(["📈 Stock Prediction (AAPL)", "📊 EDA & Risk Analysis"])

# ----------------- TAB 1: STOCK PREDICTION -----------------
with tab1:
    st.title("📈 Predict Apple Stock Closing Price")
    st.write("We will use a simple linear regression to predict the closing price of Apple stock based on time.")

    # Prepare data
    df = stock_data[['AAPL']].dropna().reset_index()
    df['Date_Ordinal'] = pd.to_datetime(df['Date']).map(datetime.toordinal)

    X = df[['Date_Ordinal']]
    y = df['AAPL']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(df['Date'], df['AAPL'], label='Actual Price')
    ax.plot(df.loc[y_test.index, 'Date'], prediction, label='Predicted Price', linestyle='--')
    ax.set_title("Apple Stock Price Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    st.write("R² Score:", model.score(X_test, y_test))

# ----------------- TAB 2: EDA -----------------
with tab2:
    st.title("📊 EDA and Risk Analysis")

    st.subheader("1. 📅 Change in Price Over Time")
    fig1, ax1 = plt.subplots()
    stock_data.plot(ax=ax1)
    ax1.set_title("Stock Prices Over Time")
    ax1.set_ylabel("Adjusted Close Price")
    st.pyplot(fig1)

    st.subheader("2. 📈 Average Daily Return")
    avg_returns = returns.mean()
    st.bar_chart(avg_returns)

    st.subheader("3. 🧮 20-Day Moving Average")
    st.line_chart(moving_avg)

    st.subheader("4. 📊 Correlation Between Stocks")
    corr = returns.corr()
    fig2, ax2 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    st.subheader("5. ⚠️ Value at Risk (95% confidence)")
    var_95 = returns.std() * 1.65  # Assuming normal distribution
    st.write("Estimated daily loss (Value at Risk - 95% confidence):")
    st.table(var_95.apply(lambda x: f"{x*100:.2f}%"))

    st.subheader("6. 🤖 Predicting Apple Stock Behavior")
    st.write("""
        We used **Linear Regression** to model Apple's stock price based on time. 
        For better accuracy, we could use:
        - LSTM/GRU (deep learning for time series)
        - ARIMA models
        - External data (macro indicators, news, etc.)
    """)