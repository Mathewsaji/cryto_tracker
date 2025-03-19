import streamlit as st
import requests
import pandas as pd
import time
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# Function to fetch real-time crypto prices
def get_crypto_price(crypto, currency='usd'):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto}&vs_currencies={currency}"
    response = requests.get(url)
    data = response.json()
    return data.get(crypto, {}).get(currency, None)

# Function to fetch historical price data
def get_historical_data(crypto, currency='usd', days=7):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto}/market_chart?vs_currency={currency}&days={days}"
    response = requests.get(url)
    data = response.json()
    if 'prices' in data:
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    return None

# Function to analyze the trend (increase or decrease) based on historical data
def analyze_trend(df):
    # Get the first and last price from the historical data
    first_price = df['price'].iloc[0]
    last_price = df['price'].iloc[-1]

    # Calculate the percentage change
    price_change = ((last_price - first_price) / first_price) * 100

    # Determine whether the price is increasing or decreasing
    if price_change > 0:
        trend = f"📈 The price has increased by {price_change:.2f}%."
        trend_color = 'green'  # Use green color for increase
    elif price_change < 0:
        trend = f"📉 The price has decreased by {abs(price_change):.2f}%."
        trend_color = 'red'  # Use red color for decrease
    else:
        trend = "🔹 The price has remained stable."
        trend_color = 'blue'  # Use blue color for no change

    return trend, trend_color

# Function to predict the next day's price using Linear Regression
def predict_next_day(df):
    # Prepare the data for linear regression
    df['timestamp'] = df['timestamp'].map(pd.Timestamp.timestamp)  # Convert datetime to timestamp
    X = np.array(df['timestamp']).reshape(-1, 1)  # Independent variable: timestamps
    y = np.array(df['price'])  # Dependent variable: prices

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict the next day's price
    next_day_timestamp = pd.Timestamp.now().timestamp() + 86400  # Add 1 day (86400 seconds)
    predicted_price = model.predict([[next_day_timestamp]])
    next_day_date = pd.to_datetime(next_day_timestamp, unit='s')

    return next_day_date, predicted_price[0]

# Streamlit UI
def main():
    st.set_page_config(page_title="Crypto Price Tracker", layout="wide")
    st.title("🚀 Real-Time Crypto Price Tracker")
    
    crypto = st.selectbox("Select Cryptocurrency", ["bitcoin", "ethereum", "dogecoin", "cardano", "solana"])
    currency = st.selectbox("Select Currency", ["usd", "eur", "inr", "gbp", "jpy"])  
    
    st.subheader(f"💰 Current Price of {crypto.capitalize()} in {currency.upper()}")
    
    # Fetch real-time price
    price = get_crypto_price(crypto, currency)
    if price:
        st.metric(label=f"{crypto.capitalize()} Price", value=f"{price} {currency.upper()}")
    else:
        st.error("Failed to fetch crypto prices. Please try again.")
    
    # Fetch historical data
    st.subheader("📈 Price Trends (Last 7 Days)")
    df = get_historical_data(crypto, currency)
    if df is not None:
        # Display historical price trend
        fig = go.Figure()

        # Plot historical prices (past 7 days)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['price'], mode='lines', name=f"{crypto.capitalize()} Price (Historical)", line=dict(color='blue')))

        # Customize layout for historical prices
        fig.update_layout(
            title=f"{crypto.capitalize()} Price Trend (Last 7 Days)",
            xaxis_title="Date",
            yaxis_title=f"Price in {currency.upper()}",
            template="plotly_dark",
            showlegend=True
        )
        # Show the historical price graph
        st.plotly_chart(fig, use_container_width=True)

        # Trend analysis
        trend, trend_color = analyze_trend(df)

        # Display the trend message
        st.markdown(f"<p style='color:{trend_color}; font-size:18px;'>{trend}</p>", unsafe_allow_html=True)
        
        # Predict next day's price
        next_day_date, predicted_price = predict_next_day(df)
        st.write(f"🔮 Predicted Price for {crypto.capitalize()} on {next_day_date.strftime('%Y-%m-%d')}: {predicted_price:.2f} {currency.upper()}")
        
    else:
        st.error("Failed to fetch historical data.")

    # Price Alert System
    st.subheader("🔔 Set Price Alerts")
    alert_price = st.number_input("Enter Alert Price", min_value=0.0, format="%.2f")
    alert_status = st.empty()
    
    if st.button("Set Alert"):
        st.success(f"Alert set for {crypto.capitalize()} when price reaches {alert_price} {currency.upper()}")
        while True:
            current_price = get_crypto_price(crypto, currency)
            if current_price and current_price >= alert_price:
                alert_status.warning(f"🚨 ALERT! {crypto.capitalize()} has reached {alert_price} {currency.upper()}!")
                break
            time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    main()
