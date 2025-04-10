import streamlit as st
import pandas as pd
from modules.stock_data import fetch_stock_data
from modules.prediction import predict_stock_price
from modules.portfolio import Portfolio
from modules.analysis import calculate_technical_indicators
from modules.utils import validate_ticker

# Page configuration
st.set_page_config(page_title="Stock Market Analysis Platform", layout="wide")

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = Portfolio()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Stock Analysis", "Portfolio", "Predictions"])

if page == "Stock Analysis":
    st.title("Stock Analysis")
    
    # Stock ticker input
    ticker = st.text_input("Enter Stock Ticker", "TSLA").upper()
    
    if st.button("Analyze"):
        if validate_ticker(ticker):
            with st.spinner('Fetching stock data...'):
                try:
                    # Fetch stock data
                    df = fetch_stock_data(ticker)
                    
                    # Display stock price chart
                    st.subheader("Stock Price History")
                    st.line_chart(df['Close'])
                    
                    # Calculate and display technical indicators
                    indicators = calculate_technical_indicators(df)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Technical Indicators")
                        st.dataframe(indicators.tail())
                    
                    with col2:
                        st.subheader("Statistics")
                        st.write(f"Current Price: {df['Close'].iloc[-1]:.2f}")
                        st.write(f"50-day MA: {indicators['MA50'].iloc[-1]:.2f}")
                        st.write(f"RSI: {indicators['RSI'].iloc[-1]:.2f}")
                    
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
        else:
            st.error("Please enter a valid ticker symbol")

elif page == "Portfolio":
    st.title("Portfolio Tracking")
    
    # Add stock to portfolio
    col1, col2, col3 = st.columns(3)
    with col1:
        new_ticker = st.text_input("Ticker", "").upper()
    with col2:
        shares = st.number_input("Number of Shares", min_value=0.0, value=0.0)
    with col3:
        if st.button("Add to Portfolio"):
            if validate_ticker(new_ticker) and shares > 0:
                st.session_state.portfolio.add_position(new_ticker, shares)
                st.success(f"Added {shares} shares of {new_ticker}")
            else:
                st.error("Please enter valid ticker and shares")
    
    # Display portfolio
    portfolio_data = st.session_state.portfolio.get_portfolio_status()
    if not portfolio_data.empty:
        st.subheader("Current Holdings")
        st.dataframe(portfolio_data)
        
        # Portfolio value chart
        st.subheader("Portfolio Value Over Time")
        portfolio_history = st.session_state.portfolio.get_portfolio_history()
        st.line_chart(portfolio_history['Total Value'])
    else:
        st.info("Your portfolio is empty. Add some stocks to get started!")

elif page == "Predictions":
    st.title("Stock Price Predictions")
    
    pred_ticker = st.text_input("Enter Stock Ticker for Prediction", "AAPL").upper()
    
    if st.button("Generate Prediction"):
        if validate_ticker(pred_ticker):
            with st.spinner('Generating prediction...'):
                try:
                    # Fetch data and make prediction
                    prediction_data = predict_stock_price(pred_ticker)

                    st.subheader("Next Day Price Prediction")
                    st.write(f"Predicted Price: {prediction_data['predicted_price']:.2f}")
                    st.write(f"Confidence Interval: {prediction_data['lower_bound']:.2f} - {prediction_data['upper_bound']:.2f}")
                    
                    # Display prediction chart
                    st.subheader("Historical Prices and Prediction")
                    st.line_chart(prediction_data['forecast_df'])
                    
                except Exception as e:
                    st.error(f"Error generating prediction: {str(e)}")
        else:
            st.error("Please enter a valid ticker symbol")
