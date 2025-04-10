import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_data(ticker, period='1y'):
    """
    Fetch stock data from Yahoo Finance
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            raise Exception("No data available for this ticker")
            
        return df
    except Exception as e:
        raise Exception(f"Error fetching stock data: {str(e)}")

def get_current_price(ticker):
    """
    Get the current price for a stock
    """
    try:
        stock = yf.Ticker(ticker)
        return stock.info['regularMarketPrice']
    except Exception as e:
        raise Exception(f"Error fetching current price: {str(e)}")
