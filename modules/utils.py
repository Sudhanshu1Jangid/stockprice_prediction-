import yfinance as yf

def validate_ticker(ticker):
    """
    Validate if a ticker symbol is valid by checking historical data availability
    """
    try:
        data = yf.Ticker(ticker).history(period="5d")
        return not data.empty
    except Exception as e:
        print(f"Validation error for {ticker}: {e}")
        return False
