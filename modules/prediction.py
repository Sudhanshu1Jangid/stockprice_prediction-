import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from modules.stock_data import fetch_stock_data
from sklearn.utils import resample


def create_features(df):
    """Create technical features for prediction"""
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['Daily_Return'] = df['Close'].pct_change()
    df = df.dropna()
    return df

def calculate_rsi(prices, period=14):
    """Calculate RSI technical indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def predict_stock_price(ticker):
    """
    Generate stock price prediction for the next day
    """
    try:
        # Fetch historical data
        df = fetch_stock_data(ticker)
        
        # Create features
        df = create_features(df)
        
        # Prepare features for prediction
        feature_columns = ['MA5', 'MA20', 'RSI', 'Daily_Return']
        X = df[feature_columns]
        y = df['Close']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make prediction
        last_features = X.iloc[-1:].values
        predicted_price = model.predict(last_features)[0]
        
        # Calculate confidence interval
        predictions = []
        for i in range(100):
            X_sample, y_sample = resample(X_train, y_train, random_state=i)
            temp_model = RandomForestRegressor(n_estimators=100, random_state=i)
            temp_model.fit(X_sample, y_sample)
            boot_pred = temp_model.predict(last_features)[0]
            predictions.append(boot_pred)

        lower_bound = np.percentile(predictions, 5)
        upper_bound = np.percentile(predictions, 95)
        # Prepare forecast dataframe
        forecast_df = pd.DataFrame({
            'Actual': df['Close'],
            'Predicted': model.predict(X)
        })
        
        return {
            'predicted_price': predicted_price,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'forecast_df': forecast_df,
        }
        
    except Exception as e:
        raise Exception(f"Error generating prediction: {str(e)}")
