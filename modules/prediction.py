import numpy as np
import pandas as pd
from modules.stock_data import fetch_stock_data
from modules.analysis import calculate_technical_indicators
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

def fetch_index_data(symbol: str, period: str = '2y') -> pd.DataFrame:
    """
    Fetch index data (e.g., S&P 500) using existing fetch_stock_data.
    """
    df = fetch_stock_data(symbol, period=period)
    return df[['Close']].rename(columns={'Close': f'{symbol}_Close'})

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators and additional engineered features.
    """
    df = calculate_technical_indicators(df)
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
    df['Momentum']   = df['Close'] - df['Close'].shift(10)
    return df.dropna()

def prepare_data(ticker: str,
                 index_symbol: str = '^GSPC',
                 period: str = '2y',
                 feature_cols: list = None) -> pd.DataFrame:
    """
    Fetch stock & index data, compute and align features.
    """
    df_stock = fetch_stock_data(ticker, period=period)
    df_stock = create_features(df_stock)

    df_index = fetch_index_data(index_symbol, period)
    df_index['Index_Return'] = df_index[f'{index_symbol}_Close'].pct_change()
    df_index = df_index.dropna()

    df = df_stock.join(df_index, how='inner')

    if feature_cols is None:
        feature_cols = [
            'Close','MA20','MA50','RSI','MACD','Volume MA',
            'Volatility','Momentum',
            f'{index_symbol}_Close','Index_Return'
        ]
    return df[feature_cols].dropna()

def create_sequences(data: np.ndarray, seq_length: int):
    """
    Build input sequences and labels for LSTM.
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])  # target = Close
    return np.array(X), np.array(y)

def predict_stock_price(
    ticker: str,
    seq_length: int = 60,
    test_size: float = 0.2,
    epochs: int = 100,
    batch_size: int = 32,
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-4,
    index_symbol: str = '^GSPC',
    period: str = '2y'
) -> dict:
    """
    Predict next-day closing price using enhanced multivariate LSTM.
    Returns:
      - predicted_price: float
      - forecast_df: pd.DataFrame (Actual vs Predicted)
      - history: Keras History object
      - metrics: dict of MAE, RMSE, R2
    """
    # 1. Prepare data
    df = prepare_data(ticker, index_symbol, period)
    values = df.values
    scaler = MinMaxScaler((0,1))
    scaled = scaler.fit_transform(values)

    # 2. Create sequences
    X, y = create_sequences(scaled, seq_length)

    # 3. Train/test split
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 4. Build model
    model = Sequential([
        LSTM(128, return_sequences=True,
             input_shape=(seq_length, X.shape[2]),
             kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        LSTM(64, kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 5. Callbacks
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    # 6. Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es, rl],
        verbose=1
    )

    # 7. Evaluate
    y_pred_s = model.predict(X_test)
    feats = X.shape[2]
    y_pred = scaler.inverse_transform(
        np.hstack([y_pred_s, np.zeros((len(y_pred_s), feats-1))])
    )[:,0]
    y_true = scaler.inverse_transform(
        np.hstack([y_test.reshape(-1,1), np.zeros((len(y_test), feats-1))])
    )[:,0]
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    metrics = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    # 8. Next-day prediction
    last_seq = scaled[-seq_length:].reshape((1, seq_length, feats))
    next_pred_s = model.predict(last_seq)
    predicted_price = scaler.inverse_transform(
        np.hstack([next_pred_s, np.zeros((1, feats-1))])
    )[:,0][0]

    # 9. Forecast DataFrame
    all_preds_s = model.predict(X)
    all_preds = scaler.inverse_transform(
        np.hstack([all_preds_s, np.zeros((len(all_preds_s), feats-1))])
    )[:,0]
    forecast_df = pd.DataFrame({
        'Actual': df['Close'].iloc[seq_length:],
        'Predicted': all_preds
    }, index=df.index[seq_length:])

    return {
        'predicted_price': predicted_price,
        'forecast_df': forecast_df,
        'history': history,
        'metrics': metrics
    }
