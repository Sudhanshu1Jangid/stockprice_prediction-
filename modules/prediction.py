import os
import numpy as np
import pandas as pd
from modules.stock_data import fetch_stock_data
from modules.analysis import calculate_technical_indicators
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Directory to store per‐ticker models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def fetch_index_data(symbol: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch index (e.g. S&P 500) closing prices via your existing data module.
    """
    df = fetch_stock_data(symbol, period=period)
    return df[["Close"]].rename(columns={"Close": f"{symbol}_Close"})

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute standard technical indicators plus engineered features.
    """
    df = calculate_technical_indicators(df)
    # Engineered features:
    df["Volatility"] = df["Close"].pct_change().rolling(window=20).std()
    df["Momentum"]   = df["Close"] - df["Close"].shift(10)
    # Calendar features:
    df["DayOfWeek"] = df.index.dayofweek / 6
    df["Month"]     = (df.index.month - 1) / 11
    return df.dropna()

def prepare_data(ticker: str,
                 index_symbol: str = "^GSPC",
                 period: str = "2y") -> tuple[pd.DataFrame, np.ndarray, MinMaxScaler]:
    """
    Fetch & merge stock + index; compute features; scale per‐feature.
    Returns:
      - df: DataFrame of raw features (for indexing)
      - scaled: numpy array of shape (n_samples, n_features)
      - scaler: fitted MinMaxScaler
    """
    # 1. fetch raw
    df_stock = fetch_stock_data(ticker, period=period)
    df_stock = create_features(df_stock)

    df_index = fetch_index_data(index_symbol, period=period)
    df_index["Index_Return"] = df_index[f"{index_symbol}_Close"].pct_change()
    df_index = df_index.dropna()

    # 2. align on date
    df = df_stock.join(df_index, how="inner").dropna()
    if df.empty:
        raise ValueError(f"Not enough overlapping data for {ticker} and {index_symbol}.")

    # 3. select features
    feature_cols = [
        "Close","MA20","MA50","RSI","MACD","Volume MA",
        "Volatility","Momentum","DayOfWeek","Month",
        f"{index_symbol}_Close","Index_Return"
    ]
    df = df[feature_cols].dropna()

    # 4. per‐feature MinMax scaling
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)

    return df, scaled, scaler

def create_sequences(data: np.ndarray, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Turn scaled feature matrix into (X,y) for LSTM.
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i,0])  # predict the ‘Close’ column
    return np.array(X), np.array(y)

def build_model(input_shape: tuple[int,int], dropout_rate=0.3, l2_reg=1e-4):
    """
    Construct and compile the LSTM model.
    """
    model = Sequential([
        LSTM(128, return_sequences=True,
             input_shape=input_shape,
             kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),

        LSTM(64, kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(32, activation="relu", kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),

        Dense(1, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def predict_stock_price(
    ticker: str,
    seq_length: int = 60,
    test_size: float = 0.2,
    epochs: int = 50,
    batch_size: int = 32,
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-4,
    index_symbol: str = "^GSPC",
    period: str = "2y"
) -> dict:
    """
    For a given ticker:
      - trains or loads a per‐ticker LSTM model
      - returns next‐day forecast, historical vs predicted DF, and metrics.
    """
    # 1. prepare and split data
    df, scaled, scaler = prepare_data(ticker, index_symbol, period)
    if len(scaled) <= seq_length:
        raise ValueError(f"Not enough data points ({len(scaled)}) for seq_length={seq_length} for ticker {ticker}.")

    X, y = create_sequences(scaled, seq_length)
    n_samples, _, n_feats = X.shape
    split = int(n_samples * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 2. model persistence
    model_path = os.path.join(MODEL_DIR, f"{ticker.replace('^','')}_lstm.h5")
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = build_model((seq_length, n_feats), dropout_rate, l2_reg)
        es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        rl = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es, rl],
            verbose=1
        )
        model.save(model_path)

    # 3. evaluate on test set
    y_pred_s = model.predict(X_test)
    pad = np.zeros((len(y_pred_s), n_feats-1))
    y_pred = scaler.inverse_transform(np.hstack([y_pred_s, pad]))[:,0]
    y_true = scaler.inverse_transform(np.hstack([y_test.reshape(-1,1), pad]))[:,0]

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    # 4. next‐day prediction
    last_seq = scaled[-seq_length:].reshape((1, seq_length, n_feats))
    next_s = model.predict(last_seq)
    predicted_price = scaler.inverse_transform(np.hstack([next_s, np.zeros((1,n_feats-1))]))[:,0][0]

    # 5. historic forecast series
    all_s = model.predict(X)
    all_preds = scaler.inverse_transform(np.hstack([all_s, np.zeros((len(all_s),n_feats-1))]))[:,0]
    forecast_df = pd.DataFrame({
        "Actual": df["Close"].iloc[seq_length:].values,
        "Predicted": all_preds
    }, index=df.index[seq_length:])

    return {
        "predicted_price": predicted_price,
        "forecast_df": forecast_df,
        "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2}
    }
