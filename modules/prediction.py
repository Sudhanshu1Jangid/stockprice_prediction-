import numpy as np
import pandas as pd
from modules.stock_data import fetch_stock_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def create_sequences(data: np.ndarray, seq_length: int):
    """
    Builds input sequences and corresponding labels for LSTM.
    Args:
        data (np.ndarray): Scaled data array of shape (n_samples, n_features)
        seq_length (int): Number of timesteps in each input sequence
    Returns:
        X, y arrays for model training
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i, 0])  # predict closing price
    return np.array(X), np.array(y)


def predict_stock_price(
    ticker: str,
    seq_length: int = 60,
    test_size: float = 0.2,
    epochs: int = 100,
    batch_size: int = 32,
    dropout_rate: float = 0.3
):
    """
    Predict next-day closing price using an optimized LSTM network.
    Returns:
      - predicted_price: float
      - forecast_df: pd.DataFrame with actual vs. predicted series
      - metrics: dict containing MAE, RMSE, R2 on test set
    """
    # 1. Fetch data
    df = fetch_stock_data(ticker)
    df = df[['Close']].dropna()

    # 2. Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df.values)

    # 3. Create sequences
    X, y = create_sequences(scaled, seq_length)

    # 4. Split into train/test
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 5. Build LSTM model
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_length, X.shape[2])),
        BatchNormalization(),
        Dropout(dropout_rate),
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 6. Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    # 7. Train
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # 8. Evaluate on test set
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(
        np.hstack([y_pred_scaled, np.zeros((y_pred_scaled.shape[0], X.shape[2]-1))])
    )[:, 0]
    y_true = scaler.inverse_transform(
        np.hstack([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], X.shape[2]-1))])
    )[:, 0]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # 9. Predict next-day price
    last_seq = scaled[-seq_length:]
    last_seq = last_seq.reshape((1, seq_length, X.shape[2]))
    pred_scaled = model.predict(last_seq)
    predicted_price = float(scaler.inverse_transform(
        np.hstack([pred_scaled, np.zeros((pred_scaled.shape[0], X.shape[2]-1))])
    )[:, 0])

    # 10. Generate full forecast for plotting
    preds_scaled = model.predict(X)
    preds = scaler.inverse_transform(
        np.hstack([preds_scaled, np.zeros((preds_scaled.shape[0], X.shape[2]-1))])
    )[:, 0]

    forecast_index = df.index[seq_length:]
    forecast_df = pd.DataFrame({
        'Actual': df['Close'].iloc[seq_length:].values,
        'Predicted': preds
    }, index=forecast_index)

    metrics = {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2)
    }

    return {
        'predicted_price': predicted_price,
        'forecast_df': forecast_df,
        'history': history,
        'metrics': metrics
    }
