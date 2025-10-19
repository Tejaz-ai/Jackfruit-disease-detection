import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
# weather_feature_engineering.py
def create_time_series_features(df, window_size=7):
    """Create time-series features for LSTM training"""
    
    # Create rolling features
    for window in [3, 7, 14]:
        df[f'temp_avg_{window}d'] = df['temperature'].rolling(window=window).mean()
        df[f'humidity_avg_{window}d'] = df['humidity'].rolling(window=window).mean()
        df[f'rainfall_sum_{window}d'] = df['rainfall'].rolling(window=window).sum()
    
    # Rate of change features
    df['temp_change_24h'] = df['temperature'].diff()
    df['humidity_change_24h'] = df['humidity'].diff()
    
    # Interaction features
    df['temp_humidity_index'] = df['temperature'] * (df['humidity'] / 100)
    
    # Count consecutive humid days
    df['consecutive_high_humidity'] = (df['humidity'] > 80).astype(int)
    for i in range(1, 8):
        df[f'humidity_high_{i}d_ago'] = df['consecutive_high_humidity'].shift(i)
    
    # Drop initial NaN values
    df = df.dropna()
    
    return df

# Load and process data
weather_df = pd.read_csv('jackfruit_weather_dataset.csv')
weather_df = create_time_series_features(weather_df)
print("Feature engineering completed. Dataset shape:", weather_df.shape)