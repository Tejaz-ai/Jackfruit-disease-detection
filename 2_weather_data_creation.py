
# create_weather_dataset.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os

def create_synthetic_weather_data(days=365):
    """Create realistic synthetic weather data for demonstration"""
    dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
    
    data = []
    for i, date in enumerate(dates):
        # Base patterns with seasonality
        base_temp = 25 + 8 * np.sin(2 * np.pi * i / 365)  # Seasonal variation
        base_humidity = 60 + 20 * np.sin(2 * np.pi * i / 180)  # Semi-annual variation
        
        # Add daily randomness
        temperature = base_temp + np.random.normal(0, 3)
        humidity = base_humidity + np.random.normal(0, 10)
        humidity = max(30, min(100, humidity))  # Keep within bounds
        
        # Rainfall pattern (monsoon season)
        rainfall = 0
        if 150 <= i % 365 <= 270:  # Monsoon months
            rainfall = np.random.exponential(2)
        
        # Create disease outbreaks based on weather conditions
        outbreak = 0
        if humidity > 85 and temperature > 28 and rainfall > 1:
            outbreak = 1  # High disease risk
        
        data.append({
            'date': date,
            'temperature': round(temperature, 1),
            'humidity': round(humidity, 1),
            'rainfall': round(rainfall, 1),
            'outbreak_occurred': outbreak
        })
    
    return pd.DataFrame(data)

# Create and save dataset
weather_df = create_synthetic_weather_data(365)
weather_df.to_csv('jackfruit_weather_dataset.csv', index=False)
print("Synthetic weather dataset created with shape:", weather_df.shape)