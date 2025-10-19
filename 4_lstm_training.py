# 4_lstm_training.py
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Keras imports
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set random seed for reproducibility
np.random.seed(42)

print("Starting LSTM Weather Model Training with Pune Data...")
print("=" * 60)

# ======================
# 1. LOAD AND PREPARE PUNE WEATHER DATA
# ======================
print("1. Loading and preparing Pune weather data...")

def load_and_prepare_data(csv_file_path):
    """Load and prepare the Pune weather data"""
    try:
        # Load the CSV data
        df = pd.read_csv(csv_file_path)
        print(f"Original data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Display basic info
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        # Check for missing values
        print(f"\nMissing values:")
        print(df.isnull().sum())
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_disease_risk_labels(df):
    """
    Create synthetic disease outbreak labels based on weather conditions
    Jackfruit diseases thrive in: High humidity (>80%), warm temps (25-32°C), rainfall
    """
    # Create outbreak labels based on weather conditions
    df['outbreak_occurred'] = 0  # Default: no outbreak
    
    # Conditions that favor disease outbreak (using your actual column names)
    high_risk_conditions = (
        (df['humidity'] > 80) & 
        (df['tempC'].between(25, 32)) & 
        (df['precipMM'] > 1.0)
    )
    
    df.loc[high_risk_conditions, 'outbreak_occurred'] = 1
    
    print(f"\nOutbreak statistics:")
    print(f"Total records: {len(df)}")
    print(f"Outbreak cases: {df['outbreak_occurred'].sum()}")
    print(f"Outbreak percentage: {df['outbreak_occurred'].mean()*100:.2f}%")
    
    return df

def create_time_series_features(df):
    """Create time-series features for LSTM training using your actual column names"""
    
    # Sort by date if available, otherwise by index
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    
    # Create rolling features using your actual column names
    for window in [3, 7, 14]:
        df[f'temp_avg_{window}d'] = df['tempC'].rolling(window=window).mean()
        df[f'humidity_avg_{window}d'] = df['humidity'].rolling(window=window).mean()
        df[f'precip_sum_{window}d'] = df['precipMM'].rolling(window=window).sum()
    
    # Rate of change features
    df['temp_change_24h'] = df['tempC'].diff()
    df['humidity_change_24h'] = df['humidity'].diff()
    
    # Interaction features
    df['temp_humidity_index'] = df['tempC'] * (df['humidity'] / 100)
    
    # Count consecutive humid days
    df['consecutive_high_humidity'] = (df['humidity'] > 75).astype(int)
    
    # Drop initial NaN values
    df = df.dropna()
    
    print(f"After feature engineering: {df.shape}")
    return df

def prepare_lstm_data(df, sequence_length=14, test_size=0.2):
    """Prepare data for LSTM training"""
    
    # Select relevant features from your actual dataset
    feature_columns = [
        'tempC', 'humidity', 'precipMM', 
        'temp_avg_7d', 'humidity_avg_7d', 'temp_humidity_index'
    ]
    
    # Make sure all required columns exist
    available_columns = [col for col in feature_columns if col in df.columns]
    print(f"Using features: {available_columns}")
    
    target_column = 'outbreak_occurred'
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[available_columns])
    
    # Create sequences for LSTM
    X, y = [], []
    for i in range(len(scaled_features) - sequence_length):
        X.append(scaled_features[i:i+sequence_length])
        y.append(df[target_column].iloc[i+sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )
    
    print(f"Training data: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Test data: X_test {X_test.shape}, y_test {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler, available_columns

# Load the Pune weather data
csv_file_path = 'weather_data/pune_weather_data.csv'
weather_df = load_and_prepare_data(csv_file_path)

if weather_df is None:
    print("Failed to load data. Exiting.")
    exit()

# Create disease risk labels
weather_df = create_disease_risk_labels(weather_df)

# Apply feature engineering
weather_df = create_time_series_features(weather_df)

# Prepare data for LSTM
X_train, X_test, y_train, y_test, scaler, feature_columns = prepare_lstm_data(weather_df)

# Save the scaler and feature columns for later use
os.makedirs('weather_data', exist_ok=True)
joblib.dump(scaler, 'weather_data/weather_scaler.pkl')
joblib.dump(feature_columns, 'weather_data/feature_columns.pkl')
print("Scaler and feature columns saved!")

# ======================
# 2. BUILD LSTM MODEL
# ======================
print("\n2. Building LSTM model...")

def build_lstm_model(input_shape):
    """Build LSTM model for weather prediction"""
    
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, input_shape=input_shape)),
        Dropout(0.3),
        
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        
        Dense(1, activation='sigmoid')
    ])
    
    return model

# Build model
input_shape = (X_train.shape[1], X_train.shape[2])
lstm_model = build_lstm_model(input_shape)

# Compile model
lstm_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

# Display model architecture
print("Model Summary:")
lstm_model.summary()

# ======================
# 3. TRAIN THE MODEL
# ======================
print("\n3. Training LSTM model...")

history = lstm_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=5)
    ]
)

# Save model
os.makedirs('models', exist_ok=True)
lstm_model.save('models/weather_lstm_model.h5')
print("LSTM model trained and saved successfully!")

# ======================
# 4. EVALUATE THE MODEL
# ======================
print("\n4. Evaluating model performance...")
def evaluate_lstm_model(model, X_test, y_test):
    """Evaluate LSTM model performance"""
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print("LSTM Model Evaluation:")
    print("=" * 50)
    print(f"Test Accuracy: {np.mean(y_pred.flatten() == y_test):.3f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Outbreak', 'Outbreak']))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Outbreak', 'Outbreak'],
                yticklabels=['No Outbreak', 'Outbreak'])
    plt.title('LSTM Confusion Matrix - Pune Weather Data')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('outputs/lstm_confusion_matrix.png', dpi=300)
    plt.show()
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/lstm_training_history.png', dpi=300)
    plt.show()
    
    return y_pred_proba

# Evaluate model
y_pred_proba = evaluate_lstm_model(lstm_model, X_test, y_test)

print("\n" + "=" * 60)
print("LSTM TRAINING WITH PUNE DATA COMPLETED SUCCESSFULLY!")
print("=" * 60)
