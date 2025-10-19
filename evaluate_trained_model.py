# evaluate_trained_model.py
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from keras.models import load_model

print("Evaluating Trained LSTM Model...")
print("=" * 50)

# Load the trained model
print("Loading trained model...")
model = load_model('models/weather_lstm_model.h5')
print("Model loaded successfully!")

# Load the scaler and feature columns
scaler = joblib.load('weather_data/weather_scaler.pkl')
feature_columns = joblib.load('weather_data/feature_columns.pkl')
print("Scaler and feature columns loaded!")

# ======================
# RECREATE TEST DATA
# ======================
print("\nRecreating test data from original CSV...")

def load_and_prepare_data(csv_file_path):
    """Load and prepare the Pune weather data"""
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Original data shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_disease_risk_labels(df):
    """Create synthetic disease outbreak labels"""
    df['outbreak_occurred'] = 0
    high_risk_conditions = (
        (df['humidity'] > 80) & 
        (df['tempC'].between(25, 32)) & 
        (df['precipMM'] > 1.0)
    )
    df.loc[high_risk_conditions, 'outbreak_occurred'] = 1
    return df

def create_time_series_features(df):
    """Create time-series features"""
    df = df.reset_index(drop=True)
    
    # Create rolling features
    for window in [3, 7, 14]:
        df[f'temp_avg_{window}d'] = df['tempC'].rolling(window=window).mean()
        df[f'humidity_avg_{window}d'] = df['humidity'].rolling(window=window).mean()
        df[f'precip_sum_{window}d'] = df['precipMM'].rolling(window=window).sum()
    
    df['temp_change_24h'] = df['tempC'].diff()
    df['humidity_change_24h'] = df['humidity'].diff()
    df['temp_humidity_index'] = df['tempC'] * (df['humidity'] / 100)
    df['consecutive_high_humidity'] = (df['humidity'] > 75).astype(int)
    
    return df.dropna()

def prepare_lstm_data(df, sequence_length=14, test_size=0.2):
    """Prepare data for LSTM evaluation"""
    available_columns = [col for col in feature_columns if col in df.columns]
    print(f"Using features: {available_columns}")
    
    target_column = 'outbreak_occurred'
    
    # Scale features using the pre-fitted scaler
    scaled_features = scaler.transform(df[available_columns])
    
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
    
    print(f"Recreated test data: X_test {X_test.shape}, y_test {y_test.shape}")
    return X_test, y_test

# Load original data
csv_file_path = 'weather_data/pune_weather_data.csv'
weather_df = load_and_prepare_data(csv_file_path)

if weather_df is None:
    print("Failed to load data. Exiting.")
    exit()

# Recreate the same preprocessing pipeline
weather_df = create_disease_risk_labels(weather_df)
weather_df = create_time_series_features(weather_df)

# Get test data
X_test, y_test = prepare_lstm_data(weather_df)

# ======================
# RECREATE TRAINING HISTORY PLOTS
# ======================
def create_training_history_plots():
    """Recreate training history plots based on your training output"""
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Based on your training output (30 epochs total)
    epochs = range(1, 31)
    
    # Training accuracy (approximated from your output)
    train_accuracy = [
        0.9927, 0.9937, 0.9936, 0.9937, 0.9940, 0.9940, 0.9940, 0.9941, 0.9942, 0.9942,
        0.9942, 0.9943, 0.9945, 0.9947, 0.9945, 0.9944, 0.9947, 0.9946, 0.9946, 0.9947,
        0.9947, 0.9945, 0.9950, 0.9950, 0.9953, 0.9953, 0.9956, 0.9955, 0.9955, 0.9956
    ]
    
    # Validation accuracy (from your output)
    val_accuracy = [
        0.9934, 0.9934, 0.9934, 0.9929, 0.9926, 0.9937, 0.9931, 0.9938, 0.9915, 0.9928,
        0.9935, 0.9935, 0.9938, 0.9928, 0.9941, 0.9935, 0.9938, 0.9935, 0.9938, 0.9941,
        0.9933, 0.9936, 0.9935, 0.9926, 0.9926, 0.9935, 0.9935, 0.9936, 0.9935, 0.9935
    ]
    
    # Training loss (approximated from your output)
    train_loss = [
        0.0312, 0.0205, 0.0191, 0.0181, 0.0173, 0.0168, 0.0167, 0.0163, 0.0161, 0.0158,
        0.0156, 0.0155, 0.0152, 0.0151, 0.0150, 0.0149, 0.0150, 0.0145, 0.0147, 0.0144,
        0.0143, 0.0142, 0.0142, 0.0140, 0.0137, 0.0127, 0.0121, 0.0123, 0.0118, 0.0120
    ]
    
    # Validation loss (from your output)
    val_loss = [
        0.0240, 0.0226, 0.0200, 0.0200, 0.0189, 0.0175, 0.0179, 0.0171, 0.0197, 0.0180,
        0.0167, 0.0187, 0.0167, 0.0180, 0.0165, 0.0176, 0.0167, 0.0167, 0.0162, 0.0160,
        0.0177, 0.0172, 0.0175, 0.0193, 0.0188, 0.0164, 0.0168, 0.0168, 0.0170, 0.0170
    ]
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    # Accuracy plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_accuracy, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_accuracy, 'r-', label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate reduction point (epoch 26)
    plt.subplot(1, 3, 3)
    lr_values = [0.001] * 25 + [0.0002] * 5  # LR reduced at epoch 26
    plt.plot(epochs, lr_values, 'g-', linewidth=2)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=26, color='red', linestyle='--', alpha=0.7, label='LR Reduction')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/lstm_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Training history plots recreated!")

def evaluate_lstm_model(model, X_test, y_test):
    """Evaluate LSTM model performance"""
    
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print("\nLSTM Model Evaluation:")
    print("=" * 50)
    print(f"Test Accuracy: {np.mean(y_pred.flatten() == y_test):.3f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Outbreak', 'Outbreak']))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Outbreak', 'Outbreak'],
                yticklabels=['No Outbreak', 'Outbreak'])
    plt.title('LSTM Confusion Matrix - Pune Weather Data')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('outputs/lstm_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results to file
    with open('outputs/model_evaluation_results.txt', 'w') as f:
        f.write("LSTM Model Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test Accuracy: {np.mean(y_pred.flatten() == y_test):.3f}\n")
        f.write(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=['No Outbreak', 'Outbreak']))
    
    return y_pred_proba

# Create training history plots
create_training_history_plots()

# Run model evaluation
print("\nRunning model evaluation...")
y_pred_proba = evaluate_lstm_model(model, X_test, y_test)

print(f"\n✅ Evaluation completed successfully!")
print(f"📊 Confusion matrix saved: {os.path.abspath('outputs/lstm_confusion_matrix.png')}")
print(f"📈 Training history saved: {os.path.abspath('outputs/lstm_training_history.png')}")
print(f"📝 Results saved: {os.path.abspath('outputs/model_evaluation_results.txt')}")