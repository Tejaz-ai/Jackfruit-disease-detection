# fusion_engine.py
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

import os

class MultimodalDiseaseFusion:
    def __init__(self):
        """Initialize fusion system with both trained models"""
        print("Loading CNN Image Model...")
        self.cnn_model = load_model('models/jackfruit_disease_cnn_model.h5')
        
        print("Loading LSTM Weather Model...")
        self.lstm_model = load_model('models/weather_lstm_model.h5')
        
        print("Loading Weather Scaler and Features...")
        self.weather_scaler = joblib.load('weather_data/weather_scaler.pkl')
        self.feature_columns = joblib.load('weather_data/feature_columns.pkl')
        
        self.weather_history = []  # Store recent weather data for sequences
        self.sequence_length = 14  # Same as training
        
        # Class names from your CNN training
        self.class_names = {
            0: 'Algal_Leaf_Spot_of_Jackfruit', 
            1: 'Black_Spot_of_Jackfruit', 
            2: 'Healthy_Leaf_of_Jackfruit'
        }
        
        print("Fusion Engine Initialized Successfully!")
        print("=" * 50)
    
    def analyze_image(self, image_path):
        """Analyze leaf image using CNN model"""
        try:
            # Load and preprocess image
            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            
            # Make prediction
            predictions = self.cnn_model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.class_names[predicted_class_idx]
            confidence = predictions[0][predicted_class_idx] * 100
            
            # Get all class probabilities
            all_predictions = {}
            for idx, class_name in self.class_names.items():
                all_predictions[class_name] = predictions[0][idx] * 100
            
            return {
                'diagnosis': predicted_class,
                'confidence': round(confidence, 2),
                'all_predictions': all_predictions,
                'timestamp': datetime.now(),
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'diagnosis': 'Error',
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now(),
                'status': 'error'
            }
    
    def predict_weather_risk(self, current_weather_data):
        """Predict disease risk from weather using LSTM model"""
        try:
            # Prepare current weather data for prediction
            current_features = []
            for col in self.feature_columns:
                if col in current_weather_data:
                    current_features.append(current_weather_data[col])
                else:
                    current_features.append(0.0)  # Default value if missing
            
            current_features = np.array(current_features).reshape(1, -1)
            scaled_features = self.weather_scaler.transform(current_features)
            
            # Add to history for sequence
            self.weather_history.append(scaled_features.flatten())
            if len(self.weather_history) > self.sequence_length:
                self.weather_history.pop(0)
            
            # Only predict if we have enough historical data
            if len(self.weather_history) == self.sequence_length:
                sequence = np.array(self.weather_history).reshape(1, self.sequence_length, -1)
                risk_probability = self.lstm_model.predict(sequence, verbose=0)[0][0]
                
                # Use adjusted threshold for better outbreak detection
                risk_level = "Low"
                if risk_probability > 0.3:  # Lower threshold
                    risk_level = "High"
                elif risk_probability > 0.15:
                    risk_level = "Medium"
                
                return {
                    'risk_probability': round(float(risk_probability), 4),
                    'risk_level': risk_level,
                    'timestamp': datetime.now(),
                    'status': 'success'
                }
            else:
                return {
                    'risk_level': 'Insufficient data',
                    'risk_probability': 0.0,
                    'data_points': len(self.weather_history),
                    'required_points': self.sequence_length,
                    'timestamp': datetime.now(),
                    'status': 'warning'
                }
                
        except Exception as e:
            return {
                'risk_level': 'Error',
                'risk_probability': 0.0,
                'error': str(e),
                'timestamp': datetime.now(),
                'status': 'error'
            }
    
    def fuse_predictions(self, image_analysis, weather_prediction):
        """Intelligently fuse image and weather predictions"""
        
        if image_analysis['status'] != 'success':
            return {
                'alert_level': 'ERROR',
                'message': f"Image analysis failed: {image_analysis.get('error', 'Unknown error')}",
                'recommended_action': 'Check image file and try again',
                'timestamp': datetime.now()
            }
        
        if weather_prediction['status'] not in ['success', 'warning']:
            return {
                'alert_level': 'ERROR',
                'message': f"Weather prediction failed: {weather_prediction.get('error', 'Unknown error')}",
                'recommended_action': 'Check weather data and try again',
                'timestamp': datetime.now()
            }
        
        image_diagnosis = image_analysis['diagnosis']
        image_confidence = image_analysis['confidence']
        weather_risk = weather_prediction['risk_level']
        weather_confidence = weather_prediction['risk_probability']
        
        # Fusion Logic Rules
        if weather_risk == "High" and image_diagnosis != "Healthy_Leaf_of_Jackfruit":
            alert_level = "CRITICAL"
            message = f"🚨 CONFIRMED OUTBREAK: {image_diagnosis} detected during high-risk weather conditions!"
            action = "Immediate treatment required. Isolate affected plants."
        
        elif weather_risk == "High" and image_diagnosis == "Healthy_Leaf_of_Jackfruit":
            alert_level = "WARNING"
            message = "⚠️ HIGH RISK: Weather conditions perfect for disease outbreak!"
            action = "Increase monitoring frequency. Apply preventive fungicide."
        
        elif weather_risk == "Medium" and image_diagnosis != "Healthy_Leaf_of_Jackfruit":
            alert_level = "WARNING"
            message = f"⚠️ DISEASE DETECTED: {image_diagnosis} found. Conditions may worsen."
            action = "Begin treatment. Monitor weather changes."
        
        elif image_diagnosis != "Healthy_Leaf_of_Jackfruit":
            alert_level = "ALERT"
            message = f"🔔 DISEASE DETECTED: {image_diagnosis} found under normal conditions."
            action = "Standard treatment recommended. Monitor plant health."
        
        else:
            alert_level = "INFO"
            message = "✅ Conditions normal. No immediate threat detected."
            action = "Continue regular monitoring practices."
        
        return {
            'alert_level': alert_level,
            'message': message,
            'recommended_action': action,
            'image_analysis': image_analysis,
            'weather_prediction': weather_prediction,
            'fusion_timestamp': datetime.now(),
            'status': 'success'
        }
    
    def process_complete_analysis(self, image_path, current_weather_data):
        """Complete multimodal analysis pipeline"""
        print(f"\n🔍 Analyzing image: {image_path}")
        image_result = self.analyze_image(image_path)
        
        print(f"🌤️  Analyzing weather data...")
        weather_result = self.predict_weather_risk(current_weather_data)
        
        print("🧠 Fusing predictions...")
        fused_result = self.fuse_predictions(image_result, weather_result)
        
        return fused_result

# Create global fusion engine instance
fusion_engine = MultimodalDiseaseFusion()