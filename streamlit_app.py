# streamlit_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from datetime import datetime
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import gdown
import os

# Page configuration
st.set_page_config(
    page_title="🍈 Jackfruit Disease Detector",
    page_icon="🍈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 3rem;
    }
    .alert-critical {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 8px solid #990000;
    }
    .alert-warning {
        background: linear-gradient(135deg, #ffbb33, #ff8800);
        color: black;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 8px solid #cc7700;
    }
    .alert-info {
        background: linear-gradient(135deg, #00C851, #007e33);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 8px solid #005a24;
    }
    .risk-meter {
        height: 25px;
        background: #ecf0f1;
        border-radius: 12px;
        margin: 1rem 0;
        overflow: hidden;
        border: 2px solid #bdc3c7;
    }
    .risk-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    .risk-low { background: linear-gradient(90deg, #00C851, #007e33); }
    .risk-medium { background: linear-gradient(90deg, #ffbb33, #ff8800); }
    .risk-high { background: linear-gradient(90deg, #ff4444, #cc0000); }
    .severity-high { color: #dc3545; font-weight: bold; font-size: 1.3em; }
    .severity-medium { color: #fd7e14; font-weight: bold; font-size: 1.3em; }
    .severity-low { color: #20c997; font-weight: bold; font-size: 1.3em; }
    .severity-none { color: #28a745; font-weight: bold; font-size: 1.3em; }
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 6px solid #3498db;
    }
    .weather-card {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def download_model_from_drive():
    """Download model from Google Drive if not exists"""
    model_path = 'jackfruit_disease_updated.h5'
    
    if not os.path.exists(model_path):
        st.sidebar.info("📥 Downloading AI model from Google Drive... (This may take 2-3 minutes)")
        
        # REPLACE THIS WITH YOUR ACTUAL GOOGLE DRIVE FILE ID
        file_id = "1NCpByQkEduyaMpKqGE0Nv4E4JCbfQ6Ab"  # ← REPLACE THIS WITH YOUR FILE ID!
        url = f"https://drive.google.com/uc?id={file_id}"
        
        try:
            gdown.download(url, model_path, quiet=False)
            st.sidebar.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Download failed: {e}")
            st.error("""
            **Please check:**
            - Your Google Drive file ID is correct
            - The file is publicly accessible
            - You have stable internet connection
            """)
            st.stop()
    
    return model_path

class MultimodalDiseaseFusion:
    def __init__(self):
        st.sidebar.info("🚀 Loading AI models...")
        
        try:
            # Download and load model from Google Drive
            model_path = download_model_from_drive()
            self.cnn_model = load_model(model_path)
            
            # Try to load LSTM model if available
            try:
                self.lstm_model = load_model('models/weather_lstm_model.h5')
                self.weather_scaler = joblib.load('weather_data/weather_scaler.pkl')
                self.feature_columns = joblib.load('weather_data/feature_columns.pkl')
                self.lstm_loaded = True
            except:
                st.sidebar.warning("⚠️ LSTM model not found, using default weather analysis")
                self.lstm_loaded = False
                self.lstm_model = None
                self.weather_scaler = None
                self.feature_columns = None
                
            self.weather_history = []
            self.sequence_length = 14
            
            # Updated class names for 7 classes
            self.class_names = {
                0: 'Healthy',
                1: 'Leaf Bright', 
                2: 'Leaf Spot',
                3: 'Rhizopus Rot',
                4: 'Algal_Leaf_Spot_of_Jackfruit', 
                5: 'Black_Spot_of_Jackfruit', 
                6: 'Healthy_leaf_of_Jackfruit'
            }
            
            # Enhanced disease information database for 7 classes
            self.disease_info = {
                'Healthy': {
                    'symptoms': 'Vibrant green color, no spots or discoloration, firm texture',
                    'maintenance': 'Continue good agricultural practices, monitor regularly',
                    'recommendation': 'Maintain soil health, provide adequate water and nutrients',
                    'severity': 'None'
                },
                'Leaf Bright': {
                    'symptoms': 'Unusual brightness or glossiness on leaf surface, possible early infection sign',
                    'causes': 'Could be early fungal infection or nutrient imbalance',
                    'treatment': 'Apply preventive fungicides, check nutrient levels',
                    'prevention': 'Regular monitoring, balanced fertilization',
                    'severity': 'Low'
                },
                'Leaf Spot': {
                    'symptoms': 'Circular or irregular spots on leaves, may be brown, black, or yellow',
                    'causes': 'Fungal or bacterial infection, often due to humid conditions',
                    'treatment': 'Apply appropriate fungicides, remove severely infected leaves',
                    'prevention': 'Improve air circulation, avoid overhead watering',
                    'severity': 'Medium'
                },
                'Rhizopus Rot': {
                    'symptoms': 'Soft rot, water-soaked lesions, white fungal growth',
                    'causes': 'Rhizopus fungus, thrives in warm humid conditions',
                    'treatment': 'Remove infected parts, apply fungicides containing captan or thiophanate-methyl',
                    'prevention': 'Proper sanitation, avoid wounding fruits, control humidity',
                    'severity': 'High'
                },
                'Algal_Leaf_Spot_of_Jackfruit': {
                    'symptoms': 'Reddish-brown spots with greenish-gray centers, velvety appearance',
                    'causes': 'Caused by algae (Cephaleuros virescens), thrives in humid conditions',
                    'treatment': 'Apply copper-based fungicides, improve air circulation, remove infected leaves',
                    'prevention': 'Proper spacing, avoid overhead watering, regular pruning',
                    'severity': 'Medium'
                },
                'Black_Spot_of_Jackfruit': {
                    'symptoms': 'Circular black spots with yellow halos, leaves may yellow and drop',
                    'causes': 'Fungal infection (often Colletotrichum species), spreads in warm wet weather',
                    'treatment': 'Apply fungicides containing mancozeb or chlorothalonil, remove infected leaves',
                    'prevention': 'Avoid wetting leaves, proper sanitation, resistant varieties',
                    'severity': 'High'
                },
                'Healthy_leaf_of_Jackfruit': {
                    'symptoms': 'Healthy green coloration, no visible spots or abnormalities',
                    'maintenance': 'Continue current care practices, regular inspection',
                    'recommendation': 'Maintain optimal growing conditions',
                    'severity': 'None'
                }
            }
            st.sidebar.success("✅ Models loaded successfully!")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading models: {str(e)}")
            st.stop()
    
    def is_plant_leaf(self, image):
        """Enhanced plant leaf detection with better validation"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Check image dimensions
            if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                return False, "Image must be RGB color format"
            
            # Check for reasonable leaf-like colors (green dominance)
            hsv_image = Image.fromarray(img_array).convert('HSV')
            hsv_array = np.array(hsv_image)
            
            # Calculate green pixel percentage (hue in green range)
            green_pixels = np.sum((hsv_array[:,:,0] >= 30) & (hsv_array[:,:,0] <= 90))
            total_pixels = hsv_array.shape[0] * hsv_array.shape[1]
            green_percentage = green_pixels / total_pixels
            
            # Check if image has sufficient green content for a leaf
            if green_percentage < 0.1:  # Less than 10% green
                return False, "Image doesn't appear to contain a plant leaf (insufficient green color)"
            
            # Check image size and aspect ratio
            width, height = image.size
            if width < 100 or height < 100:
                return False, "Image resolution too low. Please upload a higher quality image."
            
            return True, "Valid leaf image"
            
        except Exception as e:
            return False, f"Image validation error: {str(e)}"
    
    def analyze_image(self, image):
        try:
            # Enhanced image validation
            is_valid, validation_msg = self.is_plant_leaf(image)
            if not is_valid:
                raise Exception(validation_msg)
            
            # Resize image to match model input size (224x224)
            img = image.resize((224, 224))
            
            # Convert to numpy array with proper dtype
            img_array = np.array(img)
            
            # Check if image has the right number of channels
            if len(img_array.shape) == 2:  # Grayscale image
                img_array = np.stack((img_array,) * 3, axis=-1)
            elif img_array.shape[2] == 4:  # RGBA image
                img_array = img_array[:, :, :3]
            
            # Convert to float32 and normalize
            img_array = img_array.astype(np.float32) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            predictions = self.cnn_model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx] * 100)
            
            # Get all predictions for display
            all_predictions = {}
            for idx, class_name in self.class_names.items():
                all_predictions[class_name] = float(predictions[0][idx] * 100)
            
            return predicted_class, confidence, all_predictions, self.disease_info.get(predicted_class, {})
            
        except Exception as e:
            raise Exception(f"Image analysis failed: {str(e)}")
    
    def predict_weather_risk(self, weather_data):
        try:
            if not self.lstm_loaded:
                # Default weather risk calculation
                temp = weather_data.get('tempC', 28.0)
                humidity = weather_data.get('humidity', 75.0)
                rainfall = weather_data.get('precipMM', 0.5)
                
                risk_score = 0.0
                
                # Temperature factor
                if temp > 30 or temp < 20:
                    risk_score += 0.3
                elif temp > 35 or temp < 15:
                    risk_score += 0.5
                
                # Humidity factor
                if humidity > 80:
                    risk_score += 0.4
                elif humidity > 90:
                    risk_score += 0.6
                
                # Rainfall factor
                if rainfall > 5:
                    risk_score += 0.3
                elif rainfall > 10:
                    risk_score += 0.5
                
                risk_probability = min(risk_score, 1.0)
                
                # Determine risk level
                if risk_probability > 0.7:
                    risk_level = "Very High"
                elif risk_probability > 0.5:
                    risk_level = "High"
                elif risk_probability > 0.3:
                    risk_level = "Medium"
                elif risk_probability > 0.1:
                    risk_level = "Low"
                else:
                    risk_level = "Very Low"
                
                return risk_level, risk_probability
            
            # LSTM-based prediction
            current_features = []
            for col in self.feature_columns:
                current_features.append(weather_data.get(col, 0.0))
            
            scaled_features = self.weather_scaler.transform([current_features])
            
            self.weather_history.append(scaled_features.flatten())
            if len(self.weather_history) > self.sequence_length:
                self.weather_history.pop(0)
            
            while len(self.weather_history) < self.sequence_length:
                self.weather_history.append(scaled_features.flatten())
            
            sequence = np.array(self.weather_history).reshape(1, self.sequence_length, -1)
            risk_probability = float(self.lstm_model.predict(sequence, verbose=0)[0][0])
            
            risk_level = "Low"
            if risk_probability > 0.7:
                risk_level = "Very High"
            elif risk_probability > 0.5:
                risk_level = "High"
            elif risk_probability > 0.3:
                risk_level = "Medium"
            elif risk_probability > 0.1:
                risk_level = "Low"
            else:
                risk_level = "Very Low"
            
            return risk_level, risk_probability
            
        except Exception as e:
            st.error(f"Weather prediction error: {e}")
            return "Unknown", 0.0

def create_prediction_plot(image, diagnosis, confidence, all_predictions):
    """Create enhanced visualization for 7 classes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Image
    ax1.imshow(image)
    ax1.set_title(f'Jackfruit Leaf Analysis\nDiagnosis: {diagnosis}\nConfidence: {confidence:.1f}%', 
                 fontweight='bold', fontsize=12)
    ax1.axis('off')
    
    # Predictions chart for 7 classes
    classes = list(all_predictions.keys())
    probabilities = list(all_predictions.values())
    
    # Color coding based on disease severity
    colors = []
    for cls in classes:
        if 'Healthy' in cls:
            colors.append('#51cf66')  # Green for healthy
        elif cls in ['Rhizopus Rot', 'Black_Spot_of_Jackfruit']:
            colors.append('#ff6b6b')  # Red for high severity
        elif cls in ['Leaf Spot', 'Algal_Leaf_Spot_of_Jackfruit']:
            colors.append('#ffa94d')  # Orange for medium severity
        else:
            colors.append('#ffe066')  # Yellow for low severity
    
    bars = ax2.barh(classes, probabilities, color=colors)
    ax2.set_xlabel('Confidence (%)', fontweight='bold')
    ax2.set_title('7-Class Disease Probability Distribution', fontweight='bold')
    ax2.set_xlim(0, 100)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    # Initialize fusion system - FIXED SESSION STATE
    if 'fusion_system' not in st.session_state:
        try:
            st.session_state.fusion_system = MultimodalDiseaseFusion()
        except Exception as e:
            st.error(f"Failed to initialize the disease detection system: {str(e)}")
            st.stop()
    
    fusion_system = st.session_state.fusion_system
    
    # Header
    st.markdown('<div class="main-header">🍈 Advanced Jackfruit Disease Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered analysis for 7 different jackfruit conditions</div>', unsafe_allow_html=True)
    
    # Main layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📁 Upload Leaf Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a jackfruit leaf image", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a clear image of a jackfruit leaf for analysis"
        )
        
        # Image preview
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Leaf Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
        
        st.markdown("### 🌤️ Weather Conditions")
        
        # Weather inputs
        temp = st.number_input(
            "Temperature (°C)", 
            min_value=0.0, 
            max_value=50.0, 
            value=28.0, 
            step=0.1,
            help="Current temperature in Celsius"
        )
        
        humidity = st.number_input(
            "Humidity (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=75.0, 
            step=0.1,
            help="Current humidity percentage"
        )
        
        rainfall = st.number_input(
            "Rainfall (mm)", 
            min_value=0.0, 
            max_value=100.0, 
            value=0.5, 
            step=0.1,
            help="Recent rainfall in millimeters"
        )
        
        weather_condition = st.selectbox(
            "Weather Condition",
            ["Sunny", "Cloudy", "Rainy", "Humid", "Dry"],
            help="Current weather condition"
        )
        
        # Analyze button
        analyze_btn = st.button(
            "🔍 Analyze Disease", 
            type="primary", 
            use_container_width=True,
            disabled=uploaded_file is None
        )
        
        # Capabilities info
        with st.expander("💡 Detection Capabilities"):
            st.markdown("""
            - ✅ **Healthy Leaves**
            - ⚠️ **Leaf Bright** (Early signs)
            - 🔴 **Leaf Spot Diseases**
            - 🚨 **Rhizopus Rot**
            - ⚠️ **Algal Leaf Spot**
            - 🔴 **Black Spot**
            - ✅ **Healthy Jackfruit Leaves**
            """)
    
    with col2:
        st.markdown("### 📊 Analysis Results")
        
        if not analyze_btn or uploaded_file is None:
            st.info("👆 Upload a jackfruit leaf image and click 'Analyze Disease' to begin")
            st.markdown("""
            <div style='text-align: center; padding: 3rem;'>
                <h3>🔬 Advanced 7-Class Detection</h3>
                <p>Upload a clear image of a jackfruit leaf for comprehensive AI analysis</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("🔍 Analyzing image with AI... This may take a few seconds."):
                try:
                    # Prepare weather data
                    weather_data = {
                        'tempC': temp,
                        'humidity': humidity,
                        'precipMM': rainfall,
                        'temp_avg_7d': temp,
                        'humidity_avg_7d': humidity,
                        'temp_humidity_index': temp * (humidity / 100)
                    }
                    
                    # Analyze image
                    diagnosis, confidence, all_predictions, disease_info = fusion_system.analyze_image(image)
                    
                    # Predict weather risk
                    risk_level, risk_probability = fusion_system.predict_weather_risk(weather_data)
                    
                    # Enhanced fusion logic
                    severity = disease_info.get('severity', 'None')
                    
                    # Determine alert level and message
                    if severity == 'High' and risk_level in ["High", "Very High"]:
                        alert_level = "CRITICAL"
                        message = f"🚨 EMERGENCY: {diagnosis} detected! High severity disease with risky weather conditions."
                        action = "Immediate intervention required! Isolate plants, apply emergency treatment protocol."
                        alert_class = "alert-critical"
                    elif severity == 'High':
                        alert_level = "HIGH ALERT"
                        message = f"🔴 HIGH RISK: {diagnosis} identified. This is a high-severity disease."
                        action = "Begin intensive treatment immediately. Monitor plant health closely."
                        alert_class = "alert-critical"
                    elif severity == 'Medium' and risk_level in ["High", "Very High"]:
                        alert_level = "ALERT"
                        message = f"⚠️ WARNING: {diagnosis} detected with unfavorable weather conditions."
                        action = "Start treatment protocol. Increase monitoring frequency due to weather risk."
                        alert_class = "alert-warning"
                    elif diagnosis not in ['Healthy', 'Healthy_leaf_of_Jackfruit']:
                        alert_level = "NOTICE"
                        message = f"🔔 DETECTED: {diagnosis} identified on the leaf."
                        action = "Begin recommended treatment. Monitor plant development."
                        alert_class = "alert-warning"
                    else:
                        alert_level = "INFO"
                        message = "✅ PLANT HEALTHY: No diseases detected."
                        action = "Continue regular monitoring and maintenance practices."
                        alert_class = "alert-info"
                    
                    # Display alert
                    st.markdown(f"""
                    <div class="{alert_class}">
                        <h2>{alert_level} ALERT</h2>
                        <h3>{message}</h3>
                        <p><strong>💡 Recommended Action:</strong> {action}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create tabs for organized results
                    tab1, tab2, tab3 = st.tabs(["📷 Image Analysis", "🌤️ Weather Risk", "📋 Disease Info"])
                    
                    with tab1:
                        # Image analysis results
                        st.markdown('<div class="info-card">', unsafe_allow_html=True)
                        st.subheader("Image Analysis Results")
                        
                        severity_class = f"severity-{severity.lower()}"
                        st.markdown(f"""
                        - **Diagnosis:** <span class="{severity_class}">{diagnosis}</span>
                        - **Severity:** <span class="{severity_class}">{severity}</span>
                        - **Confidence Level:** {confidence:.1f}%
                        """, unsafe_allow_html=True)
                        
                        # Display prediction plot
                        fig = create_prediction_plot(image, diagnosis, confidence, all_predictions)
                        st.pyplot(fig)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with tab2:
                        # Weather risk assessment
                        st.markdown('<div class="weather-card">', unsafe_allow_html=True)
                        st.subheader("Weather Risk Assessment")
                        
                        risk_percent = min(risk_probability * 100, 100)
                        risk_class = "risk-high" if risk_percent > 70 else "risk-medium" if risk_percent > 40 else "risk-low"
                        
                        st.markdown(f"""
                        - **Risk Level:** {risk_level}
                        - **Risk Probability:** {risk_percent:.1f}%
                        """)
                        
                        # Risk meter
                        st.markdown(f"""
                        <div class="risk-meter">
                            <div class="risk-fill {risk_class}" style="width: {risk_percent}%"></div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("**Current Conditions:**")
                        st.markdown(f"""
                        - Temperature: {temp}°C
                        - Humidity: {humidity}%
                        - Rainfall: {rainfall}mm
                        - Condition: {weather_condition}
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with tab3:
                        # Disease information
                        if disease_info:
                            st.markdown('<div class="info-card">', unsafe_allow_html=True)
                            st.subheader(f"Disease Information: {diagnosis}")
                            
                            if disease_info.get('symptoms'):
                                st.markdown(f"**Symptoms:** {disease_info['symptoms']}")
                            if disease_info.get('causes'):
                                st.markdown(f"**Causes:** {disease_info['causes']}")
                            if disease_info.get('treatment'):
                                st.markdown(f"**Treatment:** {disease_info['treatment']}")
                            if disease_info.get('prevention'):
                                st.markdown(f"**Prevention:** {disease_info['prevention']}")
                            if disease_info.get('maintenance'):
                                st.markdown(f"**Maintenance:** {disease_info['maintenance']}")
                            if disease_info.get('recommendation'):
                                st.markdown(f"**Recommendation:** {disease_info['recommendation']}")
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Timestamp
                    st.caption(f"⏰ Analysis performed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                except Exception as e:
                    st.error(f"❌ Analysis Error: {str(e)}")
                    st.info("""
                    **Troubleshooting tips:**
                    - Ensure the image is a clear photo of a jackfruit leaf
                    - Check that the file is not corrupted
                    - Try with a different image format (JPG, PNG recommended)
                    - Make sure the image shows the leaf clearly with good lighting
                    """)

    # Sidebar with additional information - FIXED: Check if fusion_system exists
    with st.sidebar:
        st.markdown("### 🏥 System Status")
        st.success("✅ System Operational")
        
        if hasattr(fusion_system, 'class_names'):
            st.info(f"📊 **Available Classes:** {len(fusion_system.class_names)}")
            
            st.markdown("### 📈 Detection Classes")
            for idx, class_name in fusion_system.class_names.items():
                severity = fusion_system.disease_info.get(class_name, {}).get('severity', 'Unknown')
                severity_emoji = "🟢" if 'Healthy' in class_name else "🔴" if severity == 'High' else "🟡" if severity == 'Medium' else "🔵"
                st.write(f"{severity_emoji} {class_name}")
        
        st.markdown("### 💡 Tips for Best Results")
        st.markdown("""
        - Use clear, well-lit images
        - Focus on the leaf surface
        - Avoid blurry or dark photos
        - Capture the entire leaf when possible
        - Ensure good contrast with background
        """)

if __name__ == "__main__":
    main()
