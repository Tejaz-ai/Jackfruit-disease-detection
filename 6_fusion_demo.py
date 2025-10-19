# fusion_demo.py
from fusion_engine import MultimodalDiseaseFusion
import random

def create_sample_weather_data():
    """Create sample weather data for demonstration"""
    return {
        'tempC': random.uniform(25, 35),
        'humidity': random.uniform(60, 95),
        'precipMM': random.uniform(0, 5),
        'temp_avg_7d': random.uniform(26, 32),
        'humidity_avg_7d': random.uniform(65, 90),
        'temp_humidity_index': random.uniform(20, 30)
    }

def run_demo():
    """Run demonstration of the fusion system"""
    print("🌿 MULTIMODAL JACKFRUIT DISEASE FUSION SYSTEM")
    print("=" * 55)
    
    # Initialize fusion engine
    fusion_system = MultimodalDiseaseFusion()
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'High Risk + Disease Detected',
            'image': 'data/test/Black_Spot_of_Jackfruit/sample_image.jpg',
            'weather': {'tempC': 30, 'humidity': 90, 'precipMM': 2.5,
                       'temp_avg_7d': 29, 'humidity_avg_7d': 85, 'temp_humidity_index': 27}
        },
        {
            'name': 'High Risk + Healthy Plant', 
            'image': 'data/test/Healthy_Leaf_of_Jackfruit/sample_image.jpg',
            'weather': {'tempC': 31, 'humidity': 88, 'precipMM': 3.0,
                       'temp_avg_7d': 30, 'humidity_avg_7d': 82, 'temp_humidity_index': 26.5}
        },
        {
            'name': 'Low Risk + Disease Detected',
            'image': 'data/test/Algal_Leaf_Spot_of_Jackfruit/sample_image.jpg',
            'weather': {'tempC': 25, 'humidity': 60, 'precipMM': 0.5,
                       'temp_avg_7d': 24, 'humidity_avg_7d': 55, 'temp_humidity_index': 15}
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'='*55}")
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"{'='*55}")
        
        try:
            result = fusion_system.process_complete_analysis(
                scenario['image'], scenario['weather']
            )
            
            print(f"\n📊 RESULTS:")
            print(f"Alert Level: {result['alert_level']}")
            print(f"Message: {result['message']}")
            print(f"Action: {result['recommended_action']}")
            
            print(f"\n📷 Image Analysis:")
            print(f"  Diagnosis: {result['image_analysis']['diagnosis']}")
            print(f"  Confidence: {result['image_analysis']['confidence']}%")
            
            print(f"\n🌤️  Weather Analysis:")
            print(f"  Risk Level: {result['weather_prediction']['risk_level']}")
            print(f"  Risk Probability: {result['weather_prediction']['risk_probability']:.3f}")
            
        except Exception as e:
            print(f"❌ Error in scenario: {e}")
    
    print(f"\n{'='*55}")
    print("🎯 DEMO COMPLETED! SYSTEM IS READY FOR REAL DATA")
    print(f"{'='*55}")

if __name__ == "__main__":
    run_demo()