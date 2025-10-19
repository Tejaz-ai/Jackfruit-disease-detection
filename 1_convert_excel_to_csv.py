# 1_convert_excel_to_csv.py
import pandas as pd
import os

def convert_excel_to_csv(excel_file_path, csv_file_path):
    """Convert Excel file to CSV format"""
    try:
        # Read Excel file
        df = pd.read_excel(excel_file_path)
        print(f"Original data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Save as CSV
        os.makedirs('weather_data', exist_ok=True)
        df.to_csv(csv_file_path, index=False)
        print(f"Data successfully saved to: {csv_file_path}")
        
        return df
        
    except Exception as e:
        print(f"Error converting file: {e}")
        return None

# Convert your Excel file to CSV
if __name__ == "__main__":
    excel_file = "Pune.xlsx"  # Change this to your actual Excel file name
    csv_file = "weather_data/pune_weather_data.csv"
    
    df = convert_excel_to_csv(excel_file, csv_file)
    
    if df is not None:
        print("\nFirst 5 rows of data:")
        print(df.head())
        print(f"\nData types:\n{df.dtypes}")