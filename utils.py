import pandas as pd
import tempfile
import csv
import json
from groq import Groq
import os
from datetime import datetime

def preprocess_and_save(file):
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            return None, None, None, "Unsupported file format."

        # Data cleaning
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)

        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    pass

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w', newline='', encoding='utf-8') as temp_file:
            df.to_csv(temp_file.name, index=False, quoting=csv.QUOTE_ALL)
            return df, df.columns.tolist(), df.to_html(classes='table table-striped', index=False), temp_file.name, None
    except Exception as e:
        return None, None, None, None, str(e)


def save_analysis_history(entry):
    try:
        if os.path.exists('analysis_history.json'):
            with open('analysis_history.json', 'r') as f:
                history = json.load(f)
        else:
            history = []
    except:
        history = []
    
    history.append(entry)
    
    try:
        with open('analysis_history.json', 'w') as f:
            json.dump(history[-50:], f, indent=2)
    except Exception as e:
        print(f"Error saving history: {e}")