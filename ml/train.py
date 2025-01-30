import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from supportive_functions.data_loader import load_data
from supportive_functions.preprocessor import preprocess_data

def train_model():
    # Load and preprocess data
    df = load_data("../data/heart.csv")
    X, y = preprocess_data(df)
    
    # Train model
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, "../models/heart_model.pkl")
    return "Model trained successfully"