import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import supportfn
import supportfn.constant
import supportfn.preprocess

def train_model():
    # Load and preprocess data
    df = supportfn.constant.read_csv("../data/heart.csv")
    X, y = supportfn.preprocess.preprocess_heart_data(df)
    
    # Train model
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, "../models/heart_model.pkl")
    return "Model trained successfully"