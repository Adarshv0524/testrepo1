import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_heart_data(df, target_col="target"):
    """
    Clean and preprocess heart disease data.
    Returns:
        - Preprocessed DataFrame
        - Preprocessor object (for future inference)
    """
    # Handle missing values (example)
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    # Define features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Define column types
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
    
    # Fit-transform the data
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, y, preprocessor