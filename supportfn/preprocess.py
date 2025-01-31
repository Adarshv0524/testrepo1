import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
import pandas as pd

def apply_smote(df, target_col='target', random_state=42):
    """
    Apply SMOTE to balance dataset
    Returns:
    X_resampled, y_resampled (features and target)
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Check class distribution
    print("Class distribution before SMOTE:")
    print(y.value_counts())
    
    # Apply SMOTE
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    
    print("\nClass distribution after SMOTE:")
    print(y_res.value_counts())
    
    return X_res, y_res

def create_resampled_df(X_res, y_res):
    """Combine resampled features and target into DataFrame"""
    return pd.concat([pd.DataFrame(X_res), pd.Series(y_res, name='target')], axis=1)


def show_null_values(df):
    """Display the number of null values in each column of the DataFrame"""
    null_counts = df.isnull().sum()
    print("Null values in each column:")
    print(null_counts[null_counts > 0])

def fill_missing_values(df):
    """Fill missing values in the DataFrame using interpolation"""
    df_interpolated = df.interpolate()
    return df_interpolated

def balance_data(df, target_col='target', random_state=42):
    """
    Balance the dataset by applying SMOTE and filling missing values
    Returns:
    balanced_df (DataFrame)
    """
    # Fill missing values
    df_filled = fill_missing_values(df)
    
    # Apply SMOTE
    X_res, y_res = apply_smote(df_filled, target_col, random_state)
    
    # Create resampled DataFrame
    balanced_df = create_resampled_df(X_res, y_res)
    
    return balanced_df

def scale_features(df, numeric_features):
    """
    Scale numeric features using StandardScaler
    Returns:
    scaled_df (DataFrame)
    """
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    return df