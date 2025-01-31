import pandas as pd
import numpy as np

def numeric_stats(df: pd.DataFrame, column: str = None) -> pd.DataFrame:
    """Calculate statistics for numerical columns (specific or all)."""
    if column:
        if column not in df.select_dtypes(include=np.number).columns:
            raise ValueError(f"{column} is not a numerical column")
        return df[[column]].describe().T.assign(
            variance=df[column].var(),
            range=df[column].max() - df[column].min(),
            IQR=df[column].quantile(0.75) - df[column].quantile(0.25)
        )
    
    stats = df.describe().T
    stats['variance'] = df.var()
    stats['range'] = stats['max'] - stats['min']
    stats['IQR'] = stats['75%'] - stats['25%']
    return stats[['count', 'mean', 'std', 'variance', 
                 'min', '25%', '50%', '75%', 'max', 
                 'range', 'IQR']]

def categorical_stats(df: pd.DataFrame, column: str = None) -> pd.DataFrame:
    """Calculate statistics for categorical columns (specific or all)."""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if column:
        if column not in cat_cols:
            raise ValueError(f"{column} is not a categorical column")
        return pd.DataFrame([{
            'column': column,
            'unique_count': df[column].nunique(),
            'mode': ', '.join(map(str, df[column].mode().values)),
            'mode_count': df[column].value_counts().iloc[0],
            'missing_values': df[column].isnull().sum()
        }])
    
    stats = []
    for col in cat_cols:
        mode = df[col].mode().values
        stats.append({
            'column': col,
            'unique_count': df[col].nunique(),
            'mode': ', '.join(map(str, mode)),
            'mode_count': df[col].value_counts().values[0],
            'missing_values': df[col].isnull().sum()
        })
    return pd.DataFrame(stats)

def get_modes(df: pd.DataFrame, column: str = None) -> pd.DataFrame:
    """Get mode(s) for specific column or all columns."""
    if column:
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in DataFrame")
        return pd.DataFrame([{
            'column': column,
            'mode': ', '.join(map(str, df[column].mode().values)),
            'count': df[column].value_counts().iloc[0]
        }])
    
    modes = []
    for col in df.columns:
        col_modes = df[col].mode()
        modes.append({
            'column': col,
            'mode': ', '.join(map(str, col_modes.values)),
            'count': df[col].value_counts().iloc[0]
        })
    return pd.DataFrame(modes)

def column_correlations(df: pd.DataFrame, column: str) -> pd.Series:
    """Get correlation values for a specific column with all others."""
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")
    return df.corr()[column].sort_values(ascending=False)

def generate_stat_report(df: pd.DataFrame, 
                        column: str = None,
                        file_path: str = '../stats/stat_report.txt') -> None:
    """Generate report for specific column or entire dataframe."""
    report = []
    
    if column:
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in DataFrame")
            
        report.append(f"=== DETAILED REPORT FOR COLUMN: {column} ===")
        report.append(f"Data type: {df[column].dtype}")
        
        if np.issubdtype(df[column].dtype, np.number):
            report.append("\nNUMERICAL STATISTICS:")
            report.append(numeric_stats(df, column).to_string())
        else:
            report.append("\nCATEGORICAL STATISTICS:")
            report.append(categorical_stats(df, column).to_string())
        
        report.append("\nMODAL VALUES:")
        report.append(get_modes(df, column).to_string())
        
        if np.issubdtype(df[column].dtype, np.number):
            report.append("\nCORRELATIONS WITH OTHER FEATURES:")
            report.append(column_correlations(df, column).to_string())
    else:
        # Original full report implementation
        report.append("=== COMPREHENSIVE DATASET REPORT ===")
        report.append(f"Shape: {df.shape}")
        report.append("\nNUMERICAL STATISTICS:")
        report.append(numeric_stats(df).to_string())
        report.append("\nCATEGORICAL STATISTICS:")
        report.append(categorical_stats(df).to_string())
        report.append("\nFEATURE CORRELATIONS:")
        report.append(df.corr().to_string())
        report.append("\nMODAL VALUES:")
        report.append(get_modes(df).to_string())

    # Save to file
    import os
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Report saved to {file_path}")

# Example usage from main.py:
# generate_stat_report(df)  # Full report
# generate_stat_report(df, column='age')  # Specific column report