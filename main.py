from supportfn.constant import read_csv
from supportfn.visualize import plot_heart_distributions,plot_correlation_matrix

# Read the dataset
df = read_csv("heart.csv")  # File path is handled in `constant.py`
print(df.head())

# Generate visualizations
plot_heart_distributions(df, save_path="plots/")
plot_correlation_matrix(df)