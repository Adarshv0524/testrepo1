from supportfn.constant import read_csv
from supportfn.visualize import *
from supportfn.preprocess import *
from supportfn.stats import generate_stat_report

# Read the dataset
df = read_csv("heart.csv")  # File path is handled in `constant.py`
print(df.head(3))

# Generate visualizations
# plot_heart_distributions(df, save_path="plots/")
# plot_correlation_matrix(df)

# Handle class imbalance
# X_res, y_res = apply_smote(df)
# resampled_df = create_resampled_df(X_res, y_res)

# # Visualize balanced data
# plot_heart_distributions(resampled_df, save_path="../plots/balanced_")
# print("\nResampled Dataset Shape:", resampled_df.shape)


# generate_stat_report(df)

# perform_eda(df)
show_null_values(df)
scale_features(df , ['age', 'chol', 'thalach', 'trestbps', 'oldpeak']).to_csv('data/preprocessed/scaled_heart.csv', index=False)

plot_tsne(df, save_path="plots/")