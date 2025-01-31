import matplotlib.pyplot as plt
import seaborn as sns
import os

from supportfn.constant import read_csv
from sklearn.manifold import TSNE

def perform_eda(df, save_path="plots/"):
    os.makedirs(save_path, exist_ok=True)  # Ensure the folder exists
    
    # Set the plot size
    plt.figure(figsize=(15, 12))

    # 1. Class Distribution (Bar plot for target)
    plt.subplot(2, 3, 1)
    sns.countplot(x='target', data=df)
    plt.title("Class Distribution (Heart Disease vs No Disease)")

    # 2. Numeric Distributions: Histograms/KDE for age, chol, thalach
    plt.subplot(2, 3, 2)
    sns.histplot(df['age'], kde=True)
    plt.title("Age Distribution")

    plt.subplot(2, 3, 3)
    sns.histplot(df['chol'], kde=True)
    plt.title("Cholesterol Distribution")

    plt.subplot(2, 3, 4)
    sns.histplot(df['thalach'], kde=True)
    plt.title("Max Heart Rate (thalach) Distribution")

    # 3. Categorical Analysis: Boxplots for cp, exang, thal vs. target
    plt.subplot(2, 3, 5)
    sns.boxplot(x='target', y='cp', data=df)
    plt.title("Chest Pain (cp) vs. Heart Disease Status")

    plt.subplot(2, 3, 6)
    sns.boxplot(x='target', y='exang', data=df)
    plt.title("Exercise Induced Angina (exang) vs. Heart Disease Status")

    # Save the first set of plots
    save_file = os.path.join(save_path, "eda_part_1.png")
    plt.tight_layout()
    plt.savefig(save_file)
    plt.close()
    print(f"✅ EDA Part 1 saved at: {save_file}")

    # 4. Correlation Heatmap: Multicollinearity check (e.g., trestbps vs. age)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".1f", cmap="coolwarm")
    plt.title("Correlation Heatmap")

    # Save the correlation heatmap
    save_file = os.path.join(save_path, "correlation_heatmap.png")
    plt.tight_layout()
    plt.savefig(save_file)
    plt.close()
    print(f"✅ Correlation heatmap saved at: {save_file}")

    # 5. Pair Plots: Highlight interactions (e.g., age vs. chol colored by target)
    sns.pairplot(df[['age', 'chol', 'thalach', 'target']], hue='target', diag_kind='kde')
    save_file = os.path.join(save_path, "pair_plots.png")
    plt.savefig(save_file)
    plt.close()
    print(f"✅ Pair plots saved at: {save_file}")

    # 6. Outlier Detection: Boxplots for chol (values >500 may be errors)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='chol', data=df)
    plt.title("Cholesterol Outlier Detection")

    # Save the outlier detection plot
    save_file = os.path.join(save_path, "chol_outliers.png")
    plt.tight_layout()
    plt.savefig(save_file)
    plt.close()
    print(f"✅ Outlier detection for cholesterol saved at: {save_file}")


def plot_tsne(df, save_path="plots/"):
    os.makedirs(save_path, exist_ok=True)  # Ensure the folder exists
    
    # Extract features and target
    features = df.drop(columns=['target'])
    target = df['target']
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    
    # Create a DataFrame with t-SNE results
    tsne_df = pd.DataFrame(data=tsne_results, columns=['TSNE1', 'TSNE2'])
    tsne_df['target'] = target
    
    # Plot t-SNE results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='TSNE1', y='TSNE2', hue='target', palette='viridis', data=tsne_df)
    plt.title("t-SNE Visualization")
    
    # Save the t-SNE plot
    save_file = os.path.join(save_path, "tsne_plot.png")
    plt.tight_layout()
    plt.savefig(save_file)
    plt.close()
    print(f"✅ t-SNE plot saved at: {save_file}")