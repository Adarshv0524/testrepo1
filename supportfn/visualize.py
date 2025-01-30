import matplotlib.pyplot as plt
import seaborn as sns
import os

from supportfn.constant import read_csv

# Function to plot heart distributions
def plot_heart_distributions(df, save_path="../plots/"):
    """Plot distributions of key variables and save them."""
    os.makedirs(save_path, exist_ok=True)  # Ensure the folder exists

    plt.figure(figsize=(15, 10))
    
    # Age distribution
    plt.subplot(2, 2, 1)
    sns.histplot(df['age'], kde=True)
    plt.title("Age Distribution")
    
    # Cholesterol vs Heart Disease
    plt.subplot(2, 2, 2)
    sns.boxplot(x='target', y='chol', data=df)
    plt.title("Cholesterol by Heart Disease Status")
    
    # Save plots
    plt.tight_layout()
    save_file = os.path.join(save_path, "distributions.png")
    plt.savefig(save_file)
    plt.close()
    
    print(f"✅ Distribution plots saved at: {save_file}")

# Function to plot correlation matrix
def plot_correlation_matrix(df):
    """Generate correlation heatmap."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".1f", cmap="coolwarm")
    plt.title("Feature Correlation Matrix")
    plt.show()
