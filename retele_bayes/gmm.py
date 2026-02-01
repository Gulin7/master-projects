import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

def plot_results(X_test, y_pred, gmm):
    """Generates the 2D Classification Map and 1D Density Curves."""
    
    # Plot 1: Classification Map (2D)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', edgecolors='k', s=50, alpha=0.7)
    plt.title("Classification Map (Test Data)")
    plt.xlabel("Eruption Duration (Scaled)")
    plt.ylabel("Waiting Time (Scaled)")
    plt.grid(True, linestyle='--', alpha=0.3)

    # Plot 2: Gaussian Density (1D - Eruption Duration)
    plt.subplot(1, 2, 2)
    x_col = X_test[:, 0]
    plt.hist(x_col, bins=20, density=True, alpha=0.3, color='gray', label='Real Data')
    
    x_axis = np.linspace(x_col.min()-1, x_col.max()+1, 1000)
    for i in range(gmm.n_components):
        mean = gmm.means_[i, 0]
        var = gmm.covariances_[i][0, 0] if gmm.covariance_type == 'full' else gmm.covariances_[i][0]
        y_axis = gmm.weights_[i] * stats.norm.pdf(x_axis, mean, np.sqrt(var))
        plt.plot(x_axis, y_axis, linewidth=2, label=f'Cluster {i+1}')
        plt.fill_between(x_axis, y_axis, alpha=0.2)
    
    plt.title("Gaussian Density (Eruptions)")
    plt.xlabel("Duration (Scaled)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_old_faithful():
    print("--- Processing Old Faithful Dataset ---")

    # 1. Load Data
    # 'eruptions' = duration of eruption, 'waiting' = time to next eruption
    dataset = sm.datasets.get_rdataset('faithful')
    X = dataset.data.values
    
    # Create synthetic labels for accuracy check (Rule: Eruption > 3 mins is Group 1)
    y_true = (X[:, 0] > 3.0).astype(int)

    # 2. Preprocessing (Scaling is vital for distance calculations)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Split Data (70% Train, 30% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_true, test_size=0.30, random_state=42, stratify=y_true
    )

    # 4. Train GMM (Estimate Theta)
    # We look for 2 components (Short vs Long)
    gmm = GaussianMixture(n_components=2, covariance_type='full', n_init=10, random_state=42)
    gmm.fit(X_train)

    print(f"Estimated Theta (Means):\n{gmm.means_}")

    # 5. Classify (Predict)
    y_pred = gmm.predict(X_test)

    # Align labels (ensure Cluster 0 is mapped to Label 0)
    from scipy.stats import mode
    labels = np.zeros_like(y_pred)
    for i in range(2):
        mask = (y_pred == i)
        if np.sum(mask) > 0:
            labels[mask] = mode(y_test[mask], keepdims=True)[0][0]
    
    # 6. Evaluate
    acc = accuracy_score(y_test, labels)
    print(f"Accuracy on Test Set: {acc:.2%}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, labels))

    # 7. Visualize
    plot_results(X_test, labels, gmm)

if __name__ == '__main__':
    run_old_faithful()