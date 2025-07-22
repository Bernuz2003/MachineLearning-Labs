import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------
# Reshape utilities (from gau.py)
# ---------------------------------------------------------------------

def as_col(x):
    return x.reshape((x.size, 1))

def as_row(x):
    return x.reshape((1, x.size))

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_project_data(fname):
    """Carica il dataset del progetto (classificazione binaria)."""
    data = []
    labels = []
    
    with open(fname) as f:
        for line in f:
            try:
                parts = [p.strip() for p in line.split(',')]
                feats = np.array([float(x) for x in parts[:-1]], dtype=np.float64)
                lab = int(parts[-1])
                data.append(as_col(feats))
                labels.append(lab)
            except (ValueError, IndexError):
                continue  # skip malformed lines
    
    data_matrix = np.hstack(data)
    labels_array = np.array(labels, dtype=np.int32)
    
    print(f"Dataset: {data_matrix.shape[1]} campioni, {data_matrix.shape[0]} features")
    print(f"Classe 0 (Fake): {np.sum(labels_array == 0)}")
    print(f"Classe 1 (Genuine): {np.sum(labels_array == 1)}")
    
    return data_matrix, labels_array

# ---------------------------------------------------------------------
# Gaussian density functions (adapted from gau.py)
# ---------------------------------------------------------------------

def logpdf_GAU_1D(x, mu, var):
    """Compute log-density for 1D Gaussian distribution."""
    return -0.5 * np.log(2 * np.pi * var) - 0.5 * ((x - mu) ** 2) / var

def compute_mu_var_1D(x):
    """Compute ML estimates for mean and variance of 1D Gaussian."""
    mu = x.mean()
    var = x.var()
    return mu, var

def compute_ll_1D(x, mu, var):
    """Compute log-likelihood for 1D Gaussian."""
    return logpdf_GAU_1D(x, mu, var).sum()

# ---------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------

def plot_all_features_summary(data, labels, ml_estimates, out_dir):
    """Create summary plot showing all features for both classes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    n_features = data.shape[0]
    class_names = ['Fake', 'Genuine']
    
    fig, axes = plt.subplots(2, n_features, figsize=(4*n_features, 8))
    
    for class_idx, class_name in enumerate(class_names):
        class_data = data[:, labels == class_idx]
        
        for feat_idx in range(n_features):
            ax = axes[class_idx, feat_idx]
            
            feature_data = class_data[feat_idx, :]
            mu_ml, var_ml = ml_estimates[class_idx][feat_idx]
            
            # Histogram
            ax.hist(feature_data, bins=20, density=True, alpha=0.7, 
                   color='lightblue', edgecolor='black')
            
            # Gaussian fit
            x_range = np.linspace(feature_data.min(), feature_data.max(), 200)
            pdf_values = np.exp(logpdf_GAU_1D(x_range, mu_ml, var_ml))
            ax.plot(x_range, pdf_values, 'r-', linewidth=2)
            
            ax.set_title(f'{class_name} - F{feat_idx + 1}\nμ={mu_ml:.2f}, σ²={var_ml:.2f}')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "all_features_summary.pdf", bbox_inches='tight')
    # plt.show()

# ---------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------

def analyze_gaussian_fit_quality(feature_data, mu_ml, var_ml):
    """Analyze quality of Gaussian fit using various metrics."""
    
    # Compute log-likelihood
    ll = compute_ll_1D(feature_data, mu_ml, var_ml)
    
    # Skewness test (should be close to 0 for Gaussian)
    mean_centered = feature_data - mu_ml
    skewness = np.mean((mean_centered / np.sqrt(var_ml)) ** 3)
    
    # Kurtosis test (should be close to 3 for Gaussian)
    kurtosis = np.mean((mean_centered / np.sqrt(var_ml)) ** 4)
    
    # Kolmogorov-Smirnov-like test (simplified)
    # Compare empirical vs theoretical quantiles
    sorted_data = np.sort(feature_data)
    n = len(sorted_data)
    empirical_cdf = np.arange(1, n+1) / n
    
    # Theoretical CDF values
    from scipy.stats import norm
    theoretical_cdf = norm.cdf(sorted_data, mu_ml, np.sqrt(var_ml))
    max_diff = np.max(np.abs(empirical_cdf - theoretical_cdf))
    
    return {
        'log_likelihood': ll,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'max_cdf_diff': max_diff
    }

# ---------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------

def analyze_univariate_gaussian_fits(data, labels, output_dir="gaussian_fits"):
    """Perform complete univariate Gaussian analysis."""
    
    output_path = Path(output_dir)
    class_names = ['Fake', 'Genuine']
    n_features = data.shape[0]
    
    # Storage for ML estimates
    ml_estimates = [[] for _ in range(2)]  # 2 classes
    
    print("\n" + "="*70)
    print("STIMA PARAMETRI GAUSSIANI UNIVARIATI")
    print("="*70)
    
    # Compute ML estimates for each class and feature
    for class_idx, class_name in enumerate(class_names):
        print(f"\n📈 Classe {class_name}:")
        
        class_data = data[:, labels == class_idx]
        print(f"   Campioni: {class_data.shape[1]}")
        
        for feat_idx in range(n_features):
            feature_data = class_data[feat_idx, :]
            
            # Compute ML estimates
            mu_ml, var_ml = compute_mu_var_1D(feature_data)
            ml_estimates[class_idx].append((mu_ml, var_ml))
            
            # Compute log-likelihood
            ll = compute_ll_1D(feature_data, mu_ml, var_ml)
            
            print(f"   Feature {feat_idx + 1}: μ={mu_ml:.4f}, σ²={var_ml:.4f}, LL={ll:.2f}")
    
    print(ml_estimates)
    # Create summary plot
    plot_all_features_summary(data, labels, ml_estimates, output_path)
            
    return ml_estimates

# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("="*70)
    print("    ANALISI GAUSSIANA UNIVARIATA - DATASET PROGETTO")
    print("="*70)
    
    # Load data
    try:
        data, labels = load_project_data("trainData.csv")
    except FileNotFoundError:
        exit(1)
    
    # Perform analysis
    ml_estimates = analyze_univariate_gaussian_fits(data, labels)
    