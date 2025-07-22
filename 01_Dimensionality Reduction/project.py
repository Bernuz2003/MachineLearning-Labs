import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy.linalg

# ---------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------

def as_column(v):
    """Return *v* as a column vector with shape (size, 1)."""
    return v.reshape(v.size, 1)

def as_row(v):
    """Return *v* as a row vector with shape (1, size)."""
    return v.reshape(1, v.size)

# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def load_project_data(fname="trainData.csv"):
    """Carica il dataset del progetto (classificazione binaria)."""
    data = []
    labels = []
    
    with open(fname) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                nums = [float(x) for x in line.split(',')]
                features = np.array(nums[:-1])
                label = int(nums[-1])
                data.append(as_column(features))
                labels.append(label)
            except Exception:
                pass
    
    data_matrix = np.hstack(data)
    labels_array = np.array(labels)
    
    print(f"Dataset: {data_matrix.shape[1]} campioni, {data_matrix.shape[0]} features")
    print(f"Classe 0: {np.sum(labels_array == 0)}, Classe 1: {np.sum(labels_array == 1)}")
    
    return data_matrix, labels_array

def split_dataset_train_eval(data, labels, seed=0):
    """Divide il dataset in training e validation set (2/3 - 1/3)."""
    train_fraction = int(data.shape[1] * 2/3)
    np.random.seed(seed)
    indices = np.random.permutation(data.shape[1])
    train_indices = indices[:train_fraction]
    eval_indices = indices[train_fraction:]

    data_train = data[:, train_indices]
    labels_train = labels[train_indices]
    data_eval = data[:, eval_indices]
    labels_eval = labels[eval_indices]

    print(f"Split: {data_train.shape[1]} training, {data_eval.shape[1]} validation")
    
    return (data_train, labels_train), (data_eval, labels_eval)

# ---------------------------------------------------------------------
# PCA implementation
# ---------------------------------------------------------------------

def compute_mean_and_covariance(data):
    """Calcola media e matrice di covarianza."""
    mean_vec = as_column(data.mean(axis=1))
    centered = data - mean_vec
    cov_matrix = centered @ centered.T / data.shape[1]
    return mean_vec, cov_matrix

def compute_PCA(data, m):
    """Calcola la matrice di proiezione PCA con *m* componenti principali."""
    _, cov_matrix = compute_mean_and_covariance(data)
    U, s, _ = np.linalg.svd(cov_matrix)
    
    # Calcola varianza spiegata
    explained_variance_ratio = s / np.sum(s) * 100
    print(f"PCA {m}D: varianza spiegata = {np.sum(explained_variance_ratio[:m]):.1f}%")
    
    return U[:, :m]

def project_PCA(data, projection_matrix):
    """Proietta i dati usando la matrice di proiezione PCA."""
    return projection_matrix.T @ data

# ---------------------------------------------------------------------
# LDA implementation (binary classification)
# ---------------------------------------------------------------------

def compute_scatter_matrices(data, labels):
    """Calcola le matrici di scatter between-class (Sb) e within-class (Sw)."""
    Sb, Sw = 0, 0
    mu_global = as_column(data.mean(axis=1))
    
    for c in [0, 1]:
        class_data = data[:, labels == c]
        mu_class = as_column(class_data.mean(axis=1))
        
        # Between-class scatter
        diff = mu_class - mu_global
        Sb += class_data.shape[1] * (diff @ diff.T)
        
        # Within-class scatter
        Sw += (class_data - mu_class) @ (class_data - mu_class).T
    
    return Sb / data.shape[1], Sw / data.shape[1]

def compute_LDA(data, labels):
    """Calcola la direzione discriminante LDA per classificazione binaria."""
    Sb, Sw = compute_scatter_matrices(data, labels)
    eigenvalues, eigenvectors = scipy.linalg.eigh(Sb, Sw)
    
    # Prende l'autovettore con autovalore massimo
    w = eigenvectors[:, -1:]
    
    print(f"LDA: autovalore massimo = {eigenvalues[-1]:.3f}")
    
    return w

def project_LDA(data, w):
    """Proietta i dati usando la direzione discriminante LDA."""
    return w.T @ data

# ---------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------

def plot_pca_histograms(data_pca, labels, title="PCA"):
    """Crea istogrammi per le 6 componenti PCA."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(6):
        ax = axes[i]
        
        ax.hist(data_pca[i, labels == 0], bins=25, alpha=0.6, density=True,
                color='red', label='Fake')
        ax.hist(data_pca[i, labels == 1], bins=25, alpha=0.6, density=True,
                color='green', label='Genuine')
        
        ax.set_title(f"PC{i+1}")
        ax.set_xlabel(f"PC {i+1}")
        ax.set_ylabel("Density")
        if i == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Istogrammi PCA - 6 Componenti Principali")
    plt.tight_layout()
    plt.savefig(f"{title}_histograms.pdf")
    # plt.show()

def plot_pca_scatter_matrix(data_pca, labels, title="PCA"):
    """Crea scatter matrix per le combinazioni delle componenti PCA."""
    pairs = list(itertools.combinations(range(6), 2))
    n_pairs = len(pairs)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    # Disabilita assi non utilizzati
    for ax in axes[n_pairs:]:
        ax.axis('off')
    
    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]
        
        ax.scatter(data_pca[i, labels == 0], data_pca[j, labels == 0], 
                  s=8, color='red', alpha=0.6, label='Classe 0')
        ax.scatter(data_pca[i, labels == 1], data_pca[j, labels == 1], 
                  s=8, color='green', alpha=0.6, label='Classe 1')
        
        ax.set_xlabel(f"PC{i+1}")
        ax.set_ylabel(f"PC{j+1}")
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend()
    
    plt.suptitle("Scatter Matrix PCA - Combinazioni delle Componenti")
    plt.tight_layout()
    plt.savefig(f"{title}_scatter_matrix.pdf")
    # plt.show()

def plot_lda_histogram(data_lda, labels, title="LDA"):
    """Crea istogramma per la proiezione LDA."""
    plt.figure(figsize=(8, 6))
    
    plt.hist(data_lda[0, labels == 0], bins=30, alpha=0.6, density=True,
             color='red', label='Fake')
    plt.hist(data_lda[0, labels == 1], bins=30, alpha=0.6, density=True,
             color='green', label='Genuine')
    
    plt.xlabel("Linear Discriminant")
    plt.ylabel("Density")
    plt.title("Istogramma LDA - Proiezione 1D")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{title}_histogram.pdf")
    # plt.show()

# ---------------------------------------------------------------------
# Classification functions
# ---------------------------------------------------------------------

def lda_classify(data_train, labels_train, data_eval, threshold=None):
    """Classifica usando LDA con soglia specificata o ottimale."""
    # Calcola LDA
    w = compute_LDA(data_train, labels_train)
    
    # Proiezione
    proj_train = project_LDA(data_train, w)
    proj_eval = project_LDA(data_eval, w)
    
    # Orienta genuine (classe 1) a destra
    mean_0 = proj_train[0, labels_train == 0].mean()
    mean_1 = proj_train[0, labels_train == 1].mean()
    
    if mean_1 < mean_0:
        w = -w
        proj_train = project_LDA(data_train, w)
        proj_eval = project_LDA(data_eval, w)
        mean_0 = proj_train[0, labels_train == 0].mean()
        mean_1 = proj_train[0, labels_train == 1].mean()
    
    # Soglia di decisione
    if threshold is None:
        threshold = 0.5 * (mean_0 + mean_1)
    
    # Classificazione
    predictions = (proj_eval.ravel() >= threshold).astype(np.int32)
    
    return predictions, proj_eval, threshold

def test_different_thresholds(data_train, labels_train, data_eval, labels_eval):
    """Testa diverse soglie per ottimizzare la classificazione."""
    print("\n🔍 Test di diverse soglie...")
    
    # Ottieni proiezioni
    _, proj_eval, default_threshold = lda_classify(data_train, labels_train, data_eval)
    
    # Range di soglie
    min_val = proj_eval.min()
    max_val = proj_eval.max()
    thresholds = np.linspace(min_val, max_val, 100)
    
    best_accuracy = 0
    best_threshold = default_threshold
    
    for thr in thresholds:
        predictions = (proj_eval.ravel() >= thr).astype(np.int32)
        accuracy = 100 * (1 - (predictions != labels_eval).mean())
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = thr
    
    print(f"Soglia default: {default_threshold:.4f}")
    print(f"Soglia ottimale: {best_threshold:.4f}")
    print(f"Miglioramento: {best_accuracy - 100 * (1 - (lda_classify(data_train, labels_train, data_eval, default_threshold)[0] != labels_eval).mean()):.1f}%")
    
    return best_threshold, best_accuracy

# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("    ANALISI PROGETTO: PCA E LDA")
    print("="*60)
    
    # Caricamento dataset
    data, labels = load_project_data("trainData.csv")
    
    print(f"\n" + "="*50)
    print("FASE 1: ANALISI PCA")
    print("="*50)
    
    # PCA con 6 componenti su tutto il dataset
    P6 = compute_PCA(data, 6)
    data_pca = project_PCA(data, P6)
    
    # Visualizzazioni PCA
    plot_pca_histograms(data_pca, labels, "PCA_Full")
    plot_pca_scatter_matrix(data_pca, labels, "PCA_Full")
        
    print(f"\n" + "="*50)
    print("FASE 2: ANALISI LDA")
    print("="*50)
    
    # LDA su tutto il dataset
    w_full = compute_LDA(data, labels)
    proj_full = project_LDA(data, w_full)
    
    # Orienta per visualizzazione
    if proj_full[0, labels == 1].mean() < proj_full[0, labels == 0].mean():
        proj_full = -proj_full
    
    plot_lda_histogram(proj_full, labels, "LDA_Full")
        
    print(f"\n" + "="*50)
    print("FASE 3: CLASSIFICAZIONE")
    print("="*50)
    
    # Split del dataset
    (data_train, labels_train), (data_eval, labels_eval) = split_dataset_train_eval(data, labels)
    
    # Classificazione LDA baseline
    preds, _, threshold = lda_classify(data_train, labels_train, data_eval)
    error_rate = (preds != labels_eval).mean() * 100
    accuracy = 100 - error_rate
    
    print(f"\nLDA Baseline:")
    print(f"   Soglia: {threshold:.4f}")
    print(f"   Errore: {error_rate:.1f}%")
    print(f"   Accuratezza: {accuracy:.1f}%")
    
    # Test diverse soglie
    best_threshold, best_accuracy = test_different_thresholds(
        data_train, labels_train, data_eval, labels_eval
    )
    
    print(f"\n" + "="*50)
    print("FASE 4: PCA + LDA")
    print("="*50)
    
    print("\nPerformance PCA + LDA:")
    results = []
    
    for m in [5, 4, 3, 2, 1]:
        # PCA preprocessing
        Pm = compute_PCA(data_train, m)
        data_train_pca = project_PCA(data_train, Pm)
        data_eval_pca = project_PCA(data_eval, Pm)
        
        # LDA classification
        preds_pca, _, _ = lda_classify(data_train_pca, labels_train, data_eval_pca)
        error_rate_pca = (preds_pca != labels_eval).mean() * 100
        accuracy_pca = 100 - error_rate_pca
        
        results.append((m, accuracy_pca))
        print(f"   PCA({m}) + LDA: {error_rate_pca:.1f}% errore, {accuracy_pca:.1f}% accuratezza")
    
    print(f"\n" + "="*50)
    print("CONCLUSIONI")
    print("="*50)
    
    best_pca_m, best_pca_acc = max(results, key=lambda x: x[1])
    
    print(f"📊 Risultati finali:")
    print(f"   LDA Baseline: {accuracy:.1f}%")
    print(f"   LDA Ottimizzato: {best_accuracy:.1f}%")
    print(f"   Migliore PCA+LDA: {best_pca_acc:.1f}% (m={best_pca_m})")
    
    if best_pca_acc > accuracy:
        print(f"   → PCA preprocessing migliora di {best_pca_acc - accuracy:.1f}%")
    else:
        print(f"   → PCA preprocessing non migliora le performance")
    
    print(f"\n✅ Analisi completata!")
    print("="*60)