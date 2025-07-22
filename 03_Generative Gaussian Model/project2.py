import numpy as np
import scipy.special as sp
from pathlib import Path

# ---------------------------------------------------------------------
# ---------------------  NUMPY SHORTHANDS  ----------------------------
# ---------------------------------------------------------------------

def vcol(v: np.ndarray) -> np.ndarray:
    """Return *v* as a column vector (shape = [size, 1])."""
    return v.reshape(v.size, 1)

def vrow(v: np.ndarray) -> np.ndarray:
    """Return *v* as a row vector (shape = [1, size])."""
    return v.reshape(1, v.size)

# ---------------------------------------------------------------------
# ---------------------  DATA I/O  ------------------------------------
# ---------------------------------------------------------------------

def load_project_txt(path="trainData.csv"):
    """
    File format: f1,f2,f3,f4,f5,f6,label
    label = 1 (genuine)  |  0 (fake)
    Returns:
        D : (6,N)   float64   column-wise feature matrix
        L : (N,)    int32     label array (0/1)
    """
    samples, labels = [], []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split(',')
            if len(parts) != 7:               # skip malformed rows
                continue
            *feat, lab = parts
            samples.append(vcol(np.array(feat, dtype=np.float64)))
            labels.append(int(lab))
    D = np.hstack(samples)
    L = np.array(labels, dtype=np.int32)
    return D, L

def split_2to1(D: np.ndarray, L: np.ndarray, seed=0):
    """Random 2/3 vs 1/3 split."""
    n_tr = int(D.shape[1] * 2 / 3)
    np.random.seed(seed)
    perm = np.random.permutation(D.shape[1])
    idx_tr, idx_ev = perm[:n_tr], perm[n_tr:]
    return (D[:, idx_tr], L[idx_tr]), (D[:, idx_ev], L[idx_ev])

# ---------------------------------------------------------------------
# ---------------------  BASIC STATISTICS  ----------------------------
# ---------------------------------------------------------------------

def mean_vector(X: np.ndarray) -> np.ndarray:
    """Empirical mean μ = 1/N Σ xᵢ."""
    return vcol(X.mean(axis=1))

def covariance_matrix(X: np.ndarray) -> np.ndarray:
    """Empirical covariance Σ = 1/N Σ (xᵢ-μ)(xᵢ-μ)ᵀ."""
    centered = X - mean_vector(X)
    return centered @ centered.T / X.shape[1]

def diagonal_covariance_matrix(X: np.ndarray) -> np.ndarray:
    """Diagonal covariance matrix (Naive Bayes assumption)."""
    cov = covariance_matrix(X)
    return np.diag(np.diag(cov))

def correlation_matrix(C: np.ndarray) -> np.ndarray:
    """Compute correlation matrix from covariance matrix."""
    s = np.sqrt(np.diag(C))
    outer = vcol(s) @ vrow(s)
    return C / outer

# ---------------------------------------------------------------------
# ---------------------  MULTIVARIATE GAUSSIAN  -----------------------
# ---------------------------------------------------------------------

def logpdf_GAU_ND(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Log‑density of N‑D Gaussian for all columns in X."""
    M = X.shape[0]
    XC = X - mu
    invS = np.linalg.inv(Sigma)
    log_det = np.linalg.slogdet(Sigma)[1]
    quad = np.sum(XC * (invS @ XC), axis=0)
    return -0.5 * (M * np.log(2*np.pi) + log_det + quad)

# ---------------------------------------------------------------------
# ---------------------  PCA IMPLEMENTATION  --------------------------
# ---------------------------------------------------------------------

def PCA_projection_matrix(D_tr: np.ndarray, m: int):
    """Compute PCA projection matrix."""
    mu = mean_vector(D_tr)
    DC = D_tr - mu
    C = covariance_matrix(DC)
    s, U = np.linalg.eigh(C)      # asc order
    P = U[:, ::-1][:, :m]         # eigenvectors (desc eigenvals)
    return P, mu

def apply_PCA(P: np.ndarray, mu: np.ndarray, D: np.ndarray):
    """Apply PCA transformation."""
    return P.T @ (D - mu)

# ---------------------------------------------------------------------
# ---------------------  GAUSSIAN CLASSIFIERS  ------------------------
# ---------------------------------------------------------------------

class GaussianClassifier:
    """Gaussian classifier with support for MVG, Tied, and Naive Bayes."""
    
    def __init__(self, classifier_type="mvg"):
        self.classifier_type = classifier_type  # mvg, tied, naive_bayes
        self.mus = None
        self.covs = None
        self.Sigma_tied = None
        
    def train(self, D_tr: np.ndarray, L_tr: np.ndarray):
        """Train the classifier."""
        classes = np.unique(L_tr)
        self.mus = {}
        
        if self.classifier_type == "mvg":
            self.covs = {}
            for c in classes:
                Dc = D_tr[:, L_tr == c]
                self.mus[c] = mean_vector(Dc)
                self.covs[c] = covariance_matrix(Dc)
                
        elif self.classifier_type == "naive_bayes":
            self.covs = {}
            for c in classes:
                Dc = D_tr[:, L_tr == c]
                self.mus[c] = mean_vector(Dc)
                self.covs[c] = diagonal_covariance_matrix(Dc)
                
        elif self.classifier_type == "tied":
            N_tot = D_tr.shape[1]
            covs_temp = {}
            Ns = {}
            
            for c in classes:
                Dc = D_tr[:, L_tr == c]
                self.mus[c] = mean_vector(Dc)
                covs_temp[c] = covariance_matrix(Dc)
                Ns[c] = Dc.shape[1]
            
            # Pooled covariance
            self.Sigma_tied = sum(Ns[c] * covs_temp[c] for c in classes) / N_tot
    
    def compute_llr(self, D_ev: np.ndarray) -> np.ndarray:
        """Compute Log-Likelihood Ratios (LLR) with class 1 on top."""
        if self.classifier_type == "tied":
            log_lik_1 = logpdf_GAU_ND(D_ev, self.mus[1], self.Sigma_tied)
            log_lik_0 = logpdf_GAU_ND(D_ev, self.mus[0], self.Sigma_tied)
        else:
            log_lik_1 = logpdf_GAU_ND(D_ev, self.mus[1], self.covs[1])
            log_lik_0 = logpdf_GAU_ND(D_ev, self.mus[0], self.covs[0])
        
        return log_lik_1 - log_lik_0
    
    def predict_from_llr(self, llr: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """Compute predictions from LLR assuming uniform priors."""
        return (llr >= threshold).astype(int)
    
    def compute_error_rate(self, predictions: np.ndarray, L_true: np.ndarray) -> float:
        """Compute error rate from predictions."""
        return np.mean(predictions != L_true)

# ---------------------------------------------------------------------
# ---------------------  ANALYSIS FUNCTIONS  --------------------------
# ---------------------------------------------------------------------

def analyze_covariance_correlation(classifier: GaussianClassifier, class_names=["Fake", "Genuine"]):
    """Analyze covariance and correlation matrices."""
    print(f"\n📊 ANALISI COVARIANZA E CORRELAZIONE:")
    print("-" * 70)
    
    for c in [0, 1]:
        if classifier.classifier_type == "tied":
            cov_matrix = classifier.Sigma_tied
            print(f"\n{class_names[c]} (Classe {c}) - COVARIANZA CONDIVISA:")
        else:
            cov_matrix = classifier.covs[c]
            print(f"\n{class_names[c]} (Classe {c}):")
        
        print(f"Media μ: {classifier.mus[c].ravel()}")
        print(f"\nMatrice di Covarianza:")
        print(cov_matrix)
        
        # Correlation matrix
        corr_matrix = correlation_matrix(cov_matrix)
        print(f"\nMatrice di Correlazione:")
        print(corr_matrix)
        
        # Analysis of correlation strength
        off_diag = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        max_corr = np.max(np.abs(off_diag))
        mean_corr = np.mean(np.abs(off_diag))
        
        print(f"\nAnalisi correlazioni:")
        print(f"  Correlazione massima: {max_corr:.3f}")
        print(f"  Correlazione media: {mean_corr:.3f}")
        
        if mean_corr < 0.3:
            print(f"  → Features debolmente correlate (Naive Bayes ragionevole)")
        elif mean_corr < 0.7:
            print(f"  → Features moderatamente correlate")
        else:
            print(f"  → Features fortemente correlate (Naive Bayes problematico)")

def run_classification_experiment(name: str, D_tr: np.ndarray, L_tr: np.ndarray, 
                                D_ev: np.ndarray, L_ev: np.ndarray):
    """Run complete classification experiment with all three models."""
    print(f"\n🎯 ESPERIMENTO: {name}")
    print("=" * 60)
    
    results = {}
    classifiers = {
        "MVG": GaussianClassifier("mvg"),
        "Tied MVG": GaussianClassifier("tied"),
        "Naive Bayes": GaussianClassifier("naive_bayes")
    }
    
    for clf_name, clf in classifiers.items():
        # Train
        clf.train(D_tr, L_tr)
        
        # Evaluate
        llr = clf.compute_llr(D_ev)
        predictions = clf.predict_from_llr(llr)
        error_rate = clf.compute_error_rate(predictions, L_ev)
        accuracy = (1 - error_rate) * 100
        
        results[clf_name] = {
            'classifier': clf,
            'llr': llr,
            'predictions': predictions,
            'error_rate': error_rate,
            'accuracy': accuracy
        }
        
        print(f"{clf_name:<12}: Error = {error_rate*100:5.2f}%  |  Accuracy = {accuracy:5.2f}%")
    
    return results

def subset_features(D: np.ndarray, feature_indices: np.ndarray) -> np.ndarray:
    """Extract subset of features."""
    return D[feature_indices, :]

# ---------------------------------------------------------------------
# ---------------------  MAIN ANALYSIS  -------------------------------
# ---------------------------------------------------------------------

def main():
    print("=" * 80)
    print("    ANALISI CLASSIFICATORI GAUSSIANI - PROGETTO FINGERPRINT")
    print("=" * 80)
    
    # 1) Load and split data
    print("\n🔄 Caricamento dataset progetto...")
    D, L = load_project_txt("trainData.csv")
    (D_tr, L_tr), (D_ev, L_ev) = split_2to1(D, L, seed=0)
    
    print(f"Dataset: {D.shape[1]} campioni, {D.shape[0]} features")
    print(f"Training: {D_tr.shape[1]} campioni  |  Validation: {D_ev.shape[1]} campioni")
    
    class_counts = [np.sum(L == c) for c in [0, 1]]
    print(f"Classe 0 (Fake): {class_counts[0]} campioni")
    print(f"Classe 1 (Genuine): {class_counts[1]} campioni")
    
    # 2) Full 6-feature analysis
    print(f"\n" + "="*80)
    print("PARTE 1: ANALISI COMPLETA (6 FEATURES)")
    print("="*80)
    
    results_full = run_classification_experiment("FULL 6-D FEATURE SPACE", 
                                                D_tr, L_tr, D_ev, L_ev)
    
    # 3) Covariance and correlation analysis
    print(f"\n📊 ANALISI DETTAGLIATA COVARIANZA:")
    mvg_classifier = results_full["MVG"]["classifier"]
    analyze_covariance_correlation(mvg_classifier)
    
    # 4) Features 1-4 analysis
    print(f"\n" + "="*80)
    print("PARTE 2: ANALISI FEATURES 1-4 (scartando le ultime 2)")
    print("="*80)
    
    idx_1_4 = np.array([0, 1, 2, 3])
    D_tr_1_4 = subset_features(D_tr, idx_1_4)
    D_ev_1_4 = subset_features(D_ev, idx_1_4)
    
    results_1_4 = run_classification_experiment("FEATURES 1-4", 
                                               D_tr_1_4, L_tr, D_ev_1_4, L_ev)
    
    # 5) Features 1-2 vs 3-4 analysis
    print(f"\n" + "="*80)
    print("PARTE 3: CONFRONTO FEATURES 1-2 vs 3-4")
    print("="*80)
    
    # Features 1-2
    idx_1_2 = np.array([0, 1])
    D_tr_1_2 = subset_features(D_tr, idx_1_2)
    D_ev_1_2 = subset_features(D_ev, idx_1_2)
    
    results_1_2 = run_classification_experiment("FEATURES 1-2", 
                                               D_tr_1_2, L_tr, D_ev_1_2, L_ev)
    
    # Features 3-4
    idx_3_4 = np.array([2, 3])
    D_tr_3_4 = subset_features(D_tr, idx_3_4)
    D_ev_3_4 = subset_features(D_ev, idx_3_4)
    
    results_3_4 = run_classification_experiment("FEATURES 3-4", 
                                               D_tr_3_4, L_tr, D_ev_3_4, L_ev)
    
    # 6) PCA analysis
    print(f"\n" + "="*80)
    print("PARTE 4: ANALISI PCA PRE-PROCESSING")
    print("="*80)
    
    pca_results = {}
    for m in [5, 4, 3, 2]:
        print(f"\n→ PCA con m = {m} componenti:")
        
        # Apply PCA
        P, mu = PCA_projection_matrix(D_tr, m)
        D_tr_pca = apply_PCA(P, mu, D_tr)
        D_ev_pca = apply_PCA(P, mu, D_ev)
        
        pca_results[m] = run_classification_experiment(f"PCA m={m}", 
                                                      D_tr_pca, L_tr, D_ev_pca, L_ev)
    
    # 7) Final comparison and interpretation
    print(f"\n" + "="*80)
    print("RIEPILOGO E INTERPRETAZIONE RISULTATI")
    print("="*80)
    
    # Collect all results for comparison
    all_experiments = {
        "Full 6D": results_full,
        "Features 1-4": results_1_4,
        "Features 1-2": results_1_2,
        "Features 3-4": results_3_4,
    }
    
    # Add PCA results
    for m in [5, 4, 3, 2]:
        all_experiments[f"PCA m={m}"] = pca_results[m]
    
    print(f"\n🏆 RANKING ACCURATEZZA PER ESPERIMENTO:")
    print("-" * 60)
    
    for exp_name, exp_results in all_experiments.items():
        print(f"\n{exp_name}:")
        sorted_models = sorted(exp_results.items(), 
                             key=lambda x: x[1]['accuracy'], reverse=True)
        
        for i, (model_name, result) in enumerate(sorted_models, 1):
            print(f"  {i}. {model_name:<12}: {result['accuracy']:5.1f}%")
    
    # Find overall best result
    best_overall = None
    best_acc = 0
    best_exp = ""
    best_model = ""
    
    for exp_name, exp_results in all_experiments.items():
        for model_name, result in exp_results.items():
            if result['accuracy'] > best_acc:
                best_acc = result['accuracy']
                best_exp = exp_name
                best_model = model_name
                best_overall = result
    
    print(f"\n🎯 MIGLIOR RISULTATO ASSOLUTO:")
    print(f"  Esperimento: {best_exp}")
    print(f"  Modello: {best_model}")
    print(f"  Accuratezza: {best_acc:.2f}%")
    
    # Interpretation
    print(f"\n💡 INTERPRETAZIONE RISULTATI:")
    print("-" * 60)
    
    # Compare MVG vs Tied vs Naive Bayes on full data
    mvg_acc = results_full["MVG"]["accuracy"]
    tied_acc = results_full["Tied MVG"]["accuracy"]
    nb_acc = results_full["Naive Bayes"]["accuracy"]
    
    print(f"\n1. CONFRONTO MODELLI (6D completo):")
    print(f"   MVG: {mvg_acc:.1f}% | Tied: {tied_acc:.1f}% | Naive Bayes: {nb_acc:.1f}%")
    
    if mvg_acc > tied_acc:
        print(f"   → MVG > Tied: le classi hanno strutture di covarianza diverse")
    else:
        print(f"   → Tied ≥ MVG: le classi condividono forma di covarianza simile")
    
    if nb_acc < mvg_acc:
        diff = mvg_acc - nb_acc
        if diff > 5:
            print(f"   → Naive Bayes molto peggiore ({diff:.1f}%): features fortemente correlate")
        else:
            print(f"   → Naive Bayes leggermente peggiore: correlazioni moderate")
    
    # Effect of discarding features 5-6
    print(f"\n2. EFFETTO RIMOZIONE FEATURES 5-6:")
    full_best = max(results_full.values(), key=lambda x: x['accuracy'])['accuracy']
    feat14_best = max(results_1_4.values(), key=lambda x: x['accuracy'])['accuracy']
    
    if feat14_best > full_best:
        print(f"   → Miglioramento ({feat14_best:.1f}% vs {full_best:.1f}%): features 5-6 erano rumorose")
    else:
        print(f"   → Peggioramento ({feat14_best:.1f}% vs {full_best:.1f}%): features 5-6 contenevano info utili")
    
    # Features 1-2 vs 3-4 analysis
    print(f"\n3. CONFRONTO FEATURES 1-2 vs 3-4:")
    
    mvg_12 = results_1_2["MVG"]["accuracy"]
    tied_12 = results_1_2["Tied MVG"]["accuracy"]
    mvg_34 = results_3_4["MVG"]["accuracy"]
    tied_34 = results_3_4["Tied MVG"]["accuracy"]
    
    print(f"   Features 1-2: MVG {mvg_12:.1f}% | Tied {tied_12:.1f}%")
    print(f"   Features 3-4: MVG {mvg_34:.1f}% | Tied {tied_34:.1f}%")
    
    if abs(mvg_12 - tied_12) > abs(mvg_34 - tied_34):
        print(f"   → Tied meno efficace su 1-2: classi differiscono in varianza")
        print(f"   → Tied più efficace su 3-4: classi differiscono principalmente in media")
    
    # PCA analysis
    print(f"\n4. EFFETTO PCA:")
    pca_best_accs = [max(pca_results[m].values(), key=lambda x: x['accuracy'])['accuracy'] 
                     for m in [5, 4, 3, 2]]
    
    if max(pca_best_accs) > full_best:
        print(f"   → PCA utile: riduce dimensionalità mantenendo/migliorando performance")
        best_pca_m = [5, 4, 3, 2][np.argmax(pca_best_accs)]
        print(f"   → Miglior dimensionalità PCA: m = {best_pca_m}")
    else:
        print(f"   → PCA dannoso: perde informazioni discriminative importanti")
    
    print(f"\n" + "="*80)
    print("CONCLUSIONI:")
    print(f"Il miglior modello è {best_model} con {best_exp}")
    print(f"raggiungendo un'accuratezza del {best_acc:.2f}% sul validation set")
    print("="*80)

if __name__ == "__main__":
    main()