import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pickle
from pathlib import Path

# Import bayesRisk functions
import sys
sys.path.append('/home/bernuz/Scrivania/UNIVERSITA\'/Machine Learning and Pattern Recognition/Labs/11_Support Vector Machine')
import bayesRisk

# ---------------------------------------------------------------------
# ---------------------  UTILITY FUNCTIONS  ---------------------------
# ---------------------------------------------------------------------

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def load_project_data(path="trainData.csv"):
    """Load project data."""
    samples, labels = [], []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split(',')
            if len(parts) != 7:
                continue
            *feat, lab = parts
            samples.append(vcol(np.array(feat, dtype=np.float64)))
            labels.append(int(lab))
    D = np.hstack(samples)
    L = np.array(labels, dtype=np.int32)
    return D, L

def split_db_2to1(D, L, seed=0):
    """Split dataset 2/3 training, 1/3 validation"""
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

def center_data(DTR, DVAL):
    """Center data using training mean"""
    mu = DTR.mean(axis=1, keepdims=True)
    DTR_centered = DTR - mu
    DVAL_centered = DVAL - mu
    return DTR_centered, DVAL_centered, mu

def sanitize_filename(filename):
    """Remove invalid characters from filename"""
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '(', ')', ' ']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

# ---------------------------------------------------------------------
# ---------------------  SVM IMPLEMENTATIONS  -------------------------
# ---------------------------------------------------------------------

def train_dual_SVM_linear(DTR, LTR, C, K=1.0):
    """Train linear SVM using dual formulation"""
    ZTR = LTR * 2.0 - 1.0  # Convert labels to +1/-1
    DTR_EXT = np.vstack([DTR, np.ones((1, DTR.shape[1])) * K])
    H = np.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR)

    # Dual objective with gradient
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - np.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(
        fOpt, np.zeros(DTR_EXT.shape[1]), 
        bounds=[(0, C) for i in LTR], 
        factr=1e7, pgtol=1e-5
    )
    
    # Compute primal solution for extended data matrix
    w_hat = (vrow(alphaStar) * vrow(ZTR) * DTR_EXT).sum(1)
    
    # Extract w and b
    w, b = w_hat[0:DTR.shape[0]], w_hat[-1] * K
    
    # Compute losses
    def primalLoss(w_hat):
        S = (vrow(w_hat) @ DTR_EXT).ravel()
        return 0.5 * np.linalg.norm(w_hat)**2 + C * np.maximum(0, 1 - ZTR * S).sum()
    
    primalLoss_val, dualLoss_val = primalLoss(w_hat), -fOpt(alphaStar)[0]
    # Fix: Convert array to scalar
    duality_gap = (primalLoss_val - dualLoss_val).item() if hasattr(primalLoss_val - dualLoss_val, 'item') else primalLoss_val - dualLoss_val
    
    return w, b, primalLoss_val, dualLoss_val, duality_gap

def polyKernel(degree, c):
    """Create polynomial kernel function"""
    def polyKernelFunc(D1, D2):
        return (np.dot(D1.T, D2) + c) ** degree
    return polyKernelFunc

def rbfKernel(gamma):
    """Create RBF kernel function"""
    def rbfKernelFunc(D1, D2):
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * np.dot(D1.T, D2)
        return np.exp(-gamma * Z)
    return rbfKernelFunc

def train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps=1.0):
    """Train SVM with kernel using dual formulation"""
    ZTR = LTR * 2.0 - 1.0  # Convert labels to +1/-1
    K = kernelFunc(DTR, DTR) + eps
    H = vcol(ZTR) * vrow(ZTR) * K

    # Dual objective with gradient
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - np.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(
        fOpt, np.zeros(DTR.shape[1]), 
        bounds=[(0, C) for i in LTR], 
        factr=1e7, pgtol=1e-5
    )

    # Compute losses
    def primalLoss(alpha):
        Ha = H @ vcol(alpha)
        return 0.5 * (vrow(alpha) @ Ha).sum() + C * np.maximum(0, 1 - Ha).sum()

    primalLoss_val, dualLoss_val = primalLoss(alphaStar), -fOpt(alphaStar)[0]
    # Fix: Convert array to scalar
    duality_gap = (primalLoss_val - dualLoss_val).item() if hasattr(primalLoss_val - dualLoss_val, 'item') else primalLoss_val - dualLoss_val

    # Function to compute scores for test samples
    def fScore(DTE):
        K_test = kernelFunc(DTR, DTE) + eps
        H_test = vcol(alphaStar) * vcol(ZTR) * K_test
        return H_test.sum(0)

    return fScore, primalLoss_val, dualLoss_val, duality_gap

# ---------------------------------------------------------------------
# ---------------------  EVALUATION FUNCTIONS  ------------------------
# ---------------------------------------------------------------------

def evaluate_svm_model(scores, LVAL, target_prior=0.1):
    """Evaluate SVM model and compute DCF metrics"""
    minDCF = bayesRisk.compute_minDCF_binary_fast(scores, LVAL, target_prior, 1.0, 1.0)
    actDCF = bayesRisk.compute_actDCF_binary_fast(scores, LVAL, target_prior, 1.0, 1.0)
    return minDCF, actDCF

# ---------------------------------------------------------------------
# ---------------------  ANALYSIS FUNCTIONS  --------------------------
# ---------------------------------------------------------------------

def analyze_linear_svm(DTR, LTR, DVAL, LVAL, title_suffix="", centered=False):
    """Analyze linear SVM with different C values"""
    print(f"\n{'='*80}")
    print(f"⚔️  ANALISI SVM LINEARE - {title_suffix}")
    print(f"{'='*80}")
    
    if centered:
        print(f"📊 Dataset: Centrato (mean-centered)")
    else:
        print(f"📊 Dataset: Originale (non-centered)")
    
    print(f"📊 Training samples: {DTR.shape[1]}")
    print(f"📊 Validation samples: {DVAL.shape[1]}")
    print(f"📊 Features: {DTR.shape[0]}")
    print(f"📊 K = 1.0 (fisso)")
    
    # C values - logarithmic scale
    C_values = np.logspace(-5, 0, 11)
    
    print(f"\n📈 Valori di C da testare: {len(C_values)}")
    print(f"📈 Range C: [{C_values[0]:.1e}, {C_values[-1]:.1e}]")
    print(f"📈 Nota: C basso = regolarizzazione forte, C alto = regolarizzazione debole")
    
    minDCFs = []
    actDCFs = []
    duality_gaps = []
    
    print(f"\n🚀 TRAINING E VALUTAZIONE:")
    print("-" * 80)
    
    for i, C in enumerate(C_values):
        print(f"C = {C:.1e} ({i+1:2d}/{len(C_values)}) ", end="", flush=True)
        
        # Train model
        w, b, primal_loss, dual_loss, gap = train_dual_SVM_linear(DTR, LTR, C)
        
        # Compute scores
        scores = (vrow(w) @ DVAL + b).ravel()
        
        # Evaluate
        minDCF, actDCF = evaluate_svm_model(scores, LVAL)
        
        minDCFs.append(minDCF)
        actDCFs.append(actDCF)
        duality_gaps.append(gap)
        
        print(f"→ minDCF: {minDCF:.4f}, actDCF: {actDCF:.4f}, gap: {gap:.2e}")
    
    # Find best C
    best_idx = np.argmin(minDCFs)
    best_C = C_values[best_idx]
    
    print(f"\n🏆 RISULTATI MIGLIORI:")
    print("-" * 50)
    print(f"📍 Miglior C: {best_C:.1e}")
    print(f"📍 Miglior minDCF: {minDCFs[best_idx]:.4f}")
    print(f"📍 Corrispondente actDCF: {actDCFs[best_idx]:.4f}")
    print(f"📍 Gap calibrazione: {actDCFs[best_idx] - minDCFs[best_idx]:.4f}")
    
    # Calibration analysis
    calibration_gap = actDCFs[best_idx] - minDCFs[best_idx]
    if calibration_gap > 0.1:
        print(f"🔧 CALIBRAZIONE: Pessima (gap > 0.1) - necessaria calibrazione")
    elif calibration_gap > 0.05:
        print(f"⚠️  CALIBRAZIONE: Mediocre (gap > 0.05) - calibrazione consigliata")
    else:
        print(f"✅ CALIBRAZIONE: Buona (gap < 0.05)")
    
    # Plot results
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 3, 1)
    plt.semilogx(C_values, minDCFs, 'b-o', label='minDCF', linewidth=2, markersize=6)
    plt.semilogx(C_values, actDCFs, 'r-s', label='actDCF', linewidth=2, markersize=6)
    plt.axvline(best_C, color='green', linestyle='--', alpha=0.7, label=f'Best C={best_C:.1e}')
    plt.xlabel('C (Regularization Parameter)')
    plt.ylabel('DCF')
    plt.title(f'Linear SVM: DCF vs C - {title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.semilogx(C_values, np.array(actDCFs) - np.array(minDCFs), 'g-^', 
                 linewidth=2, markersize=6, label='actDCF - minDCF')
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(0.05, color='orange', linestyle='--', alpha=0.5, label='Fair threshold')
    plt.axhline(0.1, color='red', linestyle='--', alpha=0.5, label='Poor threshold')
    plt.axvline(best_C, color='green', linestyle='--', alpha=0.7)
    plt.xlabel('C (Regularization Parameter)')
    plt.ylabel('Calibration Gap')
    plt.title(f'Calibration Quality - {title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.loglog(C_values, duality_gaps, 'purple', marker='D', linewidth=2, markersize=6)
    plt.xlabel('C (Regularization Parameter)')
    plt.ylabel('Duality Gap')
    plt.title(f'Optimization Quality - {title_suffix}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    safe_filename = sanitize_filename(f'linear_svm_{title_suffix}')
    plt.savefig(f'{safe_filename}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return C_values, minDCFs, actDCFs, best_C

def analyze_polynomial_svm(DTR, LTR, DVAL, LVAL):
    """Analyze polynomial SVM with degree=2, c=1"""
    print(f"\n{'='*80}")
    print(f"🔮 ANALISI SVM POLINOMIALE (d=2, c=1)")
    print(f"{'='*80}")
    
    print(f"📊 Kernel: Polinomiale di grado 2")
    print(f"📊 Parametro c: 1 (bias implicito)")
    print(f"📊 ξ (eps): 0 (bias già incluso nel kernel)")
    print(f"📊 Training samples: {DTR.shape[1]}")
    print(f"📊 Validation samples: {DVAL.shape[1]}")
    
    # C values
    C_values = np.logspace(-5, 0, 11)
    kernel_func = polyKernel(2, 1)
    
    print(f"\n📈 Valori di C da testare: {len(C_values)}")
    print(f"📈 Range C: [{C_values[0]:.1e}, {C_values[-1]:.1e}]")
    
    minDCFs = []
    actDCFs = []
    duality_gaps = []
    
    print(f"\n🚀 TRAINING E VALUTAZIONE:")
    print("-" * 80)
    
    for i, C in enumerate(C_values):
        print(f"C = {C:.1e} ({i+1:2d}/{len(C_values)}) ", end="", flush=True)
        
        # Train model
        fScore, primal_loss, dual_loss, gap = train_dual_SVM_kernel(DTR, LTR, C, kernel_func, eps=0.0)
        
        # Compute scores
        scores = fScore(DVAL)
        
        # Evaluate
        minDCF, actDCF = evaluate_svm_model(scores, LVAL)
        
        minDCFs.append(minDCF)
        actDCFs.append(actDCF)
        duality_gaps.append(gap)
        
        print(f"→ minDCF: {minDCF:.4f}, actDCF: {actDCF:.4f}, gap: {gap:.2e}")
    
    # Find best C
    best_idx = np.argmin(minDCFs)
    best_C = C_values[best_idx]
    
    print(f"\n🏆 RISULTATI MIGLIORI:")
    print("-" * 50)
    print(f"📍 Miglior C: {best_C:.1e}")
    print(f"📍 Miglior minDCF: {minDCFs[best_idx]:.4f}")
    print(f"📍 Corrispondente actDCF: {actDCFs[best_idx]:.4f}")
    print(f"📍 Gap calibrazione: {actDCFs[best_idx] - minDCFs[best_idx]:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.semilogx(C_values, minDCFs, 'b-o', label='minDCF', linewidth=2, markersize=6)
    plt.semilogx(C_values, actDCFs, 'r-s', label='actDCF', linewidth=2, markersize=6)
    plt.axvline(best_C, color='green', linestyle='--', alpha=0.7, label=f'Best C={best_C:.1e}')
    plt.xlabel('C (Regularization Parameter)')
    plt.ylabel('DCF')
    plt.title('Polynomial SVM (d=2, c=1): DCF vs C')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogx(C_values, np.array(actDCFs) - np.array(minDCFs), 'g-^', 
                 linewidth=2, markersize=6, label='Calibration Gap')
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(0.05, color='orange', linestyle='--', alpha=0.5, label='Fair threshold')
    plt.axhline(0.1, color='red', linestyle='--', alpha=0.5, label='Poor threshold')
    plt.axvline(best_C, color='green', linestyle='--', alpha=0.7)
    plt.xlabel('C (Regularization Parameter)')
    plt.ylabel('actDCF - minDCF')
    plt.title('Polynomial SVM: Calibration Quality')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('polynomial_svm_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return C_values, minDCFs, actDCFs, best_C

def analyze_rbf_svm(DTR, LTR, DVAL, LVAL):
    """Analyze RBF SVM with grid search over gamma and C"""
    print(f"\n{'='*80}")
    print(f"🌀 ANALISI SVM RBF - GRID SEARCH (γ, C)")
    print(f"{'='*80}")
    
    print(f"📊 Kernel: RBF (Radial Basis Function)")
    print(f"📊 ξ (eps): 1.0 (bias term)")
    print(f"📊 Training samples: {DTR.shape[1]}")
    print(f"📊 Validation samples: {DVAL.shape[1]}")
    
    # Parameter grids
    gamma_values = [np.exp(-4), np.exp(-3), np.exp(-2), np.exp(-1)]
    C_values = np.logspace(-3, 2, 11)
    
    print(f"\n📈 Grid Search Parameters:")
    print(f"📈 γ values: {[f'{g:.3f}' for g in gamma_values]} (4 valori)")
    print(f"📈 C values: [{C_values[0]:.1e}, ..., {C_values[-1]:.1e}] ({len(C_values)} valori)")
    print(f"📈 Combinazioni totali: {len(gamma_values)} × {len(C_values)} = {len(gamma_values) * len(C_values)}")
    
    # Results storage
    results = {}
    best_result = {'minDCF': float('inf'), 'gamma': None, 'C': None, 'actDCF': None}
    
    print(f"\n🚀 GRID SEARCH - TRAINING E VALUTAZIONE:")
    print("=" * 100)
    
    for gamma_idx, gamma in enumerate(gamma_values):
        print(f"\n🔄 γ = {gamma:.3f} ({gamma_idx+1}/{len(gamma_values)})")
        print("-" * 80)
        
        kernel_func = rbfKernel(gamma)
        minDCFs_gamma = []
        actDCFs_gamma = []
        
        for c_idx, C in enumerate(C_values):
            print(f"  C = {C:.1e} ({c_idx+1:2d}/{len(C_values)}) ", end="", flush=True)
            
            # Train model
            fScore, primal_loss, dual_loss, gap = train_dual_SVM_kernel(DTR, LTR, C, kernel_func, eps=1.0)
            
            # Compute scores
            scores = fScore(DVAL)
            
            # Evaluate
            minDCF, actDCF = evaluate_svm_model(scores, LVAL)
            
            minDCFs_gamma.append(minDCF)
            actDCFs_gamma.append(actDCF)
            
            # Track best result
            if minDCF < best_result['minDCF']:
                best_result.update({
                    'minDCF': minDCF, 'actDCF': actDCF, 
                    'gamma': gamma, 'C': C
                })
            
            print(f"→ minDCF: {minDCF:.4f}, actDCF: {actDCF:.4f}")
        
        results[gamma] = {
            'C_values': C_values.copy(),
            'minDCFs': minDCFs_gamma.copy(),
            'actDCFs': actDCFs_gamma.copy()
        }
    
    print(f"\n🏆 MIGLIOR RISULTATO GLOBALE:")
    print("-" * 50)
    print(f"📍 Miglior γ: {best_result['gamma']:.3f}")
    print(f"📍 Miglior C: {best_result['C']:.1e}")
    print(f"📍 Miglior minDCF: {best_result['minDCF']:.4f}")
    print(f"📍 Corrispondente actDCF: {best_result['actDCF']:.4f}")
    print(f"📍 Gap calibrazione: {best_result['actDCF'] - best_result['minDCF']:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 8))
    
    colors = ['blue', 'red', 'green', 'purple']
    markers = ['o', 's', '^', 'D']
    
    plt.subplot(2, 2, 1)
    for i, gamma in enumerate(gamma_values):
        plt.semilogx(results[gamma]['C_values'], results[gamma]['minDCFs'], 
                     color=colors[i], marker=markers[i], linewidth=2, markersize=6,
                     label=f'γ = {gamma:.3f}')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.title('RBF SVM: minDCF vs C for different γ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    for i, gamma in enumerate(gamma_values):
        plt.semilogx(results[gamma]['C_values'], results[gamma]['actDCFs'], 
                     color=colors[i], marker=markers[i], linewidth=2, markersize=6,
                     label=f'γ = {gamma:.3f}')
    plt.xlabel('C')
    plt.ylabel('actDCF')
    plt.title('RBF SVM: actDCF vs C for different γ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    for i, gamma in enumerate(gamma_values):
        calibration_gaps = np.array(results[gamma]['actDCFs']) - np.array(results[gamma]['minDCFs'])
        plt.semilogx(results[gamma]['C_values'], calibration_gaps, 
                     color=colors[i], marker=markers[i], linewidth=2, markersize=6,
                     label=f'γ = {gamma:.3f}')
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(0.05, color='orange', linestyle='--', alpha=0.5, label='Fair threshold')
    plt.axhline(0.1, color='red', linestyle='--', alpha=0.5, label='Poor threshold')
    plt.xlabel('C')
    plt.ylabel('Calibration Gap (actDCF - minDCF)')
    plt.title('RBF SVM: Calibration Quality')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Heatmap of minDCF
    plt.subplot(2, 2, 4)
    minDCF_matrix = np.array([[results[gamma]['minDCFs'][c_idx] 
                              for c_idx in range(len(C_values))] 
                             for gamma in gamma_values])
    
    im = plt.imshow(minDCF_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='minDCF')
    plt.xticks(range(len(C_values)), [f'{c:.1e}' for c in C_values], rotation=45)
    plt.yticks(range(len(gamma_values)), [f'{g:.3f}' for g in gamma_values])
    plt.xlabel('C')
    plt.ylabel('γ')
    plt.title('RBF SVM: minDCF Heatmap')
    
    # Mark best point
    best_gamma_idx = list(gamma_values).index(best_result['gamma'])
    best_C_idx = list(C_values).index(best_result['C'])
    plt.plot(best_C_idx, best_gamma_idx, 'r*', markersize=15, label='Best')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('rbf_svm_grid_search.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results, best_result

def compare_all_svm_models(results_dict):
    """Compare all SVM models and previous results"""
    print(f"\n{'='*80}")
    print(f"🏆 CONFRONTO FINALE - TUTTI I MODELLI SVM")
    print(f"{'='*80}")
    
    model_names = []
    best_minDCFs = []
    best_actDCFs = []
    best_params = []
    
    # Extract best results
    for model_name, result in results_dict.items():
        if 'RBF' in model_name:
            model_names.append(model_name)
            best_minDCFs.append(result['minDCF'])
            best_actDCFs.append(result['actDCF'])
            best_params.append(f"γ={result['gamma']:.3f}, C={result['C']:.1e}")
        else:
            C_values, minDCFs, actDCFs, best_C = result
            best_idx = np.argmin(minDCFs)
            model_names.append(model_name)
            best_minDCFs.append(minDCFs[best_idx])
            best_actDCFs.append(actDCFs[best_idx])
            best_params.append(f"C={best_C:.1e}")
    
    # Sort by minDCF
    sorted_indices = np.argsort(best_minDCFs)
    
    print(f"\n📊 RANKING SVM MODELS (πT = 0.1):")
    print("-" * 90)
    print(f"{'Rank':<4} {'Model':<25} {'minDCF':<8} {'actDCF':<8} {'Calibration':<12} {'Best Parameters':<20}")
    print("-" * 90)
    
    for i, idx in enumerate(sorted_indices):
        calibration_gap = best_actDCFs[idx] - best_minDCFs[idx]
        calibration_status = "Good" if calibration_gap < 0.05 else "Poor" if calibration_gap > 0.1 else "Fair"
        
        print(f"{i+1:<4} {model_names[idx]:<25} {best_minDCFs[idx]:<8.4f} {best_actDCFs[idx]:<8.4f} "
              f"{calibration_status:<12} {best_params[idx]:<20}")
    
    # Visual comparison
    plt.figure(figsize=(15, 8))
    
    x_pos = np.arange(len(model_names))
    
    plt.subplot(1, 2, 1)
    bars1 = plt.bar(x_pos - 0.2, best_minDCFs, 0.4, label='minDCF', alpha=0.8, color='skyblue')
    bars2 = plt.bar(x_pos + 0.2, best_actDCFs, 0.4, label='actDCF', alpha=0.8, color='lightcoral')
    
    plt.xlabel('SVM Models')
    plt.ylabel('DCF')
    plt.title('SVM Models Comparison: DCF Values')
    plt.xticks(x_pos, model_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.subplot(1, 2, 2)
    calibration_gaps = np.array(best_actDCFs) - np.array(best_minDCFs)
    colors = ['green' if gap < 0.05 else 'orange' if gap < 0.1 else 'red' for gap in calibration_gaps]
    
    bars = plt.bar(x_pos, calibration_gaps, color=colors, alpha=0.7)
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(0.05, color='green', linestyle='--', alpha=0.5, label='Good calibration (<0.05)')
    plt.axhline(0.1, color='orange', linestyle='--', alpha=0.5, label='Fair calibration (<0.1)')
    
    plt.xlabel('SVM Models')
    plt.ylabel('Calibration Gap (actDCF - minDCF)')
    plt.title('SVM Models: Calibration Quality')
    plt.xticks(x_pos, model_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('svm_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return sorted_indices, model_names, best_minDCFs, best_actDCFs

# ---------------------------------------------------------------------
# ---------------------  MAIN EXECUTION  ------------------------------
# ---------------------------------------------------------------------

def main():
    print("=" * 80)
    print("    ANALISI SUPPORT VECTOR MACHINE - PROGETTO ML&PR")
    print("=" * 80)
    
    # Load data
    print("\n🔄 Caricamento dataset...")
    D, L = load_project_data()
    print(f"📊 Dataset caricato: {D.shape[1]} campioni, {D.shape[0]} features")
    print(f"📊 Distribuzione classi: Classe 0: {np.sum(L==0)}, Classe 1: {np.sum(L==1)}")
    
    # Split data
    print(f"\n📊 Split dataset (2/3 training, 1/3 validation)...")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    print(f"📊 Training: {DTR.shape[1]} campioni")
    print(f"📊 Validation: {DVAL.shape[1]} campioni")
    
    # Optional: Use subset for faster development
    USE_SUBSET = False  # Set to True for faster development
    if USE_SUBSET:
        print(f"\n⚠️  MODALITÀ SVILUPPO: Usando subset del dataset per velocizzare...")
        subset_size = min(1000, DTR.shape[1])
        DTR = DTR[:, :subset_size]
        LTR = LTR[:subset_size]
        print(f"📊 Training ridotto: {DTR.shape[1]} campioni")
    
    # Center data for comparison
    DTR_centered, DVAL_centered, mu = center_data(DTR, DVAL)
    
    # Store results
    results = {}
    
    # 1. Linear SVM - Original Data
    print(f"\n" + "⚔️" * 40)
    print("1️⃣  SVM LINEARE - DATI ORIGINALI")
    print("⚔️" * 40)
    
    C_values, minDCFs, actDCFs, best_C = analyze_linear_svm(
        DTR, LTR, DVAL, LVAL, "Original Data"
    )
    results["Linear SVM (Original)"] = (C_values, minDCFs, actDCFs, best_C)
    
    # 2. Linear SVM - Centered Data
    print(f"\n" + "⚔️" * 40)
    print("2️⃣  SVM LINEARE - DATI CENTRATI")
    print("⚔️" * 40)
    
    C_values_c, minDCFs_c, actDCFs_c, best_C_c = analyze_linear_svm(
        DTR_centered, LTR, DVAL_centered, LVAL, "Centered Data", centered=True
    )
    results["Linear SVM (Centered)"] = (C_values_c, minDCFs_c, actDCFs_c, best_C_c)
    
    # 3. Polynomial SVM
    print(f"\n" + "⚔️" * 40)
    print("3️⃣  SVM POLINOMIALE (d=2, c=1)")
    print("⚔️" * 40)
    
    C_values_p, minDCFs_p, actDCFs_p, best_C_p = analyze_polynomial_svm(DTR, LTR, DVAL, LVAL)
    results["Polynomial SVM (d=2, c=1)"] = (C_values_p, minDCFs_p, actDCFs_p, best_C_p)
    
    # 4. RBF SVM - Grid Search
    print(f"\n" + "⚔️" * 40)
    print("4️⃣  SVM RBF - GRID SEARCH")
    print("⚔️" * 40)
    
    rbf_results, best_rbf = analyze_rbf_svm(DTR, LTR, DVAL, LVAL)
    results["RBF SVM (Grid Search)"] = best_rbf
    
    # 5. Final Comparison
    print(f"\n" + "⚔️" * 40)
    print("5️⃣  CONFRONTO FINALE E CONCLUSIONI")
    print("⚔️" * 40)
    
    sorted_indices, model_names, best_minDCFs, best_actDCFs = compare_all_svm_models(results)
    
    # Analysis and conclusions
    print(f"\n💡 OSSERVAZIONI E CONCLUSIONI:")
    print("-" * 80)
    
    print(f"\n🔍 SVM Lineare:")
    linear_orig_minDCF = min(results["Linear SVM (Original)"][1])
    linear_cent_minDCF = min(results["Linear SVM (Centered)"][1])
    if abs(linear_orig_minDCF - linear_cent_minDCF) < 0.01:
        print(f"  • Dati centrati vs originali: risultati simili ({linear_orig_minDCF:.4f} vs {linear_cent_minDCF:.4f})")
        print(f"  • La centratura non influenza significativamente le prestazioni")
    else:
        print(f"  • Dati centrati vs originali: differenza significativa")
        if linear_cent_minDCF < linear_orig_minDCF:
            print(f"  • Dati centrati migliori: {linear_cent_minDCF:.4f} vs {linear_orig_minDCF:.4f}")
        else:
            print(f"  • Dati originali migliori: {linear_orig_minDCF:.4f} vs {linear_cent_minDCF:.4f}")
    
    print(f"\n🔮 SVM Polinomiale vs Lineare:")
    poly_minDCF = min(results["Polynomial SVM (d=2, c=1)"][1])
    if poly_minDCF < linear_orig_minDCF:
        print(f"  • SVM Polinomiale migliore: {poly_minDCF:.4f} vs {linear_orig_minDCF:.4f}")
        print(f"  • Le relazioni quadratiche sono importanti per questo dataset")
    else:
        print(f"  • SVM Lineare competitivo: {linear_orig_minDCF:.4f} vs {poly_minDCF:.4f}")
        print(f"  • Le relazioni lineari sono sufficienti")
    
    print(f"\n🌀 SVM RBF:")
    rbf_minDCF = best_rbf['minDCF']
    print(f"  • Miglior configurazione: γ={best_rbf['gamma']:.3f}, C={best_rbf['C']:.1e}")
    if rbf_minDCF < min(linear_orig_minDCF, poly_minDCF):
        print(f"  • RBF offre il miglior risultato: {rbf_minDCF:.4f}")
        print(f"  • I pattern non-lineari complessi sono importanti")
    else:
        print(f"  • RBF non migliora significativamente: {rbf_minDCF:.4f}")
    
    print(f"\n📊 Calibrazione:")
    all_gaps = [best_actDCFs[i] - best_minDCFs[i] for i in range(len(best_actDCFs))]
    avg_gap = np.mean(all_gaps)
    if avg_gap > 0.1:
        print(f"  • Gap medio calibrazione: {avg_gap:.4f} - PESSIMA")
        print(f"  • Tutti i modelli SVM necessitano calibrazione")
    elif avg_gap > 0.05:
        print(f"  • Gap medio calibrazione: {avg_gap:.4f} - MEDIOCRE")
        print(f"  • Calibrazione consigliata per la maggior parte dei modelli")
    else:
        print(f"  • Gap medio calibrazione: {avg_gap:.4f} - BUONA")
    
    # Load and compare with previous results if available
    try:
        with open('logistic_regression_results.pkl', 'rb') as f:
            lr_results = pickle.load(f)
        
        print(f"\n🔄 CONFRONTO CON MODELLI PRECEDENTI:")
        print("-" * 50)
        best_svm_minDCF = min(best_minDCFs)
        best_lr_minDCF = min([min(model[1]) for model in lr_results['models'].values() 
                             if isinstance(model, tuple) and len(model) >= 2])
        
        print(f"📊 Miglior SVM minDCF: {best_svm_minDCF:.4f}")
        print(f"📊 Miglior Logistic Regression minDCF: {best_lr_minDCF:.4f}")
        
        if best_svm_minDCF < best_lr_minDCF:
            print(f"✅ SVM superiore di {best_lr_minDCF - best_svm_minDCF:.4f}")
        else:
            print(f"⚠️  Logistic Regression superiore di {best_svm_minDCF - best_lr_minDCF:.4f}")
            
    except FileNotFoundError:
        print(f"\n⚠️  File risultati Logistic Regression non trovato - skip confronto")
    
    # Save results
    print(f"\n💾 Salvataggio risultati...")
    save_data = {
        'svm_models': results,
        'best_models': {
            'names': [model_names[i] for i in sorted_indices[:3]],
            'minDCFs': [best_minDCFs[i] for i in sorted_indices[:3]],
            'actDCFs': [best_actDCFs[i] for i in sorted_indices[:3]]
        },
        'dataset_info': {
            'n_features': DTR.shape[0],
            'n_train': DTR.shape[1],
            'n_val': DVAL.shape[1],
            'class_distribution': {'class_0': np.sum(LTR==0), 'class_1': np.sum(LTR==1)}
        }
    }
    
    with open('svm_results.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"✅ Risultati salvati in 'svm_results.pkl'")
    print(f"✅ Grafici salvati come file PNG")
    
    print(f"\n" + "=" * 80)
    print("✨ ANALISI SUPPORT VECTOR MACHINE COMPLETATA ✨")
    print("=" * 80)

if __name__ == "__main__":
    main()