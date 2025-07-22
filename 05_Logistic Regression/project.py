import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.special
from pathlib import Path
import pickle

# Import utility functions from existing files
import sys
sys.path.append('/home/bernuz/Scrivania/UNIVERSITA\'/Machine Learning and Pattern Recognition/Labs/13_Support Vector Machine')
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

def expand_features_quadratic(D):
    """Expand features for quadratic logistic regression"""
    n_features = D.shape[0]
    n_samples = D.shape[1]
    
    # Original features + quadratic terms + cross terms
    expanded_features = []
    
    # Original features
    for i in range(n_features):
        expanded_features.append(D[i:i+1, :])
    
    # Quadratic terms
    for i in range(n_features):
        expanded_features.append((D[i:i+1, :] ** 2))
    
    # Cross terms
    for i in range(n_features):
        for j in range(i+1, n_features):
            expanded_features.append(D[i:i+1, :] * D[j:j+1, :])
    
    return np.vstack(expanded_features)

# ---------------------------------------------------------------------
# ---------------------  LOGISTIC REGRESSION  -------------------------
# ---------------------------------------------------------------------

def trainLogRegBinary(DTR, LTR, l):
    """Train binary logistic regression with regularization parameter l"""
    ZTR = LTR * 2.0 - 1.0

    def logreg_obj_with_grad(v):
        w = v[:-1]
        b = v[-1]
        s = np.dot(vcol(w).T, DTR).ravel() + b

        loss = np.logaddexp(0, -ZTR * s)

        G = -ZTR / (1.0 + np.exp(ZTR * s))
        GW = (vrow(G) * DTR).mean(1) + l * w.ravel()
        Gb = G.mean()
        return loss.mean() + l / 2 * np.linalg.norm(w)**2, np.hstack([GW, np.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0=np.zeros(DTR.shape[0]+1))[0]
    return vf[:-1], vf[-1]

def trainWeightedLogRegBinary(DTR, LTR, l, pT):
    """Train weighted binary logistic regression"""
    ZTR = LTR * 2.0 - 1.0
    
    wTrue = pT / (ZTR > 0).sum()
    wFalse = (1 - pT) / (ZTR < 0).sum()

    def logreg_obj_with_grad(v):
        w = v[:-1]
        b = v[-1]
        s = np.dot(vcol(w).T, DTR).ravel() + b

        loss = np.logaddexp(0, -ZTR * s)
        loss[ZTR > 0] *= wTrue
        loss[ZTR < 0] *= wFalse

        G = -ZTR / (1.0 + np.exp(ZTR * s))
        G[ZTR > 0] *= wTrue
        G[ZTR < 0] *= wFalse
        
        GW = (vrow(G) * DTR).sum(1) + l * w.ravel()
        Gb = G.sum()
        return loss.sum() + l / 2 * np.linalg.norm(w)**2, np.hstack([GW, np.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0=np.zeros(DTR.shape[0]+1))[0]
    return vf[:-1], vf[-1]

# ---------------------------------------------------------------------
# ---------------------  EVALUATION FUNCTIONS  ------------------------
# ---------------------------------------------------------------------

def evaluate_model(w, b, DVAL, LVAL, pEmp, target_prior=0.1):
    """Evaluate trained model and compute DCF metrics"""
    sVal = np.dot(w.T, DVAL) + b
    sValLLR = sVal - np.log(pEmp / (1 - pEmp))
    
    minDCF = bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, target_prior, 1.0, 1.0)
    actDCF = bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, target_prior, 1.0, 1.0)
    
    return minDCF, actDCF, sValLLR

def evaluate_weighted_model(w, b, DVAL, LVAL, pT, target_prior=0.1):
    """Evaluate weighted model"""
    sVal = np.dot(w.T, DVAL) + b
    sValLLR = sVal - np.log(pT / (1 - pT))
    
    minDCF = bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, target_prior, 1.0, 1.0)
    actDCF = bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, target_prior, 1.0, 1.0)
    
    return minDCF, actDCF, sValLLR

# ---------------------------------------------------------------------
# ---------------------  ANALYSIS FUNCTIONS  --------------------------
# ---------------------------------------------------------------------

def sanitize_filename(filename):
    """Remove invalid characters from filename"""
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '(', ')']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def analyze_regularization(DTR, LTR, DVAL, LVAL, title_suffix="", reduced_data=False):
    """Analyze effect of regularization parameter lambda"""
    print(f"\n{'='*80}")
    print(f"🔍 ANALISI REGOLARIZZAZIONE - {title_suffix}")
    print(f"{'='*80}")
    
    if reduced_data:
        print(f"📊 Dataset ridotto: {DTR.shape[1]} campioni di training (1 ogni 50)")
    else:
        print(f"📊 Dataset completo: {DTR.shape[1]} campioni di training")
    
    print(f"📊 Validation set: {DVAL.shape[1]} campioni")
    print(f"📊 Features: {DTR.shape[0]}")
    
    # Lambda values
    lambdas = np.logspace(-4, 2, 13)
    pEmp = (LTR == 1).sum() / LTR.size
    
    print(f"\n📈 Prior empirico del training set: {pEmp:.3f}")
    print(f"📈 Valori di λ da testare: {len(lambdas)}")
    print(f"📈 Range λ: [{lambdas[0]:.1e}, {lambdas[-1]:.1e}]")
    
    minDCFs = []
    actDCFs = []
    
    print(f"\n🚀 TRAINING E VALUTAZIONE:")
    print("-" * 80)
    
    for i, lamb in enumerate(lambdas):
        print(f"λ = {lamb:.1e} ({i+1:2d}/{len(lambdas)}) ", end="", flush=True)
        
        # Train model
        w, b = trainLogRegBinary(DTR, LTR, lamb)
        
        # Evaluate
        minDCF, actDCF, _ = evaluate_model(w, b, DVAL, LVAL, pEmp)
        
        minDCFs.append(minDCF)
        actDCFs.append(actDCF)
        
        print(f"→ minDCF: {minDCF:.4f}, actDCF: {actDCF:.4f}")
    
    # Find best lambda
    best_idx = np.argmin(minDCFs)
    best_lambda = lambdas[best_idx]
    
    print(f"\n🏆 RISULTATI MIGLIORI:")
    print("-" * 50)
    print(f"📍 Miglior λ: {best_lambda:.1e}")
    print(f"📍 Miglior minDCF: {minDCFs[best_idx]:.4f}")
    print(f"📍 Corrispondente actDCF: {actDCFs[best_idx]:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.semilogx(lambdas, minDCFs, 'b-o', label='minDCF', linewidth=2, markersize=6)
    plt.semilogx(lambdas, actDCFs, 'r-s', label='actDCF', linewidth=2, markersize=6)
    plt.axvline(best_lambda, color='green', linestyle='--', alpha=0.7, label=f'Best λ={best_lambda:.1e}')
    plt.xlabel('λ (Regularization Parameter)')
    plt.ylabel('DCF')
    plt.title(f'DCF vs Regularization - {title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogx(lambdas, np.array(actDCFs) - np.array(minDCFs), 'g-^', 
                 linewidth=2, markersize=6, label='actDCF - minDCF')
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(best_lambda, color='green', linestyle='--', alpha=0.7)
    plt.xlabel('λ (Regularization Parameter)')
    plt.ylabel('Calibration Gap')
    plt.title(f'Model Calibration - {title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Fix filename for saving
    safe_filename = sanitize_filename(title_suffix)
    plt.savefig(f'logistic_regression_{safe_filename}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return lambdas, minDCFs, actDCFs, best_lambda

def analyze_weighted_logistic_regression(DTR, LTR, DVAL, LVAL):
    """Analyze prior-weighted logistic regression"""
    print(f"\n{'='*80}")
    print(f"🎯 REGRESSIONE LOGISTICA PESATA (Prior-Weighted)")
    print(f"{'='*80}")
    
    target_prior = 0.1
    lambdas = np.logspace(-4, 2, 13)
    
    print(f"📊 Prior target per il training: {target_prior}")
    print(f"📊 Training samples: {DTR.shape[1]}")
    print(f"📊 Validation samples: {DVAL.shape[1]}")
    
    minDCFs_weighted = []
    actDCFs_weighted = []
    
    print(f"\n🚀 TRAINING MODELLI PESATI:")
    print("-" * 80)
    
    for i, lamb in enumerate(lambdas):
        print(f"λ = {lamb:.1e} ({i+1:2d}/{len(lambdas)}) ", end="", flush=True)
        
        # Train weighted model
        w, b = trainWeightedLogRegBinary(DTR, LTR, lamb, target_prior)
        
        # Evaluate
        minDCF, actDCF, _ = evaluate_weighted_model(w, b, DVAL, LVAL, target_prior)
        
        minDCFs_weighted.append(minDCF)
        actDCFs_weighted.append(actDCF)
        
        print(f"→ minDCF: {minDCF:.4f}, actDCF: {actDCF:.4f}")
    
    # Find best lambda
    best_idx = np.argmin(minDCFs_weighted)
    best_lambda = lambdas[best_idx]
    
    print(f"\n🏆 RISULTATI MIGLIORI (Weighted):")
    print("-" * 50)
    print(f"📍 Miglior λ: {best_lambda:.1e}")
    print(f"📍 Miglior minDCF: {minDCFs_weighted[best_idx]:.4f}")
    print(f"📍 Corrispondente actDCF: {actDCFs_weighted[best_idx]:.4f}")
    
    return lambdas, minDCFs_weighted, actDCFs_weighted, best_lambda

def analyze_quadratic_logistic_regression(DTR, LTR, DVAL, LVAL):
    """Analyze quadratic logistic regression"""
    print(f"\n{'='*80}")
    print(f"🔥 REGRESSIONE LOGISTICA QUADRATICA")
    print(f"{'='*80}")
    
    # Expand features
    print(f"🔄 Espansione features...")
    DTR_quad = expand_features_quadratic(DTR)
    DVAL_quad = expand_features_quadratic(DVAL)
    
    print(f"📊 Features originali: {DTR.shape[0]}")
    print(f"📊 Features quadratiche: {DTR_quad.shape[0]}")
    print(f"📊 Fattore di espansione: {DTR_quad.shape[0] / DTR.shape[0]:.1f}x")
    
    pEmp = (LTR == 1).sum() / LTR.size
    lambdas = np.logspace(-4, 2, 13)
    
    minDCFs_quad = []
    actDCFs_quad = []
    
    print(f"\n🚀 TRAINING MODELLI QUADRATICI:")
    print("-" * 80)
    
    for i, lamb in enumerate(lambdas):
        print(f"λ = {lamb:.1e} ({i+1:2d}/{len(lambdas)}) ", end="", flush=True)
        
        # Train quadratic model
        w, b = trainLogRegBinary(DTR_quad, LTR, lamb)
        
        # Evaluate
        minDCF, actDCF, _ = evaluate_model(w, b, DVAL_quad, LVAL, pEmp)
        
        minDCFs_quad.append(minDCF)
        actDCFs_quad.append(actDCF)
        
        print(f"→ minDCF: {minDCF:.4f}, actDCF: {actDCF:.4f}")
    
    # Find best lambda
    best_idx = np.argmin(minDCFs_quad)
    best_lambda = lambdas[best_idx]
    
    print(f"\n🏆 RISULTATI MIGLIORI (Quadratic):")
    print("-" * 50)
    print(f"📍 Miglior λ: {best_lambda:.1e}")
    print(f"📍 Miglior minDCF: {minDCFs_quad[best_idx]:.4f}")
    print(f"📍 Corrispondente actDCF: {actDCFs_quad[best_idx]:.4f}")
    
    # Plot comparison with linear
    _, minDCFs_linear, actDCFs_linear, _ = analyze_regularization(DTR, LTR, DVAL, LVAL, "Linear (Comparison)", reduced_data=False)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.semilogx(lambdas, minDCFs_linear, 'b-o', label='Linear minDCF', linewidth=2)
    plt.semilogx(lambdas, minDCFs_quad, 'r-s', label='Quadratic minDCF', linewidth=2)
    plt.xlabel('λ (Regularization Parameter)')
    plt.ylabel('minDCF')
    plt.title('Linear vs Quadratic: minDCF')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogx(lambdas, actDCFs_linear, 'b-o', label='Linear actDCF', linewidth=2)
    plt.semilogx(lambdas, actDCFs_quad, 'r-s', label='Quadratic actDCF', linewidth=2)
    plt.xlabel('λ (Regularization Parameter)')
    plt.ylabel('actDCF')
    plt.title('Linear vs Quadratic: actDCF')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('logistic_regression_linear_vs_quadratic.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return lambdas, minDCFs_quad, actDCFs_quad, best_lambda

def compare_all_models(results_dict):
    """Compare all trained models"""
    print(f"\n{'='*80}")
    print(f"🏆 CONFRONTO FINALE - TUTTI I MODELLI")
    print(f"{'='*80}")
    
    model_names = []
    best_minDCFs = []
    best_actDCFs = []
    best_lambdas = []
    
    for model_name, (lambdas, minDCFs, actDCFs, best_lambda) in results_dict.items():
        best_idx = np.argmin(minDCFs)
        model_names.append(model_name)
        best_minDCFs.append(minDCFs[best_idx])
        best_actDCFs.append(actDCFs[best_idx])
        best_lambdas.append(best_lambda)
    
    # Sort by minDCF
    sorted_indices = np.argsort(best_minDCFs)
    
    print(f"\n📊 RANKING PER minDCF (πT = 0.1):")
    print("-" * 80)
    print(f"{'Rank':<4} {'Model':<25} {'minDCF':<8} {'actDCF':<8} {'Best λ':<12} {'Calibration':<12}")
    print("-" * 80)
    
    for i, idx in enumerate(sorted_indices):
        calibration_gap = best_actDCFs[idx] - best_minDCFs[idx]
        calibration_status = "Good" if calibration_gap < 0.01 else "Poor" if calibration_gap > 0.05 else "Fair"
        
        print(f"{i+1:<4} {model_names[idx]:<25} {best_minDCFs[idx]:<8.4f} {best_actDCFs[idx]:<8.4f} "
              f"{best_lambdas[idx]:<12.1e} {calibration_status:<12}")
    
    # Visual comparison
    plt.figure(figsize=(15, 8))
    
    x_pos = np.arange(len(model_names))
    
    plt.subplot(1, 2, 1)
    bars1 = plt.bar(x_pos - 0.2, best_minDCFs, 0.4, label='minDCF', alpha=0.8, color='skyblue')
    bars2 = plt.bar(x_pos + 0.2, best_actDCFs, 0.4, label='actDCF', alpha=0.8, color='lightcoral')
    
    plt.xlabel('Models')
    plt.ylabel('DCF')
    plt.title('Model Comparison: DCF Values')
    plt.xticks(x_pos, model_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
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
    colors = ['green' if gap < 0.01 else 'orange' if gap < 0.05 else 'red' for gap in calibration_gaps]
    
    bars = plt.bar(x_pos, calibration_gaps, color=colors, alpha=0.7)
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(0.01, color='green', linestyle='--', alpha=0.5, label='Good calibration (<0.01)')
    plt.axhline(0.05, color='orange', linestyle='--', alpha=0.5, label='Fair calibration (<0.05)')
    
    plt.xlabel('Models')
    plt.ylabel('Calibration Gap (actDCF - minDCF)')
    plt.title('Model Calibration Quality')
    plt.xticks(x_pos, model_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('all_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return sorted_indices, model_names, best_minDCFs, best_actDCFs

# ---------------------------------------------------------------------
# ---------------------  MAIN EXECUTION  ------------------------------
# ---------------------------------------------------------------------

def main():
    print("=" * 80)
    print("    ANALISI REGRESSIONE LOGISTICA - PROGETTO ML&PR")
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
    
    # Store results for comparison
    results = {}
    
    # 1. Standard Logistic Regression - Full Dataset
    print(f"\n" + "🔸" * 40)
    print("1️⃣  REGRESSIONE LOGISTICA STANDARD - DATASET COMPLETO")
    print("🔸" * 40)
    
    lambdas, minDCFs, actDCFs, best_lambda = analyze_regularization(
        DTR, LTR, DVAL, LVAL, "Full Dataset"
    )
    results["Logistic Regression (Full)"] = (lambdas, minDCFs, actDCFs, best_lambda)
    
    # 2. Standard Logistic Regression - Reduced Dataset
    print(f"\n" + "🔸" * 40)
    print("2️⃣  REGRESSIONE LOGISTICA STANDARD - DATASET RIDOTTO (1/50)")
    print("🔸" * 40)
    
    DTR_reduced = DTR[:, ::50]
    LTR_reduced = LTR[::50]
    
    lambdas_red, minDCFs_red, actDCFs_red, best_lambda_red = analyze_regularization(
        DTR_reduced, LTR_reduced, DVAL, LVAL, "Reduced Dataset (1/50)", reduced_data=True
    )
    results["Logistic Regression (Reduced)"] = (lambdas_red, minDCFs_red, actDCFs_red, best_lambda_red)
    
    # 3. Prior-Weighted Logistic Regression
    print(f"\n" + "🔸" * 40)
    print("3️⃣  REGRESSIONE LOGISTICA PESATA (Prior-Weighted)")
    print("🔸" * 40)
    
    lambdas_w, minDCFs_w, actDCFs_w, best_lambda_w = analyze_weighted_logistic_regression(
        DTR, LTR, DVAL, LVAL
    )
    results["Weighted Logistic Regression"] = (lambdas_w, minDCFs_w, actDCFs_w, best_lambda_w)
    
    # 4. Quadratic Logistic Regression
    print(f"\n" + "🔸" * 40)
    print("4️⃣  REGRESSIONE LOGISTICA QUADRATICA")
    print("🔸" * 40)
    
    lambdas_q, minDCFs_q, actDCFs_q, best_lambda_q = analyze_quadratic_logistic_regression(
        DTR, LTR, DVAL, LVAL
    )
    results["Quadratic Logistic Regression"] = (lambdas_q, minDCFs_q, actDCFs_q, best_lambda_q)
    
    # 5. Final Comparison
    print(f"\n" + "🔸" * 40)
    print("5️⃣  CONFRONTO FINALE E CONCLUSIONI")
    print("🔸" * 40)
    
    sorted_indices, model_names, best_minDCFs, best_actDCFs = compare_all_models(results)
    
    # Analysis and conclusions
    print(f"\n💡 OSSERVAZIONI E CONCLUSIONI:")
    print("-" * 80)
    
    print(f"\n🔍 Effetto della Regolarizzazione:")
    if results["Logistic Regression (Full)"][3] > 1e-2:
        print(f"  • Con dataset completo, la regolarizzazione ha effetto moderato")
        print(f"  • Molti campioni → minor rischio di overfitting")
    else:
        print(f"  • Con dataset completo, la regolarizzazione è importante")
    
    if results["Logistic Regression (Reduced)"][3] != results["Logistic Regression (Full)"][3]:
        print(f"  • Con dataset ridotto, l'ottimo λ cambia significativamente")
        print(f"  • Pochi campioni → maggior rischio di overfitting → più regolarizzazione")
    
    print(f"\n🎯 Modelli Pesati vs Standard:")
    standard_minDCF = min(results["Logistic Regression (Full)"][1])
    weighted_minDCF = min(results["Weighted Logistic Regression"][1])
    if weighted_minDCF < standard_minDCF:
        print(f"  • Il modello pesato migliora le prestazioni ({weighted_minDCF:.4f} vs {standard_minDCF:.4f})")
        print(f"  • Utile quando il prior target è diverso da quello empirico")
    else:
        print(f"  • Il modello pesato non migliora significativamente le prestazioni")
    
    print(f"\n🔥 Modello Quadratico:")
    linear_minDCF = min(results["Logistic Regression (Full)"][1])
    quadratic_minDCF = min(results["Quadratic Logistic Regression"][1])
    if quadratic_minDCF < linear_minDCF:
        print(f"  • Il modello quadratico migliora le prestazioni ({quadratic_minDCF:.4f} vs {linear_minDCF:.4f})")
        print(f"  • Le interazioni tra features sono importanti per questo dataset")
        print(f"  • La regolarizzazione diventa cruciale con più features")
    else:
        print(f"  • Il modello quadratico non migliora le prestazioni")
        print(f"  • Le relazioni lineari sono sufficienti per questo dataset")
    
    # Save results for future use
    print(f"\n💾 Salvataggio risultati per laboratori futuri...")
    save_data = {
        'models': results,
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
    
    with open('logistic_regression_results.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"✅ Risultati salvati in 'logistic_regression_results.pkl'")
    print(f"✅ Grafici salvati come file PNG")
    
    print(f"\n" + "=" * 80)
    print("✨ ANALISI REGRESSIONE LOGISTICA COMPLETATA ✨")
    print("=" * 80)

if __name__ == "__main__":
    main()