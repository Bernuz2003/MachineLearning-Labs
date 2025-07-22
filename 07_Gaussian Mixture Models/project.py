import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
from pathlib import Path

sys.path.append('../05_Logistic Regression')
sys.path.append('../06_Support Vector Machine')

# Import GMM functions
from gmm import *
import bayesRisk

try:
    # Import logistic regression functions
    sys.path.insert(0, '../05_Logistic Regression')
    import project as lr_project
    trainWeightedLogRegBinary = lr_project.trainWeightedLogRegBinary
    trainLogRegBinary = lr_project.trainLogRegBinary
    print("✅ Logistic Regression module imported")
except ImportError as e:
    print(f"⚠️  Could not import Logistic Regression: {e}")
    trainWeightedLogRegBinary = None
    trainLogRegBinary = None

try:
    # Import SVM functions from sol.py
    sys.path.insert(0, '../06_Support Vector Machine')
    import sol as svm_module
    train_dual_SVM_linear = svm_module.train_dual_SVM_linear
    train_dual_SVM_kernel = svm_module.train_dual_SVM_kernel
    polyKernel = svm_module.polyKernel
    rbfKernel = svm_module.rbfKernel
    print("✅ SVM module imported")
except ImportError as e:
    print(f"⚠️  Could not import SVM: {e}")
    try:
        # Fallback: try importing from project.py in SVM lab
        import project as svm_project
        train_dual_SVM_linear = svm_project.train_dual_SVM_linear
        train_dual_SVM_kernel = svm_project.train_dual_SVM_kernel
        polyKernel = svm_project.polyKernel
        rbfKernel = svm_project.rbfKernel
        print("✅ SVM module imported from project.py")
    except ImportError as e2:
        print(f"⚠️  Could not import SVM from project.py either: {e2}")
        train_dual_SVM_linear = None
        train_dual_SVM_kernel = None
        polyKernel = None
        rbfKernel = None

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
            fields = line.split(',')
            sample = np.array([float(x) for x in fields[0:6]]).reshape(6, 1)
            label = int(fields[6])
            samples.append(sample)
            labels.append(label)
    
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

def sanitize_filename(filename):
    """Remove invalid characters from filename"""
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '(', ')', ' ']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

# ---------------------------------------------------------------------
# ---------------------  GMM ANALYSIS FUNCTIONS  ----------------------
# ---------------------------------------------------------------------

def analyze_gmm_components(DTR, LTR, DVAL, LVAL, cov_type='full', max_components=32, target_prior=0.1):
    """Analyze GMM with different number of components for each class"""
    print(f"\n{'='*80}")
    print(f"📊 ANALISI GMM - {cov_type.upper()} COVARIANCE")
    print(f"{'='*80}")
    
    print(f"📊 Covariance Type: {cov_type}")
    print(f"📊 Max Components: {max_components}")
    print(f"📊 Training samples: {DTR.shape[1]}")
    print(f"📊 Validation samples: {DVAL.shape[1]}")
    print(f"📊 Target Prior: {target_prior}")
    
    # Test different number of components
    component_values = [2**i for i in range(int(np.log2(max_components)) + 1)]
    
    print(f"\n📈 Component values to test: {component_values}")
    
    results = {}
    best_result = {'minDCF': float('inf'), 'components': None, 'actDCF': None}
    
    print(f"\n🚀 TRAINING E VALUTAZIONE:")
    print("=" * 100)
    
    for num_components in component_values:
        print(f"\n🔄 Testing {num_components} components...")
        
        # Train GMMs for both classes
        print(f"  Training GMM for Class 0 ({(LTR==0).sum()} samples)...")
        gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], num_components, 
                                covType=cov_type, verbose=False, psiEig=0.01)
        
        print(f"  Training GMM for Class 1 ({(LTR==1).sum()} samples)...")
        gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], num_components, 
                                covType=cov_type, verbose=False, psiEig=0.01)
        
        # Compute log-likelihood ratios for validation
        ll0 = logpdf_GMM(DVAL, gmm0)
        ll1 = logpdf_GMM(DVAL, gmm1)
        llr_scores = ll1 - ll0
        
        # Evaluate performance
        minDCF = bayesRisk.compute_minDCF_binary_fast(llr_scores, LVAL, target_prior, 1.0, 1.0)
        actDCF = bayesRisk.compute_actDCF_binary_fast(llr_scores, LVAL, target_prior, 1.0, 1.0)
        
        results[num_components] = {
            'minDCF': minDCF,
            'actDCF': actDCF,
            'llr_scores': llr_scores,
            'gmm0': gmm0,
            'gmm1': gmm1
        }
        
        # Track best result
        if minDCF < best_result['minDCF']:
            best_result.update({
                'minDCF': minDCF,
                'actDCF': actDCF,
                'components': num_components,
                'scores': llr_scores,
                'model_info': f'GMM {cov_type} ({num_components} comp)'
            })
        
        print(f"  → Components: {num_components:2d} | minDCF: {minDCF:.4f} | actDCF: {actDCF:.4f} | Gap: {actDCF - minDCF:.4f}")
    
    print(f"\n🏆 MIGLIOR RISULTATO ({cov_type.upper()}):")
    print("-" * 50)
    print(f"📍 Migliori Components: {best_result['components']}")
    print(f"📍 Miglior minDCF: {best_result['minDCF']:.4f}")
    print(f"📍 Corrispondente actDCF: {best_result['actDCF']:.4f}")
    print(f"📍 Gap calibrazione: {best_result['actDCF'] - best_result['minDCF']:.4f}")
    
    return results, best_result

# ---------------------------------------------------------------------
# ---------------------  OTHER CLASSIFIERS  ---------------------------
# ---------------------------------------------------------------------

def analyze_logistic_regression(DTR, LTR, DVAL, LVAL, target_prior=0.1):
    """Analyze logistic regression models from previous lab"""
    print(f"\n{'='*80}")
    print(f"📊 ANALISI LOGISTIC REGRESSION")
    print(f"{'='*80}")
    
    if trainWeightedLogRegBinary is None or trainLogRegBinary is None:
        print("⚠️  Logistic Regression functions not available")
        return None
    
    # Test different lambda values (from previous lab analysis)
    lambdas = np.logspace(-4, 2, 13)
    
    best_result = {'minDCF': float('inf'), 'lambda': None, 'actDCF': None}
    
    print(f"\n🚀 TESTING DIFFERENT CONFIGURATIONS:")
    print("-" * 80)
    
    # 1. Standard Logistic Regression
    print("\n🔍 Standard Logistic Regression:")
    pEmp = (LTR == 1).sum() / LTR.size
    
    for lamb in lambdas:
        try:
            w, b = trainLogRegBinary(DTR, LTR, lamb)
            scores = (w.T @ DVAL + b - np.log(pEmp / (1 - pEmp))).ravel()
            
            minDCF = bayesRisk.compute_minDCF_binary_fast(scores, LVAL, target_prior, 1.0, 1.0)
            actDCF = bayesRisk.compute_actDCF_binary_fast(scores, LVAL, target_prior, 1.0, 1.0)
            
            if minDCF < best_result['minDCF']:
                best_result.update({
                    'minDCF': minDCF,
                    'actDCF': actDCF,
                    'lambda': lamb,
                    'scores': scores,
                    'type': 'Standard',
                    'model_info': f'Standard LogReg (λ={lamb:.1e})'
                })
            
        except Exception as e:
            print(f"  Error with λ={lamb:.1e}: {e}")
            continue
    
    # 2. Weighted Logistic Regression
    print("\n🔍 Weighted Logistic Regression:")
    
    for lamb in lambdas:
        try:
            w, b = trainWeightedLogRegBinary(DTR, LTR, lamb, target_prior)
            scores = (w.T @ DVAL + b - np.log(target_prior / (1 - target_prior))).ravel()
            
            minDCF = bayesRisk.compute_minDCF_binary_fast(scores, LVAL, target_prior, 1.0, 1.0)
            actDCF = bayesRisk.compute_actDCF_binary_fast(scores, LVAL, target_prior, 1.0, 1.0)
            
            if minDCF < best_result['minDCF']:
                best_result.update({
                    'minDCF': minDCF,
                    'actDCF': actDCF,
                    'lambda': lamb,
                    'scores': scores,
                    'type': 'Weighted',
                    'model_info': f'Weighted LogReg (λ={lamb:.1e}, pT={target_prior})'
                })
            
        except Exception as e:
            print(f"  Error with λ={lamb:.1e}: {e}")
            continue
    
    if best_result['minDCF'] == float('inf'):
        print("⚠️  No valid Logistic Regression model found")
        return None
    
    print(f"\n🏆 BEST LOGISTIC REGRESSION:")
    print("-" * 50)
    print(f"📍 Type: {best_result['type']}")
    print(f"📍 Best λ: {best_result['lambda']:.1e}")
    print(f"📍 Best minDCF: {best_result['minDCF']:.4f}")
    print(f"📍 Corresponding actDCF: {best_result['actDCF']:.4f}")
    
    return best_result

def analyze_svm(DTR, LTR, DVAL, LVAL, target_prior=0.1):
    """Analyze SVM models from previous lab"""
    print(f"\n{'='*80}")
    print(f"📊 ANALISI SUPPORT VECTOR MACHINE")
    print(f"{'='*80}")
    
    if train_dual_SVM_linear is None:
        print("⚠️  SVM functions not available")
        return None
    
    # Test different C values and kernels (from previous lab analysis)
    C_values = np.logspace(-5, 0, 11)
    
    best_result = {'minDCF': float('inf'), 'C': None, 'actDCF': None}
    
    print(f"\n🚀 TESTING DIFFERENT CONFIGURATIONS:")
    print("-" * 80)
    
    # 1. Linear SVM
    print("\n🔍 Linear SVM:")
    
    for C in C_values:
        try:
            w, b = train_dual_SVM_linear(DTR, LTR, C, K=1.0)
            scores = (vrow(w) @ DVAL + b).ravel()
            
            minDCF = bayesRisk.compute_minDCF_binary_fast(scores, LVAL, target_prior, 1.0, 1.0)
            actDCF = bayesRisk.compute_actDCF_binary_fast(scores, LVAL, target_prior, 1.0, 1.0)
            
            if minDCF < best_result['minDCF']:
                best_result.update({
                    'minDCF': minDCF,
                    'actDCF': actDCF,
                    'C': C,
                    'scores': scores,
                    'type': 'Linear',
                    'model_info': f'Linear SVM (C={C:.1e})'
                })
                
            print(f"  C={C:.1e}: minDCF={minDCF:.4f}, actDCF={actDCF:.4f}")
            
        except Exception as e:
            print(f"  Error with C={C:.1e}: {e}")
            continue
    
    # 2. RBF SVM (only if kernel functions available)
    if rbfKernel is not None and train_dual_SVM_kernel is not None:
        print("\n🔍 RBF SVM:")
        
        gammas = [0.1, 1.0, 10.0]
        
        for C in [0.1, 1.0]:  # Reduce C values for kernel SVM
            for gamma in gammas:
                try:
                    kernelFunc = rbfKernel(gamma)
                    fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps=1.0)
                    scores = fScore(DVAL)
                    
                    minDCF = bayesRisk.compute_minDCF_binary_fast(scores, LVAL, target_prior, 1.0, 1.0)
                    actDCF = bayesRisk.compute_actDCF_binary_fast(scores, LVAL, target_prior, 1.0, 1.0)
                    
                    if minDCF < best_result['minDCF']:
                        best_result.update({
                            'minDCF': minDCF,
                            'actDCF': actDCF,
                            'C': C,
                            'gamma': gamma,
                            'scores': scores,
                            'type': 'RBF',
                            'model_info': f'RBF SVM (C={C:.1e}, γ={gamma})'
                        })
                    
                    print(f"  C={C:.1e}, γ={gamma}: minDCF={minDCF:.4f}, actDCF={actDCF:.4f}")
                    
                except Exception as e:
                    print(f"  Error with C={C:.1e}, γ={gamma}: {e}")
                    continue
    
    # 3. Polynomial SVM (only if kernel functions available)
    if polyKernel is not None and train_dual_SVM_kernel is not None:
        print("\n🔍 Polynomial SVM:")
        
        degrees = [2]
        cs = [0, 1]
        
        for C in [0.1, 1.0]:
            for degree in degrees:
                for c in cs:
                    try:
                        kernelFunc = polyKernel(degree, c)
                        fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps=1.0)
                        scores = fScore(DVAL)
                        
                        minDCF = bayesRisk.compute_minDCF_binary_fast(scores, LVAL, target_prior, 1.0, 1.0)
                        actDCF = bayesRisk.compute_actDCF_binary_fast(scores, LVAL, target_prior, 1.0, 1.0)
                        
                        if minDCF < best_result['minDCF']:
                            best_result.update({
                                'minDCF': minDCF,
                                'actDCF': actDCF,
                                'C': C,
                                'degree': degree,
                                'c': c,
                                'scores': scores,
                                'type': 'Polynomial',
                                'model_info': f'Poly SVM (C={C:.1e}, d={degree}, c={c})'
                            })
                        
                        print(f"  C={C:.1e}, d={degree}, c={c}: minDCF={minDCF:.4f}, actDCF={actDCF:.4f}")
                        
                    except Exception as e:
                        print(f"  Error with C={C:.1e}, d={degree}, c={c}: {e}")
                        continue
    
    if best_result['minDCF'] == float('inf'):
        print("⚠️  No valid SVM model found")
        return None
    
    print(f"\n🏆 BEST SVM:")
    print("-" * 50)
    print(f"📍 Type: {best_result['type']}")
    print(f"📍 Best minDCF: {best_result['minDCF']:.4f}")
    print(f"📍 Corresponding actDCF: {best_result['actDCF']:.4f}")
    
    return best_result

# ---------------------------------------------------------------------
# ---------------------  PLOTTING FUNCTIONS  --------------------------
# ---------------------------------------------------------------------

def plot_gmm_results(results_full, results_diag, best_full, best_diag):
    """Plot GMM results comparison"""
    
    plt.figure(figsize=(15, 10))
    
    # Extract data for plotting
    components_full = list(results_full.keys())
    minDCFs_full = [results_full[c]['minDCF'] for c in components_full]
    actDCFs_full = [results_full[c]['actDCF'] for c in components_full]
    
    components_diag = list(results_diag.keys())
    minDCFs_diag = [results_diag[c]['minDCF'] for c in components_diag]
    actDCFs_diag = [results_diag[c]['actDCF'] for c in components_diag]
    
    # Plot 1: minDCF comparison
    plt.subplot(2, 2, 1)
    plt.semilogx(components_full, minDCFs_full, 'b-o', label='Full Covariance', 
                linewidth=2, markersize=6)
    plt.semilogx(components_diag, minDCFs_diag, 'r-s', label='Diagonal Covariance', 
                linewidth=2, markersize=6)
    plt.axvline(best_full['components'], color='blue', linestyle='--', alpha=0.7, 
               label=f'Best Full ({best_full["components"]})')
    plt.axvline(best_diag['components'], color='red', linestyle='--', alpha=0.7, 
               label=f'Best Diag ({best_diag["components"]})')
    plt.xlabel('Number of Components')
    plt.ylabel('minDCF')
    plt.title('GMM: minDCF vs Number of Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: actDCF comparison
    plt.subplot(2, 2, 2)
    plt.semilogx(components_full, actDCFs_full, 'b-o', label='Full Covariance', 
                linewidth=2, markersize=6)
    plt.semilogx(components_diag, actDCFs_diag, 'r-s', label='Diagonal Covariance', 
                linewidth=2, markersize=6)
    plt.axvline(best_full['components'], color='blue', linestyle='--', alpha=0.7)
    plt.axvline(best_diag['components'], color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Number of Components')
    plt.ylabel('actDCF')
    plt.title('GMM: actDCF vs Number of Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Calibration gap
    plt.subplot(2, 2, 3)
    gap_full = [actDCFs_full[i] - minDCFs_full[i] for i in range(len(components_full))]
    gap_diag = [actDCFs_diag[i] - minDCFs_diag[i] for i in range(len(components_diag))]
    
    plt.semilogx(components_full, gap_full, 'b-o', label='Full Covariance', 
                linewidth=2, markersize=6)
    plt.semilogx(components_diag, gap_diag, 'r-s', label='Diagonal Covariance', 
                linewidth=2, markersize=6)
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(0.05, color='orange', linestyle='--', alpha=0.5, label='Fair threshold')
    plt.axhline(0.1, color='red', linestyle='--', alpha=0.5, label='Poor threshold')
    plt.xlabel('Number of Components')
    plt.ylabel('Calibration Gap (actDCF - minDCF)')
    plt.title('GMM: Calibration Quality')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Direct comparison
    plt.subplot(2, 2, 4)
    x_pos = np.arange(len(components_full))
    width = 0.35
    
    plt.bar(x_pos - width/2, minDCFs_full, width, label='Full - minDCF', alpha=0.8, color='blue')
    plt.bar(x_pos + width/2, minDCFs_diag, width, label='Diag - minDCF', alpha=0.8, color='red')
    
    plt.xlabel('Number of Components')
    plt.ylabel('minDCF')
    plt.title('GMM: Direct minDCF Comparison')
    plt.xticks(x_pos, [str(c) for c in components_full])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('gmm_analysis_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_bayes_error_plot(models_dict, DVAL, LVAL, title="Bayes Error Plot"):
    """Create Bayes error plot for model comparison"""
    
    plt.figure(figsize=(12, 8))
    
    # Range of effective priors (log-odds from -4 to +4)
    log_odds_range = np.linspace(-4, 4, 41)
    effective_priors = 1 / (1 + np.exp(-log_odds_range))
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for idx, (model_name, scores) in enumerate(models_dict.items()):
        minDCFs = []
        actDCFs = []
        
        for prior in effective_priors:
            minDCF = bayesRisk.compute_minDCF_binary_fast(scores, LVAL, prior, 1.0, 1.0)
            actDCF = bayesRisk.compute_actDCF_binary_fast(scores, LVAL, prior, 1.0, 1.0)
            minDCFs.append(minDCF)
            actDCFs.append(actDCF)
        
        plt.plot(log_odds_range, minDCFs, color=colors[idx % len(colors)], linestyle='-', 
                linewidth=2, label=f'{model_name} - minDCF')
        plt.plot(log_odds_range, actDCFs, color=colors[idx % len(colors)], linestyle='--', 
                linewidth=2, alpha=0.7, label=f'{model_name} - actDCF')
    
    plt.xlabel('log-odds (log(π/(1-π)))')
    plt.ylabel('DCF')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axvline(np.log(0.1/0.9), color='black', linestyle=':', alpha=0.7, 
               label='Target Application (π=0.1)')
    
    plt.tight_layout()
    plt.savefig(sanitize_filename(f'{title.lower()}_bayes_error_plot.png'), 
               dpi=300, bbox_inches='tight')
    plt.show()

def compare_all_classifiers_complete(best_gmm_full, best_gmm_diag, best_lr, best_svm):
    """Complete comparison including all classifiers from the course"""
    
    print(f"\n{'='*80}")
    print(f"🏆 CONFRONTO COMPLETO - TUTTI I CLASSIFICATORI DEL CORSO")
    print(f"{'='*80}")
    
    # Combine all models
    all_models = {
        'GMM Full': best_gmm_full,
        'GMM Diagonal': best_gmm_diag
    }
    
    if best_lr:
        all_models['Logistic Regression'] = best_lr
    if best_svm:
        all_models['SVM'] = best_svm
    
    print(f"\n📊 RISULTATI COMPARATIVI:")
    print("-" * 90)
    print(f"{'Classifier':<20} {'minDCF':<8} {'actDCF':<8} {'Gap':<8} {'Calibration':<12} {'Model Info'}")
    print("-" * 90)
    
    best_overall = {'minDCF': float('inf'), 'name': None, 'model': None}
    
    for name, model in all_models.items():
        if model is None:
            continue
            
        minDCF = model['minDCF']
        actDCF = model['actDCF']
        gap = actDCF - minDCF
        
        # Calibration quality
        if gap < 0.05:
            calibration = "Good"
        elif gap < 0.1:
            calibration = "Fair"
        else:
            calibration = "Poor"
        
        model_info = model.get('model_info', 'N/A')
        
        print(f"{name:<20} {minDCF:<8.4f} {actDCF:<8.4f} {gap:<8.4f} {calibration:<12} {model_info}")
        
        if minDCF < best_overall['minDCF']:
            best_overall = {'minDCF': minDCF, 'actDCF': actDCF, 'name': name, 'model': model}
    
    print("-" * 90)
    
    # Create Bayes error plot for all models
    print(f"\n📈 Generazione Bayes Error Plot per tutti i modelli...")
    
    available_models = {name: model['scores'] for name, model in all_models.items() if model is not None}
    if available_models:
        D, L = load_project_data()
        (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
        create_bayes_error_plot(available_models, DVAL, LVAL, 
                               "All Classifiers - Bayes Error Plot")
    
    print(f"\n🥇 MIGLIOR CLASSIFICATORE GLOBALE:")
    print(f"📍 Metodo: {best_overall['name']}")
    print(f"📍 minDCF: {best_overall['minDCF']:.4f}")
    print(f"📍 actDCF: {best_overall['actDCF']:.4f}")
    
    # Save results for calibration lab
    save_results_for_calibration_lab(all_models, best_overall)
    
    return best_overall, all_models

def save_results_for_calibration_lab(all_models, best_overall):
    """Save results for use in calibration lab"""
    
    calibration_data = {
        'best_models': all_models,
        'best_overall': best_overall,
        'dataset_info': {
            'target_prior': 0.1,
            'evaluation_metric': 'minDCF'
        }
    }
    
    with open('best_models_for_calibration.pkl', 'wb') as f:
        pickle.dump(calibration_data, f)
    
    print(f"\n💾 Risultati salvati per il lab di calibrazione: 'best_models_for_calibration.pkl'")

# ---------------------------------------------------------------------
# ---------------------  MAIN EXECUTION  ------------------------------
# ---------------------------------------------------------------------

def main():
    print(f"{'='*80}")
    print(f"    ANALISI GAUSSIAN MIXTURE MODELS - PROGETTO ML&PR")
    print(f"{'='*80}")
    
    # Load data
    print(f"\n🔄 Caricamento dataset...")
    D, L = load_project_data()
    
    print(f"📊 Dataset caricato: {D.shape[1]} campioni, {D.shape[0]} features")
    print(f"📊 Distribuzione classi: Classe 0: {(L==0).sum()}, Classe 1: {(L==1).sum()}")
    
    # Split data
    print(f"\n📊 Split dataset (2/3 training, 1/3 validation)...")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    print(f"📊 Training: {DTR.shape[1]} campioni")
    print(f"📊 Validation: {DVAL.shape[1]} campioni")
    
    # Analysis parameters
    target_prior = 0.1
    max_components = 32
    
    print(f"\n🎯 Parametri analisi:")
    print(f"🎯 Target Prior: {target_prior}")
    print(f"🎯 Max Components: {max_components}")
    
    # 1. Analyze Full Covariance GMM
    print(f"\n{'🔸'*40}")
    print(f"1️⃣  GMM FULL COVARIANCE")
    print(f"{'🔸'*40}")
    
    results_full, best_full = analyze_gmm_components(
        DTR, LTR, DVAL, LVAL, 
        cov_type='full', 
        max_components=max_components, 
        target_prior=target_prior
    )
    
    # 2. Analyze Diagonal Covariance GMM
    print(f"\n{'🔸'*40}")
    print(f"2️⃣  GMM DIAGONAL COVARIANCE")
    print(f"{'🔸'*40}")
    
    results_diag, best_diag = analyze_gmm_components(
        DTR, LTR, DVAL, LVAL, 
        cov_type='diagonal', 
        max_components=max_components, 
        target_prior=target_prior
    )
    
    # 3. Analyze Logistic Regression (from previous lab)
    print(f"\n{'🔸'*40}")
    print(f"3️⃣  LOGISTIC REGRESSION (da lab precedente)")
    print(f"{'🔸'*40}")
    
    best_lr = analyze_logistic_regression(DTR, LTR, DVAL, LVAL, target_prior)
    
    # 4. Analyze SVM (from previous lab)
    print(f"\n{'🔸'*40}")
    print(f"4️⃣  SUPPORT VECTOR MACHINE (da lab precedente)")
    print(f"{'🔸'*40}")
    
    best_svm = analyze_svm(DTR, LTR, DVAL, LVAL, target_prior)
    
    # 5. Plot GMM comparison
    print(f"\n📊 Generazione grafici comparativi GMM...")
    plot_gmm_results(results_full, results_diag, best_full, best_diag)
    
    # 6. Complete comparison with all classifiers
    print(f"\n{'🏆'*40}")
    print(f"5️⃣  CONFRONTO COMPLETO CON TUTTI I CLASSIFICATORI")
    print(f"{'🏆'*40}")
    
    best_overall, all_models = compare_all_classifiers_complete(best_full, best_diag, best_lr, best_svm)
    
    # 7. Analysis and conclusions
    print(f"\n{'💡'*40}")
    print(f"6️⃣  ANALISI E CONCLUSIONI")
    print(f"{'💡'*40}")
    
    print(f"\n🔍 ANALISI COMPARATIVA:")
    print("-" * 50)
    
    # GMM Analysis
    if best_full['minDCF'] < best_diag['minDCF']:
        print(f"✅ GMM Full Covariance è migliore di Diagonal:")
        print(f"   - Full: {best_full['minDCF']:.4f} con {best_full['components']} componenti")
        print(f"   - Diagonal: {best_diag['minDCF']:.4f} con {best_diag['components']} componenti")
    else:
        print(f"✅ GMM Diagonal Covariance è migliore di Full:")
        print(f"   - Diagonal: {best_diag['minDCF']:.4f} con {best_diag['components']} componenti")
        print(f"   - Full: {best_full['minDCF']:.4f} con {best_full['components']} componenti")
    
    # Overall comparison
    print(f"\n🏆 RANKING FINALE:")
    print("-" * 30)
    sorted_models = sorted([(name, model) for name, model in all_models.items() if model is not None], 
                          key=lambda x: x[1]['minDCF'])
    
    for i, (name, model) in enumerate(sorted_models):
        print(f"  {i+1}. {name}: minDCF = {model['minDCF']:.4f}")
    
    print(f"\n📊 OSSERVAZIONI:")
    print("-" * 20)
    
    if best_overall['name'].startswith('GMM'):
        print(f"• GMM è il metodo migliore per questo dataset")
        print(f"• Le caratteristiche del dataset si adattano bene al modello GMM")
    elif best_overall['name'] == 'Logistic Regression':
        print(f"• Logistic Regression è il metodo migliore")
        print(f"• Il dataset potrebbe avere relazioni principalmente lineari")
    elif best_overall['name'] == 'SVM':
        print(f"• SVM è il metodo migliore")
        print(f"• Le capacità di margin maximization dell'SVM sono efficaci")
    
    # Calibration analysis
    well_calibrated = [name for name, model in all_models.items() 
                      if model and (model['actDCF'] - model['minDCF']) < 0.05]
    
    if well_calibrated:
        print(f"• Modelli ben calibrati: {', '.join(well_calibrated)}")
    else:
        print(f"• Nessun modello è ben calibrato - necessaria calibrazione!")
    
    print(f"\n✅ ANALISI COMPLETA SALVATA!")
    
    return {
        'results_full': results_full,
        'results_diag': results_diag,
        'best_full': best_full,
        'best_diag': best_diag,
        'best_lr': best_lr,
        'best_svm': best_svm,
        'best_overall': best_overall,
        'all_models': all_models
    }

if __name__ == "__main__":
    results = main()