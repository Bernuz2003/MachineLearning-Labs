import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
from pathlib import Path

# Import necessary modules
sys.path.append('/home/bernuz/Scrivania/UNIVERSITA\'/Machine Learning and Pattern Recognition/Labs/08_Calibration Fusion')
import bayesRisk
import logReg

# Also import from GMM lab for model regeneration
sys.path.append('../07_Gaussian Mixture Models')
try:
    from gmm import *
    GMM_AVAILABLE = True
except ImportError:
    print("⚠️  GMM module not available")
    GMM_AVAILABLE = False

# Import from other labs for model regeneration with explicit module names
try:
    sys.path.insert(0, '../05_Logistic Regression')
    # Import with a different name to avoid conflict
    import project
    # Remove current path to avoid conflicts
    sys.path.remove('../05_Logistic Regression')
    
    # Save the functions we need
    trainWeightedLogRegBinary = project.trainWeightedLogRegBinary
    trainLogRegBinary = project.trainLogRegBinary
    
    LR_AVAILABLE = True
    print("✅ Logistic Regression functions imported")
except (ImportError, AttributeError) as e:
    print(f"⚠️  Logistic Regression module not available: {e}")
    trainWeightedLogRegBinary = None
    trainLogRegBinary = None
    LR_AVAILABLE = False

try:
    sys.path.insert(0, '../06_Support Vector Machine')
    # Import SVM module
    import sol as svm_module
    
    # Save the functions we need
    train_dual_SVM_linear = svm_module.train_dual_SVM_linear
    train_dual_SVM_kernel = svm_module.train_dual_SVM_kernel
    polyKernel = svm_module.polyKernel
    rbfKernel = svm_module.rbfKernel
    
    # Remove from path
    sys.path.remove('../06_Support Vector Machine')
    
    SVM_AVAILABLE = True
    print("✅ SVM functions imported")
except (ImportError, AttributeError) as e:
    print(f"⚠️  SVM module not available: {e}")
    train_dual_SVM_linear = None
    train_dual_SVM_kernel = None
    polyKernel = None
    rbfKernel = None
    SVM_AVAILABLE = False

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

def create_evaluation_set(D, L, eval_fraction=0.2, seed=42):
    """Create evaluation set from training data (since evalData.txt doesn't exist)"""
    print(f"ℹ️  Creating evaluation set from training data ({eval_fraction*100:.0f}% of data)")
    
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    
    n_eval = int(D.shape[1] * eval_fraction)
    eval_idx = idx[:n_eval]
    remaining_idx = idx[n_eval:]
    
    D_eval = D[:, eval_idx]
    L_eval = L[eval_idx]
    D_remaining = D[:, remaining_idx]
    L_remaining = L[remaining_idx]
    
    print(f"📊 Evaluation set: {D_eval.shape[1]} samples")
    print(f"📊 Remaining for training/validation: {D_remaining.shape[1]} samples")
    
    return D_eval, L_eval, D_remaining, L_remaining

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

def extract_train_val_folds_from_ary(X, idx, KFOLD=5):
    """Extract i-th fold from a 1-D numpy array for K-fold"""
    return np.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != idx]), X[idx::KFOLD]

def bayesPlot(S, L, left=-3, right=3, npts=21):
    """Generate Bayes error plot data"""
    effPriorLogOdds = np.linspace(left, right, npts)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        actDCF.append(bayesRisk.compute_actDCF_binary_fast(S, L, effPrior, 1.0, 1.0))
        minDCF.append(bayesRisk.compute_minDCF_binary_fast(S, L, effPrior, 1.0, 1.0))
    return effPriorLogOdds, actDCF, minDCF

# ---------------------------------------------------------------------
# ---------------------  SCORE LOADING/GENERATION  --------------------
# ---------------------------------------------------------------------

def load_best_models_from_gmm_lab():
    """Load best models from GMM lab"""
    try:
        with open('../07_Gaussian Mixture Models/best_models_for_calibration.pkl', 'rb') as f:
            data = pickle.load(f)
        print("✅ Loaded best models from GMM lab")
        return data
    except FileNotFoundError:
        print("⚠️  GMM lab results not found - will generate simplified models")
        return None

def regenerate_model_scores(model_info, DTR, LTR, DVAL, LVAL):
    """Regenerate scores for a specific model based on saved info"""
    
    model_name = model_info.get('model_info', 'Unknown')
    print(f"  Regenerating scores for: {model_name}")
    
    # Determine model type from model_info
    if 'GMM' in model_name:
        if not GMM_AVAILABLE:
            print("    ⚠️  GMM not available, skipping")
            return None
            
        # Extract parameters from model_info
        cov_type = 'full' if 'full' in model_name.lower() else 'diagonal'
        
        # Extract number of components (default to 4 if not found)
        import re
        comp_match = re.search(r'\((\d+) comp\)', model_name)
        num_components = int(comp_match.group(1)) if comp_match else 4
        
        # Train GMMs
        gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], num_components, 
                                covType=cov_type, verbose=False, psiEig=0.01)
        gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], num_components, 
                                covType=cov_type, verbose=False, psiEig=0.01)
        
        # Compute scores
        ll0 = logpdf_GMM(DVAL, gmm0)
        ll1 = logpdf_GMM(DVAL, gmm1)
        scores = ll1 - ll0
        
    elif 'LogReg' in model_name:
        if not LR_AVAILABLE:
            print("    ⚠️  Logistic Regression not available, skipping")
            return None
            
        # Extract lambda value - improved regex for scientific notation
        import re
        lambda_match = re.search(r'λ=([0-9.e-]+)', model_name)
        lamb = float(lambda_match.group(1)) if lambda_match else 1e-3
        
        if 'Weighted' in model_name:
            # Extract pT value
            pt_match = re.search(r'pT=([0-9.]+)', model_name)
            pT = float(pt_match.group(1)) if pt_match else 0.1
            
            w, b = trainWeightedLogRegBinary(DTR, LTR, lamb, pT)
            scores = (w.T @ DVAL + b - np.log(pT / (1 - pT))).ravel()
        else:
            pEmp = (LTR == 1).sum() / LTR.size
            w, b = trainLogRegBinary(DTR, LTR, lamb)
            scores = (w.T @ DVAL + b - np.log(pEmp / (1 - pEmp))).ravel()
            
    elif 'SVM' in model_name:
        if not SVM_AVAILABLE:
            print("    ⚠️  SVM not available, skipping")
            return None
            
        # Extract C value - improved regex for scientific notation
        import re
        c_match = re.search(r'C=([0-9.e+-]+)', model_name)
        C = float(c_match.group(1)) if c_match else 0.1
        
        if 'Linear' in model_name:
            w, b = train_dual_SVM_linear(DTR, LTR, C, K=1.0)
            scores = (vrow(w) @ DVAL + b).ravel()
        elif 'RBF' in model_name:
            # Extract gamma value - improved regex
            gamma_match = re.search(r'γ=([0-9.e+-]+)', model_name)
            gamma = float(gamma_match.group(1)) if gamma_match else 1.0
            
            kernelFunc = rbfKernel(gamma)
            fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps=1.0)
            scores = fScore(DVAL)
        elif 'Poly' in model_name:
            # Extract degree and c values
            deg_match = re.search(r'd=(\d+)', model_name)
            c_param_match = re.search(r'c=(\d+)', model_name)
            degree = int(deg_match.group(1)) if deg_match else 2
            c_param = int(c_param_match.group(1)) if c_param_match else 1
            
            kernelFunc = polyKernel(degree, c_param)
            fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps=1.0)
            scores = fScore(DVAL)
        else:
            print("    ⚠️  Unknown SVM type, using linear")
            w, b = train_dual_SVM_linear(DTR, LTR, C, K=1.0)
            scores = (vrow(w) @ DVAL + b).ravel()
    else:
        print(f"    ⚠️  Unknown model type: {model_name}")
        return None
    
    return scores

def load_and_regenerate_scores(DTR, LTR, DVAL, LVAL):
    """Load best models and regenerate scores for current data split"""
    
    # Load saved models
    saved_data = load_best_models_from_gmm_lab()
    
    if saved_data is None:
        print("⚠️  No saved models found, generating simplified models")
        return generate_simplified_scores(DTR, LTR, DVAL, LVAL)
    
    print(f"\n🔄 Regenerating scores for best models...")
    
    best_models = saved_data['best_models']
    scores_dict = {}
    
    for model_name, model_info in best_models.items():
        if model_info is None:
            continue
            
        print(f"\n📊 Processing {model_name}...")
        
        # Regenerate scores
        scores = regenerate_model_scores(model_info, DTR, LTR, DVAL, LVAL)
        
        if scores is not None:
            # Evaluate performance
            minDCF = bayesRisk.compute_minDCF_binary_fast(scores, LVAL, 0.1, 1.0, 1.0)
            actDCF = bayesRisk.compute_actDCF_binary_fast(scores, LVAL, 0.1, 1.0, 1.0)
            
            scores_dict[model_name] = {
                'scores': scores,
                'model_info': model_info.get('model_info', f'{model_name} model'),
                'minDCF': minDCF,
                'actDCF': actDCF
            }
            
            print(f"    ✅ minDCF: {minDCF:.4f}, actDCF: {actDCF:.4f}")
        else:
            print(f"    ❌ Failed to regenerate scores")
    
    if len(scores_dict) == 0:
        print("⚠️  No models could be regenerated, using simplified fallback")
        return generate_simplified_scores(DTR, LTR, DVAL, LVAL)
    
    return scores_dict

def generate_simplified_scores(DTR, LTR, DVAL, LVAL):
    """Generate simplified scores as fallback"""
    print("🔄 Generating simplified models as fallback...")
    
    scores = {}
    
    # Simple Logistic Regression
    if LR_AVAILABLE:
        print("  Training simple Logistic Regression...")
        try:
            w_lr, b_lr = trainWeightedLogRegBinary(DTR, LTR, 1e-3, 0.1)
            scores_lr = (w_lr.T @ DVAL + b_lr - np.log(0.1 / 0.9)).ravel()
            scores['Logistic Regression'] = {
                'scores': scores_lr,
                'model_info': 'Weighted LogReg (λ=1e-3, pT=0.1)',
                'minDCF': bayesRisk.compute_minDCF_binary_fast(scores_lr, LVAL, 0.1, 1.0, 1.0),
                'actDCF': bayesRisk.compute_actDCF_binary_fast(scores_lr, LVAL, 0.1, 1.0, 1.0)
            }
        except Exception as e:
            print(f"    Error: {e}")
    
    # Simple SVM
    if SVM_AVAILABLE:
        print("  Training simple SVM...")
        try:
            w_svm, b_svm = train_dual_SVM_linear(DTR, LTR, 0.1, K=1.0)
            scores_svm = (vrow(w_svm) @ DVAL + b_svm).ravel()
            scores['SVM'] = {
                'scores': scores_svm,
                'model_info': 'Linear SVM (C=0.1)',
                'minDCF': bayesRisk.compute_minDCF_binary_fast(scores_svm, LVAL, 0.1, 1.0, 1.0),
                'actDCF': bayesRisk.compute_actDCF_binary_fast(scores_svm, LVAL, 0.1, 1.0, 1.0)
            }
        except Exception as e:
            print(f"    Error: {e}")
    
    # Simple GMM
    if GMM_AVAILABLE:
        print("  Training simple GMM...")
        try:
            gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], 4, covType='full', verbose=False, psiEig=0.01)
            gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], 4, covType='full', verbose=False, psiEig=0.01)
            
            ll0 = logpdf_GMM(DVAL, gmm0)
            ll1 = logpdf_GMM(DVAL, gmm1)
            scores_gmm = ll1 - ll0
            
            scores['GMM Full'] = {
                'scores': scores_gmm,
                'model_info': 'Full GMM (4 components)',
                'minDCF': bayesRisk.compute_minDCF_binary_fast(scores_gmm, LVAL, 0.1, 1.0, 1.0),
                'actDCF': bayesRisk.compute_actDCF_binary_fast(scores_gmm, LVAL, 0.1, 1.0, 1.0)
            }
        except Exception as e:
            print(f"    Error: {e}")
    
    return scores

# ---------------------------------------------------------------------
# ---------------------  CALIBRATION FUNCTIONS  -----------------------
# ---------------------------------------------------------------------

def kfold_calibration_analysis(scores, labels, target_prior=0.1, KFOLD=5):
    """Perform K-fold calibration analysis"""
    print(f"\n🔄 K-FOLD CALIBRATION ANALYSIS (K={KFOLD})")
    print("-" * 60)
    
    # Test different training priors
    training_priors = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
    
    results = {}
    
    for pT in training_priors:
        print(f"  Testing training prior pT = {pT:.1f}...")
        
        calibrated_scores = []
        calibrated_labels = []
        
        # K-fold cross-validation
        for foldIdx in range(KFOLD):
            # Split into calibration training and validation
            SCAL, SVAL = extract_train_val_folds_from_ary(scores, foldIdx, KFOLD)
            LCAL, LVAL_fold = extract_train_val_folds_from_ary(labels, foldIdx, KFOLD)
            
            # Train calibration model
            w, b = logReg.trainWeightedLogRegBinary(vrow(SCAL), LCAL, 0, pT)
            
            # Apply calibration to validation fold
            calibrated_SVAL = (w.T @ vrow(SVAL) + b - np.log(pT / (1-pT))).ravel()
            
            calibrated_scores.append(calibrated_SVAL)
            calibrated_labels.append(LVAL_fold)
        
        # Pool results
        pooled_scores = np.hstack(calibrated_scores)
        pooled_labels = np.hstack(calibrated_labels)
        
        # Evaluate
        minDCF = bayesRisk.compute_minDCF_binary_fast(pooled_scores, pooled_labels, target_prior, 1.0, 1.0)
        actDCF = bayesRisk.compute_actDCF_binary_fast(pooled_scores, pooled_labels, target_prior, 1.0, 1.0)
        
        results[pT] = {
            'minDCF': minDCF,
            'actDCF': actDCF,
            'calibrated_scores': pooled_scores,
            'calibrated_labels': pooled_labels
        }
        
        print(f"    pT = {pT:.1f}: minDCF = {minDCF:.4f}, actDCF = {actDCF:.4f}")
    
    # Find best training prior
    best_pT = min(training_priors, key=lambda p: results[p]['actDCF'])
    
    print(f"\n🏆 Best training prior: pT = {best_pT:.1f}")
    print(f"📊 Best actDCF: {results[best_pT]['actDCF']:.4f}")
    print(f"📊 Corresponding minDCF: {results[best_pT]['minDCF']:.4f}")
    
    return results, best_pT

def final_calibration_model(scores, labels, best_pT, target_prior=0.1):
    """Train final calibration model on full dataset"""
    print(f"\n🎯 Training final calibration model (pT = {best_pT:.1f})...")
    
    # Train on full dataset
    w, b = logReg.trainWeightedLogRegBinary(vrow(scores), labels, 0, best_pT)
    
    def calibrate_scores(test_scores):
        return (w.T @ vrow(test_scores) + b - np.log(best_pT / (1-best_pT))).ravel()
    
    return calibrate_scores, w, b

# ---------------------------------------------------------------------
# ---------------------  FUSION FUNCTIONS  ----------------------------
# ---------------------------------------------------------------------

def kfold_fusion_analysis(scores_dict, labels, target_prior=0.1, KFOLD=5):
    """Perform K-fold fusion analysis"""
    print(f"\n🔗 K-FOLD FUSION ANALYSIS")
    print("-" * 60)
    
    # Get score arrays
    score_names = list(scores_dict.keys())
    score_arrays = [scores_dict[name]['scores'] for name in score_names]
    
    print(f"📊 Fusing {len(score_names)} systems: {score_names}")
    
    # Test different training priors
    training_priors = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
    
    results = {}
    
    for pT in training_priors:
        print(f"  Testing training prior pT = {pT:.1f}...")
        
        fused_scores = []
        fused_labels = []
        
        # K-fold cross-validation
        for foldIdx in range(KFOLD):
            # Split each score array
            SCAL_arrays = []
            SVAL_arrays = []
            
            for score_array in score_arrays:
                SCAL, SVAL = extract_train_val_folds_from_ary(score_array, foldIdx, KFOLD)
                SCAL_arrays.append(SCAL)
                SVAL_arrays.append(SVAL)
            
            LCAL, LVAL_fold = extract_train_val_folds_from_ary(labels, foldIdx, KFOLD)
            
            # Build feature matrices
            SCAL_matrix = np.vstack(SCAL_arrays)
            SVAL_matrix = np.vstack(SVAL_arrays)
            
            # Train fusion model
            w, b = logReg.trainWeightedLogRegBinary(SCAL_matrix, LCAL, 0, pT)
            
            # Apply fusion to validation fold
            fused_SVAL = (w.T @ SVAL_matrix + b - np.log(pT / (1-pT))).ravel()
            
            fused_scores.append(fused_SVAL)
            fused_labels.append(LVAL_fold)
        
        # Pool results
        pooled_scores = np.hstack(fused_scores)
        pooled_labels = np.hstack(fused_labels)
        
        # Evaluate
        minDCF = bayesRisk.compute_minDCF_binary_fast(pooled_scores, pooled_labels, target_prior, 1.0, 1.0)
        actDCF = bayesRisk.compute_actDCF_binary_fast(pooled_scores, pooled_labels, target_prior, 1.0, 1.0)
        
        results[pT] = {
            'minDCF': minDCF,
            'actDCF': actDCF,
            'fused_scores': pooled_scores,
            'fused_labels': pooled_labels
        }
        
        print(f"    pT = {pT:.1f}: minDCF = {minDCF:.4f}, actDCF = {actDCF:.4f}")
    
    # Find best training prior
    best_pT = min(training_priors, key=lambda p: results[p]['actDCF'])
    
    print(f"\n🏆 Best fusion training prior: pT = {best_pT:.1f}")
    print(f"📊 Best fusion actDCF: {results[best_pT]['actDCF']:.4f}")
    print(f"📊 Corresponding fusion minDCF: {results[best_pT]['minDCF']:.4f}")
    
    return results, best_pT

def final_fusion_model(scores_dict, labels, best_pT):
    """Train final fusion model on full dataset"""
    print(f"\n🎯 Training final fusion model (pT = {best_pT:.1f})...")
    
    # Build feature matrix
    score_arrays = [scores_dict[name]['scores'] for name in scores_dict.keys()]
    score_matrix = np.vstack(score_arrays)
    
    # Train on full dataset
    w, b = logReg.trainWeightedLogRegBinary(score_matrix, labels, 0, best_pT)
    
    def fuse_scores(test_scores_dict):
        test_arrays = [test_scores_dict[name] for name in scores_dict.keys()]
        test_matrix = np.vstack(test_arrays)
        return (w.T @ test_matrix + b - np.log(best_pT / (1-best_pT))).ravel()
    
    return fuse_scores, w, b

# ---------------------------------------------------------------------
# ---------------------  VISUALIZATION FUNCTIONS  ---------------------
# ---------------------------------------------------------------------

def plot_calibration_comparison(original_scores, calibrated_scores, labels, model_name):
    """Plot calibration comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original scores
    logOdds, actDCF_orig, minDCF_orig = bayesPlot(original_scores, labels)
    axes[0].plot(logOdds, minDCF_orig, 'b--', linewidth=2, label='minDCF')
    axes[0].plot(logOdds, actDCF_orig, 'b-', linewidth=2, label='actDCF (original)')
    axes[0].set_title(f'{model_name} - Original Scores')
    axes[0].set_xlabel('log-odds')
    axes[0].set_ylabel('DCF')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # Calibrated scores
    logOdds, actDCF_cal, minDCF_cal = bayesPlot(calibrated_scores, labels)
    axes[1].plot(logOdds, minDCF_cal, 'r--', linewidth=2, label='minDCF')
    axes[1].plot(logOdds, actDCF_cal, 'r-', linewidth=2, label='actDCF (calibrated)')
    axes[1].set_title(f'{model_name} - Calibrated Scores')
    axes[1].set_xlabel('log-odds')
    axes[1].set_ylabel('DCF')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    # Comparison
    axes[2].plot(logOdds, minDCF_orig, 'b--', linewidth=2, label='minDCF')
    axes[2].plot(logOdds, actDCF_orig, 'b:', linewidth=2, label='actDCF (original)')
    axes[2].plot(logOdds, actDCF_cal, 'r-', linewidth=2, label='actDCF (calibrated)')
    axes[2].axvline(np.log(0.1/0.9), color='black', linestyle='-', alpha=0.5, label='Target app (π=0.1)')
    axes[2].set_title(f'{model_name} - Calibration Effect')
    axes[2].set_xlabel('log-odds')
    axes[2].set_ylabel('DCF')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1)
    
    plt.tight_layout()
    filename = f'calibration_comparison_{model_name.lower().replace(" ", "_").replace("/", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_fusion_comparison(individual_scores, fused_scores, labels, score_names):
    """Plot fusion comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Individual systems
    for i, (name, scores) in enumerate(individual_scores.items()):
        logOdds, actDCF, minDCF = bayesPlot(scores, labels)
        axes[0].plot(logOdds, minDCF, '--', color=colors[i], linewidth=2, label=f'{name} - minDCF')
        axes[0].plot(logOdds, actDCF, '-', color=colors[i], linewidth=2, label=f'{name} - actDCF')
    
    axes[0].axvline(np.log(0.1/0.9), color='black', linestyle='-', alpha=0.5, label='Target app (π=0.1)')
    axes[0].set_title('Individual Systems')
    axes[0].set_xlabel('log-odds')
    axes[0].set_ylabel('DCF')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # Fusion comparison
    logOdds, actDCF_fused, minDCF_fused = bayesPlot(fused_scores, labels)
    
    # Plot best individual system for comparison
    best_name = min(individual_scores.keys(), 
                   key=lambda x: bayesRisk.compute_actDCF_binary_fast(individual_scores[x], labels, 0.1, 1.0, 1.0))
    logOdds, actDCF_best, minDCF_best = bayesPlot(individual_scores[best_name], labels)
    
    axes[1].plot(logOdds, minDCF_best, 'b--', linewidth=2, label=f'Best individual ({best_name}) - minDCF')
    axes[1].plot(logOdds, actDCF_best, 'b:', linewidth=2, label=f'Best individual ({best_name}) - actDCF')
    axes[1].plot(logOdds, minDCF_fused, 'r--', linewidth=2, label='Fusion - minDCF')
    axes[1].plot(logOdds, actDCF_fused, 'r-', linewidth=2, label='Fusion - actDCF')
    axes[1].axvline(np.log(0.1/0.9), color='black', linestyle='-', alpha=0.5, label='Target app (π=0.1)')
    
    axes[1].set_title('Fusion vs Best Individual')
    axes[1].set_xlabel('log-odds')
    axes[1].set_ylabel('DCF')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('fusion_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# ---------------------------------------------------------------------
# ---------------------  EVALUATION FUNCTIONS  ------------------------
# ---------------------------------------------------------------------

def evaluate_on_evaluation_set(calibration_models, fusion_model, scores_dict, 
                              eval_scores_dict, eval_labels, target_prior=0.1):
    """Evaluate models on evaluation set"""
    print(f"\n{'='*80}")
    print(f"📊 EVALUATION ON EVALUATION SET")
    print(f"{'='*80}")
    
    results = {}
    
    # Evaluate individual calibrated systems
    print(f"\n🔍 Individual Systems (Calibrated):")
    print("-" * 50)
    
    for model_name in scores_dict.keys():
        if model_name in calibration_models:
            calibrate_func = calibration_models[model_name]
            eval_scores = eval_scores_dict[model_name]
            
            # Apply calibration
            calibrated_eval_scores = calibrate_func(eval_scores)
            
            # Evaluate
            minDCF = bayesRisk.compute_minDCF_binary_fast(calibrated_eval_scores, eval_labels, target_prior, 1.0, 1.0)
            actDCF = bayesRisk.compute_actDCF_binary_fast(calibrated_eval_scores, eval_labels, target_prior, 1.0, 1.0)
            
            results[f'{model_name}_calibrated'] = {
                'scores': calibrated_eval_scores,
                'minDCF': minDCF,
                'actDCF': actDCF
            }
            
            print(f"  {model_name:15s}: minDCF = {minDCF:.4f}, actDCF = {actDCF:.4f}")
    
    # Evaluate fusion
    print(f"\n🔗 Fusion System:")
    print("-" * 30)
    
    fused_eval_scores = fusion_model(eval_scores_dict)
    minDCF_fusion = bayesRisk.compute_minDCF_binary_fast(fused_eval_scores, eval_labels, target_prior, 1.0, 1.0)
    actDCF_fusion = bayesRisk.compute_actDCF_binary_fast(fused_eval_scores, eval_labels, target_prior, 1.0, 1.0)
    
    results['fusion'] = {
        'scores': fused_eval_scores,
        'minDCF': minDCF_fusion,
        'actDCF': actDCF_fusion
    }
    
    print(f"  Fusion          : minDCF = {minDCF_fusion:.4f}, actDCF = {actDCF_fusion:.4f}")
    
    # Find best system
    best_system = min(results.keys(), key=lambda x: results[x]['actDCF'])
    
    print(f"\n🏆 Best System on Evaluation Set: {best_system}")
    print(f"📊 Best actDCF: {results[best_system]['actDCF']:.4f}")
    
    return results, best_system

# ---------------------------------------------------------------------
# ---------------------  MAIN EXECUTION  ------------------------------
# ---------------------------------------------------------------------

def main():
    print("=" * 80)
    print("    CALIBRATION AND FUSION - PROGETTO ML&PR")
    print("=" * 80)
    
    # Load data and create evaluation set
    print("\n🔄 Loading and preparing datasets...")
    D, L = load_project_data()
    
    # Create evaluation set (since evalData.txt doesn't exist)
    D_eval, L_eval, D_remaining, L_remaining = create_evaluation_set(D, L, eval_fraction=0.2, seed=42)
    
    print(f"📊 Total dataset: {D.shape[1]} samples, {D.shape[0]} features")
    print(f"📊 Class distribution (total): Class 0: {(L==0).sum()}, Class 1: {(L==1).sum()}")
    
    # Split remaining data for training/validation (same as previous labs)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D_remaining, L_remaining, seed=0)
    print(f"📊 Training set: {DTR.shape[1]} samples")
    print(f"📊 Validation set: {DVAL.shape[1]} samples (for calibration)")
    print(f"📊 Evaluation set: {D_eval.shape[1]} samples")
    
    # Load or generate scores from best models
    print(f"\n🔄 Loading/regenerating scores from best models...")
    scores_dict = load_and_regenerate_scores(DTR, LTR, DVAL, LVAL)
    
    print(f"\n📊 Available models:")
    for name, info in scores_dict.items():
        print(f"  • {name}: {info['model_info']}")
        print(f"    minDCF = {info['minDCF']:.4f}, actDCF = {info['actDCF']:.4f}")
    
    # Generate evaluation scores
    print(f"\n🔄 Generating evaluation scores...")
    eval_scores_dict = {}
    
    saved_data = load_best_models_from_gmm_lab()
    if saved_data:
        best_models = saved_data['best_models']
        for model_name, model_info in best_models.items():
            if model_info is None or model_name not in scores_dict:
                continue
            scores = regenerate_model_scores(model_info, DTR, LTR, D_eval, L_eval)
            if scores is not None:
                eval_scores_dict[model_name] = scores
    else:
        # Fallback: regenerate on evaluation set
        for name in scores_dict.keys():
            print(f"  Regenerating {name} on evaluation set...")
            # This would need the saved model info - for now skip
            pass
    
    target_prior = 0.1
    
    # 1. CALIBRATION ANALYSIS
    print(f"\n{'🎯'*40}")
    print("1️⃣  CALIBRATION ANALYSIS")
    print('🎯' * 40)
    
    calibration_models = {}
    calibration_results = {}
    
    for model_name, model_info in scores_dict.items():
        print(f"\n📊 Calibrating {model_name}...")
        scores = model_info['scores']
        
        # K-fold calibration analysis
        cal_results, best_pT = kfold_calibration_analysis(scores, LVAL, target_prior)
        calibration_results[model_name] = cal_results
        
        # Train final calibration model
        calibrate_func, w, b = final_calibration_model(scores, LVAL, best_pT, target_prior)
        calibration_models[model_name] = calibrate_func
        
        # Apply calibration and visualize
        calibrated_scores = calibrate_func(scores)
        plot_calibration_comparison(scores, calibrated_scores, LVAL, model_name)
        
        # Print improvement
        original_actDCF = bayesRisk.compute_actDCF_binary_fast(scores, LVAL, target_prior, 1.0, 1.0)
        calibrated_actDCF = bayesRisk.compute_actDCF_binary_fast(calibrated_scores, LVAL, target_prior, 1.0, 1.0)
        improvement = original_actDCF - calibrated_actDCF
        
        print(f"✅ {model_name} calibration complete:")
        print(f"   Original actDCF: {original_actDCF:.4f}")
        print(f"   Calibrated actDCF: {calibrated_actDCF:.4f}")
        print(f"   Improvement: {improvement:.4f}")
    
    # 2. FUSION ANALYSIS
    print(f"\n{'🔗'*40}")
    print("2️⃣  FUSION ANALYSIS")
    print('🔗' * 40)
    
    # K-fold fusion analysis
    fusion_results, best_fusion_pT = kfold_fusion_analysis(scores_dict, LVAL, target_prior)
    
    # Train final fusion model
    fusion_model, w_fusion, b_fusion = final_fusion_model(scores_dict, LVAL, best_fusion_pT)
    
    # Apply fusion and visualize
    individual_scores = {name: info['scores'] for name, info in scores_dict.items()}
    fused_scores = fusion_model(individual_scores)
    
    plot_fusion_comparison(individual_scores, fused_scores, LVAL, list(scores_dict.keys()))
    
    # Compare fusion with individual systems
    print(f"\n📊 FUSION vs INDIVIDUAL SYSTEMS:")
    print("-" * 50)
    
    for name, info in scores_dict.items():
        calibrated_scores = calibration_models[name](info['scores'])
        actDCF_cal = bayesRisk.compute_actDCF_binary_fast(calibrated_scores, LVAL, target_prior, 1.0, 1.0)
        print(f"  {name} (calibrated): actDCF = {actDCF_cal:.4f}")
    
    actDCF_fusion = bayesRisk.compute_actDCF_binary_fast(fused_scores, LVAL, target_prior, 1.0, 1.0)
    print(f"  Fusion:              actDCF = {actDCF_fusion:.4f}")
    
    # 3. MODEL SELECTION
    print(f"\n{'🎯'*40}")
    print("3️⃣  FINAL MODEL SELECTION")
    print('🎯' * 40)
    
    # Compare all options
    all_options = {}
    
    for name in scores_dict.keys():
        calibrated_scores = calibration_models[name](scores_dict[name]['scores'])
        actDCF = bayesRisk.compute_actDCF_binary_fast(calibrated_scores, LVAL, target_prior, 1.0, 1.0)
        all_options[f"{name}_calibrated"] = actDCF
    
    all_options["fusion"] = actDCF_fusion
    
    # Select best model
    best_model = min(all_options.keys(), key=lambda x: all_options[x])
    
    print(f"\n🏆 FINAL MODEL SELECTION:")
    print("-" * 40)
    for model, actDCF in sorted(all_options.items(), key=lambda x: x[1]):
        status = "👑 SELECTED" if model == best_model else ""
        print(f"  {model:20s}: actDCF = {actDCF:.4f} {status}")
    
    print(f"\n✅ Selected model: {best_model}")
    print(f"📊 Validation actDCF: {all_options[best_model]:.4f}")
    
    # 4. EVALUATION (if evaluation scores available)
    if eval_scores_dict:
        print(f"\n{'📊'*40}")
        print("4️⃣  EVALUATION ON EVALUATION SET")
        print('📊' * 40)
        
        eval_results, best_eval_system = evaluate_on_evaluation_set(
            calibration_models, fusion_model, scores_dict, 
            eval_scores_dict, L_eval, target_prior
        )
        
        # Plot evaluation results
        plt.figure(figsize=(15, 8))
        
        # Bayes error plots for evaluation
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        
        plt.subplot(1, 2, 1)
        for i, (system_name, result) in enumerate(eval_results.items()):
            logOdds, actDCF, minDCF = bayesPlot(result['scores'], L_eval)
            plt.plot(logOdds, minDCF, '--', color=colors[i % len(colors)], linewidth=2, label=f'{system_name} - minDCF')
            plt.plot(logOdds, actDCF, '-', color=colors[i % len(colors)], linewidth=2, label=f'{system_name} - actDCF')
        
        plt.axvline(np.log(0.1/0.9), color='black', linestyle='-', alpha=0.5, label='Target app (π=0.1)')
        plt.title('Evaluation Set - All Systems')
        plt.xlabel('log-odds')
        plt.ylabel('DCF')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Compare validation vs evaluation for best system
        plt.subplot(1, 2, 2)
        
        if best_model == "fusion":
            val_scores = fused_scores
            eval_scores = eval_results['fusion']['scores']
        else:
            model_name = best_model.replace("_calibrated", "")
            val_scores = calibration_models[model_name](scores_dict[model_name]['scores'])
            eval_scores = eval_results[best_model]['scores']
        
        logOdds, actDCF_val, minDCF_val = bayesPlot(val_scores, LVAL)
        logOdds, actDCF_eval, minDCF_eval = bayesPlot(eval_scores, L_eval)
        
        plt.plot(logOdds, minDCF_val, 'b--', linewidth=2, label='Validation - minDCF')
        plt.plot(logOdds, actDCF_val, 'b-', linewidth=2, label='Validation - actDCF')
        plt.plot(logOdds, minDCF_eval, 'r--', linewidth=2, label='Evaluation - minDCF')
        plt.plot(logOdds, actDCF_eval, 'r-', linewidth=2, label='Evaluation - actDCF')
        plt.axvline(np.log(0.1/0.9), color='black', linestyle='-', alpha=0.5, label='Target app (π=0.1)')
        
        plt.title(f'Best Model ({best_model}) - Validation vs Evaluation')
        plt.xlabel('log-odds')
        plt.ylabel('DCF')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. FINAL ANALYSIS AND CONCLUSIONS
        print(f"\n{'💡'*40}")
        print("5️⃣  ANALYSIS AND CONCLUSIONS")
        print('💡' * 40)
        
        print(f"\n📊 VALIDATION vs EVALUATION:")
        print("-" * 35)
        val_actDCF = all_options[best_model]
        eval_actDCF = eval_results[best_eval_system]['actDCF']
        generalization_gap = eval_actDCF - val_actDCF
        
        print(f"  Validation actDCF:  {val_actDCF:.4f}")
        print(f"  Evaluation actDCF:  {eval_actDCF:.4f}")
        print(f"  Generalization gap: {generalization_gap:.4f}")
        
        if abs(generalization_gap) < 0.01:
            print(f"  ✅ Good generalization!")
        elif generalization_gap > 0.05:
            print(f"  ⚠️  Significant performance drop on evaluation")
        else:
            print(f"  ℹ️  Acceptable generalization")
    else:
        print(f"\n⚠️  Evaluation set scores not available - skipping evaluation analysis")
    
    print(f"\n🔍 CALIBRATION EFFECTIVENESS:")
    print("-" * 40)
    for model_name in scores_dict.keys():
        original_actDCF = bayesRisk.compute_actDCF_binary_fast(scores_dict[model_name]['scores'], LVAL, target_prior, 1.0, 1.0)
        calibrated_scores = calibration_models[model_name](scores_dict[model_name]['scores'])
        calibrated_actDCF = bayesRisk.compute_actDCF_binary_fast(calibrated_scores, LVAL, target_prior, 1.0, 1.0)
        improvement = original_actDCF - calibrated_actDCF
        
        print(f"  {model_name}:")
        print(f"    Original:   {original_actDCF:.4f}")
        print(f"    Calibrated: {calibrated_actDCF:.4f}")
        print(f"    Improvement: {improvement:.4f} ({improvement/original_actDCF*100:+.1f}%)")
    
    print(f"\n🔗 FUSION EFFECTIVENESS:")
    print("-" * 30)
    best_individual = min(scores_dict.keys(), 
                         key=lambda x: bayesRisk.compute_actDCF_binary_fast(
                             calibration_models[x](scores_dict[x]['scores']), LVAL, target_prior, 1.0, 1.0))
    
    best_individual_actDCF = bayesRisk.compute_actDCF_binary_fast(
        calibration_models[best_individual](scores_dict[best_individual]['scores']), LVAL, target_prior, 1.0, 1.0)
    
    fusion_improvement = best_individual_actDCF - actDCF_fusion
    
    print(f"  Best individual ({best_individual}): {best_individual_actDCF:.4f}")
    print(f"  Fusion:                              {actDCF_fusion:.4f}")
    print(f"  Improvement: {fusion_improvement:.4f} ({fusion_improvement/best_individual_actDCF*100:+.1f}%)")
    
    # Save final results
    print(f"\n💾 Saving results...")
    final_results = {
        'best_model': best_model,
        'calibration_models': calibration_results,
        'fusion_results': fusion_results,
        'final_metrics': {
            'validation_actDCF': all_options[best_model],
        },
        'model_info': {
            'scores_dict': {name: info['model_info'] for name, info in scores_dict.items()},
            'target_prior': target_prior
        }
    }
    
    with open('calibration_fusion_results.pkl', 'wb') as f:
        pickle.dump(final_results, f)
    
    print(f"✅ Results saved to 'calibration_fusion_results.pkl'")
    print(f"✅ Plots saved as PNG files")
    
    print(f"\n" + "=" * 80)
    print("✨ CALIBRATION AND FUSION ANALYSIS COMPLETE ✨")
    print("=" * 80)
    
    return final_results

if __name__ == "__main__":
    results = main()