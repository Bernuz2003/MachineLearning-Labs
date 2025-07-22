import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from pathlib import Path

# Import functions from bayes_model.py
def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp):
    th = -np.log( (prior * Cfn) / ((1 - prior) * Cfp) )
    return np.int32(llr > th)

def compute_confusion_matrix(predictedLabels, classLabels):
    nClasses = classLabels.max() + 1
    M = np.zeros((nClasses, nClasses), dtype=np.int32)
    for i in range(classLabels.size):
        M[predictedLabels[i], classLabels[i]] += 1
    return M

def compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=True):
    M = compute_confusion_matrix(predictedLabels, classLabels)
    Pfn = M[0,1] / (M[0,1] + M[1,1])
    Pfp = M[1,0] / (M[0,0] + M[1,0])
    bayesError = prior * Cfn * Pfn + (1-prior) * Cfp * Pfp
    if normalize:
        return bayesError / np.minimum(prior * Cfn, (1-prior)*Cfp)
    return bayesError

def compute_empirical_Bayes_risk_binary_llr_optimal_decisions(llr, classLabels, prior, Cfn, Cfp, normalize=True):
    predictedLabels = compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp)
    return compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=normalize)

def compute_Pfn_Pfp_allThresholds_fast(llr, classLabels):
    llrSorter = np.argsort(llr)
    llrSorted = llr[llrSorter]
    classLabelsSorted = classLabels[llrSorter]

    Pfp = []
    Pfn = []
    
    nTrue = (classLabelsSorted==1).sum()
    nFalse = (classLabelsSorted==0).sum()
    nFalseNegative = 0
    nFalsePositive = nFalse
    
    Pfn.append(nFalseNegative / nTrue)
    Pfp.append(nFalsePositive / nFalse)
    
    for idx in range(len(llrSorted)):
        if classLabelsSorted[idx] == 1:
            nFalseNegative += 1
        if classLabelsSorted[idx] == 0:
            nFalsePositive -= 1
        Pfn.append(nFalseNegative / nTrue)
        Pfp.append(nFalsePositive / nFalse)

    llrSorted = np.concatenate([-np.array([np.inf]), llrSorted])

    PfnOut = []
    PfpOut = []
    thresholdsOut = []
    for idx in range(len(llrSorted)):
        if idx == len(llrSorted) - 1 or llrSorted[idx+1] != llrSorted[idx]:
            PfnOut.append(Pfn[idx])
            PfpOut.append(Pfp[idx])
            thresholdsOut.append(llrSorted[idx])
            
    return np.array(PfnOut), np.array(PfpOut), np.array(thresholdsOut)

def compute_minDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):
    Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds_fast(llr, classLabels)
    minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / np.minimum(prior * Cfn, (1-prior)*Cfp)
    idx = np.argmin(minDCF)
    if returnThreshold:
        return minDCF[idx], th[idx]
    else:
        return minDCF[idx]

# ---------------------------------------------------------------------
# ---------------------  PROJECT DATA LOADING  -----------------------
# ---------------------------------------------------------------------

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

def split_2to1(D, L, seed=0):
    """Split data 2/3 training, 1/3 validation."""
    n_tr = int(D.shape[1] * 2 / 3)
    np.random.seed(seed)
    perm = np.random.permutation(D.shape[1])
    idx_tr, idx_ev = perm[:n_tr], perm[n_tr:]
    return (D[:, idx_tr], L[idx_tr]), (D[:, idx_ev], L[idx_ev])

# ---------------------------------------------------------------------
# ---------------------  GAUSSIAN CLASSIFIERS  -----------------------
# ---------------------------------------------------------------------

def mean_vector(X):
    return vcol(X.mean(axis=1))

def covariance_matrix(X):
    centered = X - mean_vector(X)
    return centered @ centered.T / X.shape[1]

def diagonal_covariance_matrix(X):
    cov = covariance_matrix(X)
    return np.diag(np.diag(cov))

def logpdf_GAU_ND(X, mu, Sigma):
    M = X.shape[0]
    XC = X - mu
    invS = np.linalg.inv(Sigma)
    log_det = np.linalg.slogdet(Sigma)[1]
    quad = np.sum(XC * (invS @ XC), axis=0)
    return -0.5 * (M * np.log(2*np.pi) + log_det + quad)

class GaussianClassifier:
    """Gaussian classifier for binary classification."""
    
    def __init__(self, classifier_type="mvg"):
        self.classifier_type = classifier_type
        self.mus = None
        self.covs = None
        self.Sigma_tied = None
        
    def train(self, D_tr, L_tr):
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
    
    def compute_llr(self, D_ev):
        """Compute Log-Likelihood Ratios (LLR) with class 1 on top."""
        if self.classifier_type == "tied":
            log_lik_1 = logpdf_GAU_ND(D_ev, self.mus[1], self.Sigma_tied)
            log_lik_0 = logpdf_GAU_ND(D_ev, self.mus[0], self.Sigma_tied)
        else:
            log_lik_1 = logpdf_GAU_ND(D_ev, self.mus[1], self.covs[1])
            log_lik_0 = logpdf_GAU_ND(D_ev, self.mus[0], self.covs[0])
        
        return log_lik_1 - log_lik_0

# ---------------------------------------------------------------------
# ---------------------  EFFECTIVE PRIOR ANALYSIS  -------------------
# ---------------------------------------------------------------------

def compute_effective_prior(prior, Cfn, Cfp):
    """Compute effective prior from application parameters."""
    return (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)

def analyze_applications():
    """Analyze the five given applications."""
    applications = [
        (0.5, 1.0, 1.0, "Uniform prior and costs"),
        (0.9, 1.0, 1.0, "High prior for genuine (most users legit)"),
        (0.1, 1.0, 1.0, "High prior for fake (most users impostors)"),
        (0.5, 1.0, 9.0, "Strong security (high FP cost)"),
        (0.5, 9.0, 1.0, "Ease of use (high FN cost)")
    ]
    
    print("🎯 ANALISI APPLICAZIONI:")
    print("=" * 80)
    print(f"{'Application':<40} {'π':<6} {'Cfn':<6} {'Cfp':<6} {'π̃eff':<8}")
    print("-" * 80)
    
    effective_priors = []
    for prior, Cfn, Cfp, desc in applications:
        eff_prior = compute_effective_prior(prior, Cfn, Cfp)
        effective_priors.append(eff_prior)
        print(f"{desc:<40} {prior:<6.1f} {Cfn:<6.1f} {Cfp:<6.1f} {eff_prior:<8.3f}")
    
    print(f"\n💡 OSSERVAZIONI:")
    print(f"- Sicurezza forte (Cfp alto) → π̃eff basso ({effective_priors[3]:.3f})")
    print(f"- Facilità d'uso (Cfn alto) → π̃eff alto ({effective_priors[4]:.3f})")
    print(f"- I costi si riflettono nell'effective prior equivalente")
    
    return effective_priors

# ---------------------------------------------------------------------
# ---------------------  MAIN ANALYSIS  ------------------------------
# ---------------------------------------------------------------------

def main():
    print("=" * 80)
    print("    ANALISI BAYES DECISION - PROGETTO FINGERPRINT")
    print("=" * 80)
    
    # Load and split data
    print("\n🔄 Caricamento dataset...")
    D, L = load_project_data("trainData.csv")
    (D_tr, L_tr), (D_ev, L_ev) = split_2to1(D, L, seed=0)
    
    print(f"Dataset: {D.shape[1]} campioni, {D.shape[0]} features")
    print(f"Training: {D_tr.shape[1]} | Validation: {D_ev.shape[1]}")
    print(f"Classe 0 (Fake): {np.sum(L_ev == 0)} | Classe 1 (Genuine): {np.sum(L_ev == 1)}")
    
    # Analyze applications
    print(f"\n" + "="*80)
    print("PARTE 1: ANALISI APPLICAZIONI E EFFECTIVE PRIOR")
    print("="*80)
    
    effective_priors = analyze_applications()
    
    # Train classifiers
    print(f"\n🔄 Training classificatori...")
    classifiers = {
        "MVG": GaussianClassifier("mvg"),
        "Tied": GaussianClassifier("tied"),
        "Naive Bayes": GaussianClassifier("naive_bayes")
    }
    
    llrs = {}
    for name, clf in classifiers.items():
        clf.train(D_tr, L_tr)
        llrs[name] = clf.compute_llr(D_ev)
        print(f"  ✅ {name} addestrato")
    
    # Focus on three main applications
    focus_applications = [
        (0.1, 1.0, 1.0, "π̃ = 0.1 (Security-focused)"),
        (0.5, 1.0, 1.0, "π̃ = 0.5 (Balanced)"),
        (0.9, 1.0, 1.0, "π̃ = 0.9 (User-friendly)")
    ]
    
    print(f"\n" + "="*80)
    print("PARTE 2: ANALISI DETTAGLIATA - TRE APPLICAZIONI PRINCIPALI")
    print("="*80)
    
    results = {}
    
    for prior, Cfn, Cfp, desc in focus_applications:
        print(f"\n🎯 {desc}")
        print("-" * 60)
        
        app_results = {}
        
        for clf_name, llr in llrs.items():
            # Compute actDCF and minDCF
            actDCF = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(
                llr, L_ev, prior, Cfn, Cfp, normalize=True)
            minDCF = compute_minDCF_binary_fast(llr, L_ev, prior, Cfn, Cfp)
            
            # Calibration loss
            calibration_loss = actDCF - minDCF
            calibration_loss_pct = (calibration_loss / minDCF) * 100 if minDCF > 0 else 0
            
            app_results[clf_name] = {
                'actDCF': actDCF,
                'minDCF': minDCF,
                'calibration_loss': calibration_loss,
                'calibration_loss_pct': calibration_loss_pct
            }
            
            print(f"{clf_name:<12}: actDCF = {actDCF:.3f} | minDCF = {minDCF:.3f} | "
                  f"Calib. Loss = {calibration_loss:.3f} ({calibration_loss_pct:.1f}%)")
        
        results[desc] = app_results
        
        # Best model analysis
        best_minDCF = min(app_results.values(), key=lambda x: x['minDCF'])
        best_model_minDCF = [k for k, v in app_results.items() if v['minDCF'] == best_minDCF['minDCF']][0]
        
        best_calibrated = min(app_results.values(), key=lambda x: x['calibration_loss_pct'])
        best_model_calib = [k for k, v in app_results.items() if v['calibration_loss_pct'] == best_calibrated['calibration_loss_pct']][0]
        
        print(f"\n📊 Migliore per minDCF: {best_model_minDCF} ({best_minDCF['minDCF']:.3f})")
        print(f"📊 Meglio calibrato: {best_model_calib} ({best_calibrated['calibration_loss_pct']:.1f}% loss)")
    
    # Model ranking consistency analysis
    print(f"\n" + "="*80)
    print("PARTE 3: CONSISTENZA RANKING MODELLI")
    print("="*80)
    
    print(f"\n📈 Ranking per minDCF:")
    for desc in results.keys():
        sorted_models = sorted(results[desc].items(), key=lambda x: x[1]['minDCF'])
        ranking = " > ".join([f"{name} ({res['minDCF']:.3f})" for name, res in sorted_models])
        print(f"{desc:<25}: {ranking}")
    
    print(f"\n🎯 Calibrazione (loss percentuale):")
    for desc in results.keys():
        sorted_models = sorted(results[desc].items(), key=lambda x: x[1]['calibration_loss_pct'])
        ranking = " > ".join([f"{name} ({res['calibration_loss_pct']:.1f}%)" for name, res in sorted_models])
        print(f"{desc:<25}: {ranking}")
    
    # Bayes error plots
    print(f"\n" + "="*80)
    print("PARTE 4: BAYES ERROR PLOTS")
    print("="*80)
    
    # Prior log odds range (-4, +4)
    effPriorLogOdds = np.linspace(-4, 4, 41)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))
    
    plt.figure(figsize=(15, 5))
    
    for i, (clf_name, llr) in enumerate(llrs.items()):
        plt.subplot(1, 3, i+1)
        
        actDCFs = []
        minDCFs = []
        
        for effPrior in effPriors:
            actDCF = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(
                llr, L_ev, effPrior, 1.0, 1.0, normalize=True)
            minDCF = compute_minDCF_binary_fast(llr, L_ev, effPrior, 1.0, 1.0)
            
            actDCFs.append(actDCF)
            minDCFs.append(minDCF)
        
        plt.plot(effPriorLogOdds, actDCFs, 'r-', label=f'actDCF {clf_name}', linewidth=2)
        plt.plot(effPriorLogOdds, minDCFs, 'b-', label=f'minDCF {clf_name}', linewidth=2)
        
        # Mark the three focus applications
        for prior, _, _, desc in focus_applications:
            log_odds = np.log(prior / (1 - prior))
            if -4 <= log_odds <= 4:
                idx = np.argmin(np.abs(effPriorLogOdds - log_odds))
                plt.plot(log_odds, actDCFs[idx], 'ro', markersize=8)
                plt.plot(log_odds, minDCFs[idx], 'bo', markersize=8)
        
        plt.xlabel('Prior Log-Odds')
        plt.ylabel('DCF')
        plt.title(f'{clf_name} Classifier')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig('bayes_error_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary and interpretation
    print(f"\n" + "="*80)
    print("INTERPRETAZIONE E CONCLUSIONI")
    print("="*80)
    
    print(f"\n💡 OSSERVAZIONI PRINCIPALI:")
    print("-" * 50)
    
    # Check ranking consistency
    rankings = {}
    for desc in results.keys():
        ranking = [name for name, _ in sorted(results[desc].items(), key=lambda x: x[1]['minDCF'])]
        rankings[desc] = ranking
    
    all_rankings = list(rankings.values())
    consistent = all(ranking == all_rankings[0] for ranking in all_rankings)
    
    print(f"1. CONSISTENZA RANKING:")
    if consistent:
        print(f"   ✅ Il ranking dei modelli è CONSISTENTE tra le applicazioni")
        print(f"   📊 Ordine: {' > '.join(all_rankings[0])}")
    else:
        print(f"   ⚠️  Il ranking dei modelli VARIA tra le applicazioni")
        for desc, ranking in rankings.items():
            print(f"   {desc}: {' > '.join(ranking)}")
    
    print(f"\n2. CALIBRAZIONE:")
    well_calibrated_threshold = 10  # 10% threshold for good calibration
    
    for desc in results.keys():
        print(f"\n   {desc}:")
        for clf_name, res in results[desc].items():
            status = "✅ Ben calibrato" if res['calibration_loss_pct'] < well_calibrated_threshold else "⚠️ Mal calibrato"
            print(f"     {clf_name:<12}: {res['calibration_loss_pct']:>5.1f}% loss - {status}")
    
    print(f"\n3. PERFORMANCE GENERALE:")
    # Find overall best model
    avg_minDCF = {}
    for clf_name in classifiers.keys():
        avg_minDCF[clf_name] = np.mean([results[desc][clf_name]['minDCF'] for desc in results.keys()])
    
    best_overall = min(avg_minDCF.items(), key=lambda x: x[1])
    
    print(f"   🏆 Migliore modello complessivo: {best_overall[0]} (minDCF medio: {best_overall[1]:.3f})")
    
    # Calibration analysis
    avg_calib_loss = {}
    for clf_name in classifiers.keys():
        avg_calib_loss[clf_name] = np.mean([results[desc][clf_name]['calibration_loss_pct'] for desc in results.keys()])
    
    best_calibrated_overall = min(avg_calib_loss.items(), key=lambda x: x[1])
    print(f"   🎯 Meglio calibrato: {best_calibrated_overall[0]} (loss medio: {best_calibrated_overall[1]:.1f}%)")
    
    print(f"\n4. IMPLICAZIONI PRATICHE:")
    print(f"   - Per applicazioni security-critical: privilegiare minDCF basso")
    print(f"   - Per sistemi real-time: considerare anche la calibrazione")
    print(f"   - Il modello {best_overall[0]} offre il miglior compromesso generale")
    
    print(f"\n" + "="*80)

if __name__ == "__main__":
    main()