import numpy
import bayesRisk
import logReg
import matplotlib
import matplotlib.pyplot as plt

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def bayesPlot(S, L, left = -3, right = 3, npts = 21):
    
    effPriorLogOdds = numpy.linspace(left, right, npts)
    effPriors = 1.0 / (1.0 + numpy.exp(-effPriorLogOdds))
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        actDCF.append(bayesRisk.compute_actDCF_binary_fast(S, L, effPrior, 1.0, 1.0))
        minDCF.append(bayesRisk.compute_minDCF_binary_fast(S, L, effPrior, 1.0, 1.0))
    return effPriorLogOdds, actDCF, minDCF

# Extract i-th fold from a 1-D numpy array (as for the single fold case, we do not need to shuffle scores in this case, but it may be necessary if samples are sorted in peculiar ways to ensure that validation and calibration sets are independent and with similar characteristics   
def extract_train_val_folds_from_ary(X, idx):
    return numpy.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != idx]), X[idx::KFOLD]

if __name__ == '__main__':

    SAMEFIGPLOTS = True # set to False to have 1 figure per plot
    
    scores_sys_1 = numpy.load('Data/scores_1.npy')
    scores_sys_2 = numpy.load('Data/scores_2.npy')
    eval_scores_sys_1 = numpy.load('Data/eval_scores_1.npy')
    eval_scores_sys_2 = numpy.load('Data/eval_scores_2.npy')
    labels = numpy.load('Data/labels.npy')
    eval_labels = numpy.load('Data/eval_labels.npy')
    
    if SAMEFIGPLOTS:
        fig = plt.figure(figsize=(16,9))
        axes = fig.subplots(3,3, sharex='all')
        fig.suptitle('K-fold')
    else:
        axes = numpy.array([ [plt.figure().gca(), plt.figure().gca(), plt.figure().gca()], [plt.figure().gca(), plt.figure().gca(), plt.figure().gca()], [None, plt.figure().gca(), plt.figure().gca()] ])

    print()
    print('*** K-FOLD ***')
    print()
    
    KFOLD = 5

    ###
    #
    # K-fold version
    #
    # Note: minDCF of calibrated folds may change with respect to the one we computed at the beginning over the whole dataset, since we are pooling scores of different folds that have undergone a different affine transformation
    #
    # Pay attention that, for fusion and model comparison, we need the folds to be the same across the two systems
    #
    # We use K = 5 (KFOLD variable)
    #
    ###

    # We start with the computation of the system performance on the calibration set (whole dataset)
    print('Sys1: minDCF (0.2) = %.3f - actDCF (0.2) = %.3f' % (
        bayesRisk.compute_minDCF_binary_fast(scores_sys_1, labels, 0.2, 1.0, 1.0),
        bayesRisk.compute_actDCF_binary_fast(scores_sys_1, labels, 0.2, 1.0, 1.0)))

    print('Sys2: minDCF (0.2) = %.3f - actDCF (0.2) = %.3f' % (
        bayesRisk.compute_minDCF_binary_fast(scores_sys_2, labels, 0.2, 1.0, 1.0),
        bayesRisk.compute_actDCF_binary_fast(scores_sys_2, labels, 0.2, 1.0, 1.0)))

    # Comparison of actDCF / minDCF of both systems
    logOdds, actDCF, minDCF = bayesPlot(scores_sys_1, labels)
    axes[0,0].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'minDCF')
    axes[0,0].plot(logOdds, actDCF, color='C0', linestyle='-', label = 'actDCF')

    logOdds, actDCF, minDCF = bayesPlot(scores_sys_2, labels)
    axes[1,0].plot(logOdds, minDCF, color='C1', linestyle='--', label = 'minDCF')
    axes[1,0].plot(logOdds, actDCF, color='C1', linestyle='-', label = 'actDCF')
    
    axes[0,0].set_ylim(0, 0.8)    
    axes[0,0].legend()

    axes[1,0].set_ylim(0, 0.8)    
    axes[1,0].legend()
    
    axes[0,0].set_title('System 1 - validation - non-calibrated scores')
    axes[1,0].set_title('System 2 - validation - non-calibrated scores')
    
    # We calibrate both systems (independently)

    # System 1
    calibrated_scores_sys_1 = [] # We will add to the list the scores computed for each fold
    labels_sys_1 = [] # We need to ensure that we keep the labels aligned with the scores. The simplest thing to do is to just extract each fold label and pool all the fold labels together in the same order as we pool the corresponding scores.

    # We plot the non-calibrated minDCF and actDCF for reference
    logOdds, actDCF, minDCF = bayesPlot(scores_sys_1, labels)
    axes[0,1].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'minDCF (pre-cal.)')
    axes[0,1].plot(logOdds, actDCF, color='C0', linestyle=':', label = 'actDCF (pre-cal.)')
    print ('System 1')
    print ('\tValidation set')
    print ('\t\tminDCF(p=0.2), no cal.: %.3f' % bayesRisk.compute_minDCF_binary_fast(scores_sys_1, labels, 0.2, 1.0, 1.0)) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
    print ('\t\tactDCF(p=0.2), no cal.: %.3f' % bayesRisk.compute_actDCF_binary_fast(scores_sys_1, labels, 0.2, 1.0, 1.0))
    
    # We train the calibration model for the prior pT = 0.2
    pT = 0.2
    # Train KFOLD times the calibration model
    for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training
        SCAL, SVAL = extract_train_val_folds_from_ary(scores_sys_1, foldIdx)
        LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx)
        # Train the model on the KFOLD - 1 training folds
        w, b = logReg.trainWeightedLogRegBinary(vrow(SCAL), LCAL, 0, pT)
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ vrow(SVAL) + b - numpy.log(pT / (1-pT))).ravel()
        # Add the scores of this validation fold to the cores list
        calibrated_scores_sys_1.append(calibrated_SVAL)
        # Add the corresponding labels to preserve alignment between scores and labels
        labels_sys_1.append(LVAL)

    # Re-build the score and label arrays (pooling) - these contains an entry for every element in the original dataset (but the order of the samples is different)
    calibrated_scores_sys_1 = numpy.hstack(calibrated_scores_sys_1)
    labels_sys_1 = numpy.hstack(labels_sys_1)

    # Evaluate the performance on pooled scores - we need to use the label vector labels_sys_1 since it's aligned to calibrated_scores_sys_1    
    print ('\t\tminDCF(p=0.2), cal.   : %.3f' % bayesRisk.compute_minDCF_binary_fast(calibrated_scores_sys_1, labels_sys_1, 0.2, 1.0, 1.0)) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
    print ('\t\tactDCF(p=0.2), cal.   : %.3f' % bayesRisk.compute_actDCF_binary_fast(calibrated_scores_sys_1, labels_sys_1, 0.2, 1.0, 1.0))
    
    logOdds, actDCF, minDCF = bayesPlot(calibrated_scores_sys_1, labels_sys_1)
    axes[0,1].plot(logOdds, actDCF, color='C0', linestyle='-', label = 'actDCF (cal.)') # NOTE: actDCF of the calibrated pooled scores MAY be lower than the global minDCF we computed earlier, since ache fold is calibrated on its own (thus it's as if we were estimating a possibly different threshold for each fold, whereas minDCF employs a single threshold for all scores)
    axes[0,1].legend()

    axes[0,1].set_title('System 1 - validation')
    axes[0,1].set_ylim(0, 0.8)    
    
    # For K-fold the final model is a new model re-trained over the whole set, using the optimal hyperparameters we selected during the k-fold procedure (in this case we have no hyperparameter, so we simply train a new model on the whole dataset)

    w, b = logReg.trainWeightedLogRegBinary(vrow(scores_sys_1), labels, 0, pT)

    # We can use the trained model for application / evaluation data
    calibrated_eval_scores_sys_1 = (w.T @ vrow(eval_scores_sys_1) + b - numpy.log(pT / (1-pT))).ravel()

    print ('\tEvaluation set')
    print ('\t\tminDCF(p=0.2)         : %.3f' % bayesRisk.compute_minDCF_binary_fast(eval_scores_sys_1, eval_labels, 0.2, 1.0, 1.0))
    print ('\t\tactDCF(p=0.2), no cal.: %.3f' % bayesRisk.compute_actDCF_binary_fast(eval_scores_sys_1, eval_labels, 0.2, 1.0, 1.0))
    print ('\t\tactDCF(p=0.2), cal.   : %.3f' % bayesRisk.compute_actDCF_binary_fast(calibrated_eval_scores_sys_1, eval_labels, 0.2, 1.0, 1.0))    
    
    # We plot minDCF, non-calibrated DCF and calibrated DCF for system 1
    logOdds, actDCF_precal, minDCF = bayesPlot(eval_scores_sys_1, eval_labels)
    logOdds, actDCF_cal, _ = bayesPlot(calibrated_eval_scores_sys_1, eval_labels) # minDCF is the same
    axes[0,2].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'minDCF')
    axes[0,2].plot(logOdds, actDCF_precal, color='C0', linestyle=':', label = 'actDCF (pre-cal.)')
    axes[0,2].plot(logOdds, actDCF_cal, color='C0', linestyle='-', label = 'actDCF (cal.)')
    axes[0,2].set_ylim(0.0, 0.8)
    axes[0,2].set_title('System 1 - evaluation')
    axes[0,2].legend()
    

    
    # System 2
    calibrated_scores_sys_2 = [] # We will add to the list the scores computed for each fold
    labels_sys_2 = [] # We need to ensure that we keep the labels aligned with the scores. The simplest thing to do is to just extract each fold label and pool all the fold labels together in the same order as we pool the corresponding scores.

    # We plot the non-calibrated minDCF and actDCF for reference
    logOdds, actDCF, minDCF = bayesPlot(scores_sys_2, labels)
    axes[1,1].plot(logOdds, minDCF, color='C1', linestyle='--', label = 'minDCF (pre-cal.)')
    axes[1,1].plot(logOdds, actDCF, color='C1', linestyle=':', label = 'actDCF (pre-cal.)')
    print ('System 2')
    print ('\tValidation set')
    print ('\t\tminDCF(p=0.2), no cal.: %.3f' % bayesRisk.compute_minDCF_binary_fast(scores_sys_2, labels, 0.2, 1.0, 1.0)) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
    print ('\t\tactDCF(p=0.2), no cal.: %.3f' % bayesRisk.compute_actDCF_binary_fast(scores_sys_2, labels, 0.2, 1.0, 1.0))
    
    # We train the calibration model for the prior pT = 0.2
    pT = 0.2
    # Train KFOLD times the calibration model
    for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training
        SCAL, SVAL = extract_train_val_folds_from_ary(scores_sys_2, foldIdx)
        LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx)
        # Train the model on the KFOLD - 1 training folds
        w, b = logReg.trainWeightedLogRegBinary(vrow(SCAL), LCAL, 0, pT)
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ vrow(SVAL) + b - numpy.log(pT / (1-pT))).ravel()
        # Add the scores of this validation fold to the cores list
        calibrated_scores_sys_2.append(calibrated_SVAL)
        # Add the corresponding labels to preserve alignment between scores and labels
        labels_sys_2.append(LVAL)

    # Re-build the score and label arrays (pooling) - these contains an entry for every element in the original dataset (but the order of the samples is different)
    calibrated_scores_sys_2 = numpy.hstack(calibrated_scores_sys_2)
    labels_sys_2 = numpy.hstack(labels_sys_2)

    # Evaluate the performance on pooled scores - we need to use the label vector labels_sys_2 since it's aligned to calibrated_scores_sys_2    
    print ('\t\tminDCF(p=0.2), cal.   : %.3f' % bayesRisk.compute_minDCF_binary_fast(calibrated_scores_sys_2, labels_sys_2, 0.2, 1.0, 1.0)) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
    print ('\t\tactDCF(p=0.2), cal.   : %.3f' % bayesRisk.compute_actDCF_binary_fast(calibrated_scores_sys_2, labels_sys_2, 0.2, 1.0, 1.0))
    
    logOdds, actDCF, minDCF = bayesPlot(calibrated_scores_sys_2, labels_sys_2)
    axes[1,1].plot(logOdds, actDCF, color='C1', linestyle='-', label = 'actDCF (cal.)') # NOTE: actDCF of the calibrated pooled scores MAY be lower than the global minDCF we computed earlier, since ache fold is calibrated on its own (thus it's as if we were estimating a possibly different threshold for each fold, whereas minDCF employs a single threshold for all scores)
    axes[1,1].legend()

    axes[1,1].set_ylim(0, 0.8)            
    axes[1,1].set_title('System 2 - validation')
    
    # For K-fold the final model is a new model re-trained over the whole set, using the optimal hyperparameters we selected during the k-fold procedure (in this case we have no hyperparameter, so we simply train a new model on the whole dataset)

    w, b = logReg.trainWeightedLogRegBinary(vrow(scores_sys_2), labels, 0, pT)

    # We can use the trained model for application / evaluation data
    calibrated_eval_scores_sys_2 = (w.T @ vrow(eval_scores_sys_2) + b - numpy.log(pT / (1-pT))).ravel()

    print ('\tEvaluation set')
    print ('\t\tminDCF(p=0.2)         : %.3f' % bayesRisk.compute_minDCF_binary_fast(eval_scores_sys_2, eval_labels, 0.2, 1.0, 1.0))
    print ('\t\tactDCF(p=0.2), no cal.: %.3f' % bayesRisk.compute_actDCF_binary_fast(eval_scores_sys_2, eval_labels, 0.2, 1.0, 1.0))
    print ('\t\tactDCF(p=0.2), cal.   : %.3f' % bayesRisk.compute_actDCF_binary_fast(calibrated_eval_scores_sys_2, eval_labels, 0.2, 1.0, 1.0))    
    
    # We plot minDCF, non-calibrated DCF and calibrated DCF for system 1
    logOdds, actDCF_precal, minDCF = bayesPlot(eval_scores_sys_2, eval_labels)
    logOdds, actDCF_cal, _ = bayesPlot(calibrated_eval_scores_sys_2, eval_labels) # minDCF is the same
    axes[1,2].plot(logOdds, minDCF, color='C1', linestyle='--', label = 'minDCF')
    axes[1,2].plot(logOdds, actDCF_precal, color='C1', linestyle=':', label = 'actDCF (pre-cal.)')
    axes[1,2].plot(logOdds, actDCF_cal, color='C1', linestyle='-', label = 'actDCF (cal.)')
    axes[1,2].set_ylim(0.0, 0.8)
    axes[1,2].set_title('System 2 - evaluation')
    axes[1,2].legend()


    # Fusion #
    
    fusedScores = [] # We will add to the list the scores computed for each fold
    fusedLabels = [] # We need to ensure that we keep the labels aligned with the scores. The simplest thing to do is to just extract each fold label and pool all the fold labels together in the same order as we pool the corresponding scores.
    
    # We train the fusion for the prior pT = 0.2
    pT = 0.2
    
    # Train KFOLD times the fusion model
    for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training        
        SCAL1, SVAL1 = extract_train_val_folds_from_ary(scores_sys_1, foldIdx)
        SCAL2, SVAL2 = extract_train_val_folds_from_ary(scores_sys_2, foldIdx)
        LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx)
        # Build the training scores "feature" matrix
        SCAL = numpy.vstack([SCAL1, SCAL2])
        # Train the model on the KFOLD - 1 training folds
        w, b = logReg.trainWeightedLogRegBinary(SCAL, LCAL, 0, pT)
        # Build the validation scores "feature" matrix
        SVAL = numpy.vstack([SVAL1, SVAL2])
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ SVAL + b - numpy.log(pT / (1-pT))).ravel()
        # Add the scores of this validation fold to the cores list
        fusedScores.append(calibrated_SVAL)
        # Add the corresponding labels to preserve alignment between scores and labels
        fusedLabels.append(LVAL)

    # Re-build the score and label arrays (pooling) - these contains an entry for every element in the original dataset (but the order of the samples is different)        
    fusedScores = numpy.hstack(fusedScores)
    fusedLabels = numpy.hstack(fusedLabels)

    # Evaluate the performance on pooled scores - we need to use the label vector fusedLabels since it's aligned to calScores_sys_2 (plot on same figure as system 1 and system 2)

    print ('Fusion')
    print ('\tValidation set')
    print ('\t\tminDCF(p=0.2)         : %.3f' % bayesRisk.compute_minDCF_binary_fast(fusedScores, fusedLabels, 0.2, 1.0, 1.0)) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
    print ('\t\tactDCF(p=0.2)         : %.3f' % bayesRisk.compute_actDCF_binary_fast(fusedScores, fusedLabels, 0.2, 1.0, 1.0))

    # As comparison, we select calibrated models trained with prior 0.2 (our target application)
    logOdds, actDCF, minDCF = bayesPlot(calibrated_scores_sys_1, labels_sys_1)
    axes[2,1].set_title('Fusion - validation')
    axes[2,1].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'S1 - minDCF')
    axes[2,1].plot(logOdds, actDCF, color='C0', linestyle='-', label = 'S1 - actDCF')
    logOdds, actDCF, minDCF = bayesPlot(calibrated_scores_sys_2, labels_sys_2)
    axes[2,1].plot(logOdds, minDCF, color='C1', linestyle='--', label = 'S2 - minDCF')
    axes[2,1].plot(logOdds, actDCF, color='C1', linestyle='-', label = 'S2 - actDCF')    
    
    logOdds, actDCF, minDCF = bayesPlot(fusedScores, fusedLabels)
    axes[2,1].plot(logOdds, minDCF, color='C2', linestyle='--', label = 'S1 + S2 - KFold - minDCF(0.2)')
    axes[2,1].plot(logOdds, actDCF, color='C2', linestyle='-', label = 'S1 + S2 - KFold - actDCF(0.2)')
    axes[2,1].set_ylim(0.0, 0.8)
    axes[2,1].legend()

    # For K-fold the final model is a new model re-trained over the whole set, using the optimal hyperparameters we selected during the k-fold procedure (in this case we have no hyperparameter, so we simply train a new model on the whole dataset)

    SMatrix = numpy.vstack([scores_sys_1, scores_sys_2])
    w, b = logReg.trainWeightedLogRegBinary(SMatrix, labels, 0, pT)

    # Apply model to application / evaluation data
    SMatrixEval = numpy.vstack([eval_scores_sys_1, eval_scores_sys_2])
    fused_eval_scores = (w.T @ SMatrixEval + b - numpy.log(pT / (1-pT))).ravel()

    print ('\tEvaluation set')
    print ('\t\tminDCF(p=0.2)         : %.3f' % bayesRisk.compute_minDCF_binary_fast(fused_eval_scores, eval_labels, 0.2, 1.0, 1.0))
    print ('\t\tactDCF(p=0.2)         : %.3f' % bayesRisk.compute_actDCF_binary_fast(fused_eval_scores, eval_labels, 0.2, 1.0, 1.0))
    
    # We plot minDCF, actDCF for calibrated system 1, calibrated system 2 and fusion
    logOdds, actDCF, minDCF = bayesPlot(calibrated_eval_scores_sys_1, eval_labels)
    axes[2,2].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'S1 - minDCF')
    axes[2,2].plot(logOdds, actDCF, color='C0', linestyle='-', label = 'S1 - actDCF')
    logOdds, actDCF, minDCF = bayesPlot(calibrated_eval_scores_sys_2, eval_labels)
    axes[2,2].plot(logOdds, minDCF, color='C1', linestyle='--', label = 'S2 - minDCF')
    axes[2,2].plot(logOdds, actDCF, color='C1', linestyle='-', label = 'S2 - actDCF')
    
    logOdds, actDCF, minDCF = bayesPlot(fused_eval_scores, eval_labels) # minDCF is the same
    axes[2,2].plot(logOdds, minDCF, color='C2', linestyle='--', label = 'S1 + S2 - minDCF')
    axes[2,2].plot(logOdds, actDCF, color='C2', linestyle='-', label = 'S1 + S2 - actDCF')
    axes[2,2].set_ylim(0.0, 0.8)
    axes[2,2].set_title('Fusion - evaluation')
    axes[2,2].legend()
    
    plt.show()
    
    
    
    
    
