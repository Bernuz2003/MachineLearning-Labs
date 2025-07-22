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

if __name__ == '__main__':

    SAMEFIGPLOTS = True # set to False to have 1 figure per plot
    
    scores_sys_1 = numpy.load('Data/scores_1.npy')
    scores_sys_2 = numpy.load('Data/scores_2.npy')
    eval_scores_sys_1 = numpy.load('Data/eval_scores_1.npy')
    eval_scores_sys_2 = numpy.load('Data/eval_scores_2.npy')
    labels = numpy.load('Data/labels.npy')
    eval_labels = numpy.load('Data/eval_labels.npy')
    
    #####
    #
    # SINGLE-FOLD SPLIT
    #
    # We divide the dataset (former validation set) in calibration training and calibration validation sets
    #
    # Results cannot be directly compared with those we computed earlier because we are using different validation sets
    #
    # Pay attention that, for fusion and model comparison, we need the splits to be the same across the two systems
    #
    # We take 1/3 scores for calibration training and 2/3 for validation. We can simply take every 3rd score for training and the reamining scores for validation (in general, we should make sure that the calibration and validation scores are independent and have similar characteristics, so random shuffles of the scores may be necessary)
    #
    #####

    if SAMEFIGPLOTS:
        fig = plt.figure(figsize=(16,9))
        axes = fig.subplots(3,3, sharex='all')
        axes[2,0].axis('off')
        fig.suptitle('Single fold')
    else:
        axes = numpy.array([ [plt.figure().gca(), plt.figure().gca(), plt.figure().gca()], [plt.figure().gca(), plt.figure().gca(), plt.figure().gca()], [None, plt.figure().gca(), plt.figure().gca()] ])


    print()
    print('*** Single-fold approach ***')
    print()
    
    SCAL1, SVAL1 = scores_sys_1[::3], numpy.hstack([scores_sys_1[1::3], scores_sys_1[2::3]]) # System 1
    SCAL2, SVAL2 = scores_sys_2[::3], numpy.hstack([scores_sys_2[1::3], scores_sys_2[2::3]]) # System 2
    LCAL, LVAL = labels[::3], numpy.hstack([labels[1::3], labels[2::3]]) # Labels
    
    # We recompute the system performance on the calibration validation sets (SVAL1 and SVAL2)
    print('Sys1: minDCF (0.2) = %.3f - actDCF (0.2) = %.3f' % (
        bayesRisk.compute_minDCF_binary_fast(SVAL1, LVAL, 0.2, 1.0, 1.0),
        bayesRisk.compute_actDCF_binary_fast(SVAL1, LVAL, 0.2, 1.0, 1.0)))

    print('Sys2: minDCF (0.2) = %.3f - actDCF (0.2) = %.3f' % (
        bayesRisk.compute_minDCF_binary_fast(SVAL2, LVAL, 0.2, 1.0, 1.0),
        bayesRisk.compute_actDCF_binary_fast(SVAL2, LVAL, 0.2, 1.0, 1.0)))

    # Comparison of actDCF / minDCF of both systems
    logOdds, actDCF, minDCF = bayesPlot(SVAL1, LVAL)
    axes[0,0].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'minDCF')
    axes[0,0].plot(logOdds, actDCF, color='C0', linestyle='-', label = 'actDCF')

    logOdds, actDCF, minDCF = bayesPlot(SVAL2, LVAL)
    axes[1,0].plot(logOdds, minDCF, color='C1', linestyle='--', label = 'minDCF')
    axes[1,0].plot(logOdds, actDCF, color='C1', linestyle='-', label = 'actDCF')
    
    axes[0,0].set_ylim(0, 0.8)    
    axes[0,0].legend()

    axes[1,0].set_ylim(0, 0.8)    
    axes[1,0].legend()
    
    axes[0,0].set_title('System 1 - calibration validation - non-calibrated scores')
    axes[1,0].set_title('System 2 - calibration validation - non-calibrated scores')
    

    # We calibrate both systems (independently)
    
    # System 1
    print()

    # We plot the non-calibrated minDCF and actDCF for reference
    logOdds, actDCF, minDCF = bayesPlot(SVAL1, LVAL)
    axes[0,1].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'minDCF')
    axes[0,1].plot(logOdds, actDCF, color='C0', linestyle=':', label = 'actDCF (pre-cal.)')
    print ('System 1')
    print ('\tValidation set')
    print ('\t\tminDCF(p=0.2)         : %.3f' % bayesRisk.compute_minDCF_binary_fast(SVAL1, LVAL, 0.2, 1.0, 1.0))
    print ('\t\tactDCF(p=0.2), no cal.: %.3f' % bayesRisk.compute_actDCF_binary_fast(SVAL1, LVAL, 0.2, 1.0, 1.0))

    # We calibrate for target prior pT = 0.2
    pT = 0.2
    # Logistic regression calibration, trained on calibration training set
    w, b = logReg.trainWeightedLogRegBinary(vrow(SCAL1), LCAL, 0, pT)
    # Apply the logistic regression model to the calibration validation set
    calibrated_SVAL1 = (w.T @ vrow(SVAL1) + b - numpy.log(pT / (1-pT))).ravel()
    # Evaluate the calibrated scores
    print ('\t\tactDCF(p=0.2), cal.   : %.3f' % (bayesRisk.compute_actDCF_binary_fast(calibrated_SVAL1, LVAL, 0.2, 1.0, 1.0)))
    
    logOdds, actDCF, _ = bayesPlot(calibrated_SVAL1, LVAL)
    axes[0,1].plot(logOdds, actDCF, label = 'actDCF (cal.)')
    axes[0,1].set_ylim(0.0, 0.8)
    axes[0,1].set_title('System 1 - calibration validation')
    axes[0,1].legend()
        
    # The final model for single-fold systems corresponds to the model we trained on the calibration training set - we can directly apply it to application (evaluation) data
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
    print()    

    # We plot the non-calibrated minDCF and actDCF for reference
    logOdds, actDCF, minDCF = bayesPlot(SVAL2, LVAL)
    axes[1,1].plot(logOdds, minDCF, color='C1', linestyle='--', label = 'minDCF')
    axes[1,1].plot(logOdds, actDCF, color='C1', linestyle=':', label = 'actDCF (pre-cal).')    
    print ('System 2')
    print ('\tValidation set')
    print ('\t\tminDCF(p=0.2)         : %.3f' % bayesRisk.compute_minDCF_binary_fast(SVAL2, LVAL, 0.2, 1.0, 1.0))
    print ('\t\tactDCF(p=0.2), no cal.: %.3f' % bayesRisk.compute_actDCF_binary_fast(SVAL2, LVAL, 0.2, 1.0, 1.0))

    # We calibrate for target prior 0.2
    pT = 0.2
    # Logistic regression calibration, trained on calibration training set
    w, b = logReg.trainWeightedLogRegBinary(vrow(SCAL2), LCAL, 0, pT)
    # Apply the logistic regression model to the calibration validation set
    calibrated_SVAL2 = (w.T @ vrow(SVAL2) + b - numpy.log(pT / (1-pT))).ravel()
    # Evaluate the calibrated scores
    print ('\t\tactDCF(p=0.2), cal.   : %.3f' % (bayesRisk.compute_actDCF_binary_fast(calibrated_SVAL2, LVAL, 0.2, 1.0, 1.0)))
    
    logOdds, actDCF, _ = bayesPlot(calibrated_SVAL2, LVAL)
    axes[1,1].plot(logOdds, actDCF, color='C1', label = 'actDCF (cal.)')
    axes[1,1].set_ylim(0.0, 0.8)
    axes[1,1].set_title('System 2 - calibration validation')
    axes[1,1].legend()

    # The final model for single-fold systems corresponds to the model we trained on the calibration training set - we can directly apply it to application (evaluation) data
    calibrated_eval_scores_sys_2 = (w.T @ vrow(eval_scores_sys_2) + b - numpy.log(pT / (1-pT))).ravel()

    print ('\tEvaluation set')
    print ('\t\tminDCF(p=0.2)         : %.3f' % bayesRisk.compute_minDCF_binary_fast(eval_scores_sys_2, eval_labels, 0.2, 1.0, 1.0))
    print ('\t\tactDCF(p=0.2), no cal.: %.3f' % bayesRisk.compute_actDCF_binary_fast(eval_scores_sys_2, eval_labels, 0.2, 1.0, 1.0))
    print ('\t\tactDCF(p=0.2), cal.   : %.3f' % bayesRisk.compute_actDCF_binary_fast(calibrated_eval_scores_sys_2, eval_labels, 0.2, 1.0, 1.0))    
    
    # We plot minDCF, non-calibrated DCF and calibrated DCF for system 2
    logOdds, actDCF_precal, minDCF = bayesPlot(eval_scores_sys_2, eval_labels)
    logOdds, actDCF_cal, _ = bayesPlot(calibrated_eval_scores_sys_2, eval_labels) # minDCF is the same
    axes[1,2].plot(logOdds, minDCF, color='C1', linestyle='--', label = 'minDCF')
    axes[1,2].plot(logOdds, actDCF_precal, color='C1', linestyle=':', label = 'actDCF (pre-cal.)')
    axes[1,2].plot(logOdds, actDCF_cal, color='C1', linestyle='-', label = 'actDCF (cal.)')    

    axes[1,2].set_ylim(0.0, 0.8)
    axes[1,2].set_title('System 2 - evaluation')
    axes[1,2].legend()
    
    # We fuse/calibrate both systems

    # As comparison, we select calibrated models trained with prior 0.2 (our target application)
    # Note that minDCF of calibrated / non calibrated scores is the same
    logOdds, actDCF, minDCF = bayesPlot(calibrated_SVAL1, LVAL)
    axes[2,1].set_title('Fusion - calibration validation')
    axes[2,1].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'S1 - minDCF')
    axes[2,1].plot(logOdds, actDCF, color='C0', linestyle='-', label = 'S1 - actDCF')
    logOdds, actDCF, minDCF = bayesPlot(calibrated_SVAL2, LVAL)
    axes[2,1].plot(logOdds, minDCF, color='C1', linestyle='--', label = 'S2 - minDCF')
    axes[2,1].plot(logOdds, actDCF, color='C1', linestyle='-', label = 'S2 - actDCF')
    
    print ('Fusion')
    pT = 0.2
    # Train fusion model using the scores of both systems as features - we build the matrix of features by stacking the row vectors of the scores of the two systems
    w, b = logReg.trainWeightedLogRegBinary(numpy.vstack([SCAL1, SCAL2]), LCAL, 0, pT)
    # Apply calibration to the validation scores - again,w e need to build the score "feature" matrix
    fused_SVAL = (w.T @ numpy.vstack([SVAL1, SVAL2]) + b - numpy.log(pT / (1-pT))).ravel()

    print ('\tValidation set')
    print ('\t\tminDCF(p=0.2)         : %.3f' % bayesRisk.compute_minDCF_binary_fast(fused_SVAL, LVAL, 0.2, 1.0, 1.0))
    print ('\t\tactDCF(p=0.2)         : %.3f' % bayesRisk.compute_actDCF_binary_fast(fused_SVAL, LVAL, 0.2, 1.0, 1.0))

    logOdds, actDCF, minDCF = bayesPlot(fused_SVAL, LVAL)
    axes[2,1].plot(logOdds, minDCF, color='C2', linestyle='--', label = 'Fusion - minDCF')
    axes[2,1].plot(logOdds, actDCF, color='C2', linestyle='-', label = 'Fusion - actDCF')    
    axes[2,1].set_ylim(0, 0.8)
    axes[2,1].legend()

    # The final model for single-fold systems corresponds to the model we trained on the calibration training set - we can directly apply it to application (evaluation) data
    fused_eval_scores = (w.T @ numpy.vstack([eval_scores_sys_1, eval_scores_sys_2]) + b - numpy.log(pT / (1-pT))).ravel()

    print ('\tEvaluation set')
    print ('\t\tminDCF(p=0.2)         : %.3f' % bayesRisk.compute_minDCF_binary_fast(fused_eval_scores, eval_labels, 0.2, 1.0, 1.0))
    print ('\t\tactDCF(p=0.2)         : %.3f' % bayesRisk.compute_actDCF_binary_fast(fused_eval_scores, eval_labels, 0.2, 1.0, 1.0))    

    # We add to the plot the individual systems as reference
    logOdds, actDCF, minDCF = bayesPlot(calibrated_eval_scores_sys_1, eval_labels)
    axes[2,2].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'S1 - minDCF')
    axes[2,2].plot(logOdds, actDCF, color='C0', linestyle='-', label = 'S1 - actDCF')

    logOdds, actDCF, minDCF = bayesPlot(calibrated_eval_scores_sys_2, eval_labels)
    axes[2,2].plot(logOdds, minDCF, color='C1', linestyle='--', label = 'S2 - minDCF')
    axes[2,2].plot(logOdds, actDCF, color='C1', linestyle='-', label = 'S2 - actDCF')
    
    logOdds, actDCF, minDCF = bayesPlot(fused_eval_scores, eval_labels)
    axes[2,2].plot(logOdds, minDCF, color='C2', linestyle='--', label = 'Fusion - minDCF')
    axes[2,2].plot(logOdds, actDCF, color='C2', linestyle='-', label = 'Fusion - actDCF')    
    axes[2,2].set_ylim(0, 0.8)
    axes[2,2].set_title('Fusion - evaluation')
    axes[2,2].legend()

    plt.show()
