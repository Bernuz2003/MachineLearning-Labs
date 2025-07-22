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

    # We analyze the results for the scores of the whole dataset
    
    print('Sys1: minDCF (0.2) = %.3f - actDCF (0.2) = %.3f' % (
        bayesRisk.compute_minDCF_binary_fast(scores_sys_1, labels, 0.2, 1.0, 1.0),
        bayesRisk.compute_actDCF_binary_fast(scores_sys_1, labels, 0.2, 1.0, 1.0)))

    print('Sys2: minDCF (0.2) = %.3f - actDCF (0.2) = %.3f' % (
        bayesRisk.compute_minDCF_binary_fast(scores_sys_2, labels, 0.2, 1.0, 1.0),
        bayesRisk.compute_actDCF_binary_fast(scores_sys_2, labels, 0.2, 1.0, 1.0)))

    # Comparison of actDCF / minDCF of both systems
    plt.figure()
    logOdds, actDCF, minDCF = bayesPlot(scores_sys_1, labels)
    plt.plot(logOdds, minDCF, color='C0', linestyle='--', label = 'minDCF (System 1)')
    plt.plot(logOdds, actDCF, color='C0', linestyle='-', label = 'actDCF (System 1)')

    logOdds, actDCF, minDCF = bayesPlot(scores_sys_2, labels)
    plt.plot(logOdds, minDCF, color='C1', linestyle='--', label = 'minDCF (System 2)')
    plt.plot(logOdds, actDCF, color='C1', linestyle='-', label = 'actDCF (System 2)')
    
    plt.ylim([0, 0.8])
    plt.legend()
    plt.show()
