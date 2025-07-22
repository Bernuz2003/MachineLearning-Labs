from gmm import *

import bayesRisk

if __name__ == '__main__':

    # IRIS
    print('IRIS')
    D, L = load_iris()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    for covType in ['full', 'diagonal', 'tied']:
        for numC in [1,2,4,8,16]:
            gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], numC, covType = covType, verbose=False, psiEig = 0.01)
            gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], numC, covType = covType, verbose=False, psiEig = 0.01)
            gmm2 = train_GMM_LBG_EM(DTR[:, LTR==2], numC, covType = covType, verbose=False, psiEig = 0.01)

            SVAL = []
            SVAL.append(logpdf_GMM(DVAL, gmm0))
            SVAL.append(logpdf_GMM(DVAL, gmm1))
            SVAL.append(logpdf_GMM(DVAL, gmm2))
            SVAL = numpy.vstack(SVAL) # Class-conditional log-likelihoods
            SVAL += vcol(numpy.log(numpy.ones(3)/3)) # We add the log-prior to get the log-joint
            PVAL = SVAL.argmax(0) # Predictions
            print('Cov Type:', covType.ljust(10), '- %d Gau -' % numC, end = ' ')
            print('Error rate: %.1f%%' % ((LVAL != PVAL).sum() / LVAL.size * 100))

    # Additional binary task

    print()
    print('Binary task')    
    D, L = numpy.load('Data/ext_data_binary.npy'), numpy.load('Data/ext_data_binary_labels.npy')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    for covType in ['full', 'diagonal', 'tied']:
        print(covType)
        for numC in [1,2,4,8,16]:
            gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], numC, covType = covType, verbose=False, psiEig = 0.01)
            gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], numC, covType = covType, verbose=False, psiEig = 0.01)

            SLLR = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)
            #print(numC, covType)
            #print ('minDCF - pT = 0.5: %.4f' % bayesRisk.compute_minDCF_binary_fast(SLLR, LVAL, 0.5, 1.0, 1.0))
            #print ('actDCF - pT = 0.5: %.4f' % bayesRisk.compute_actDCF_binary_fast(SLLR, LVAL, 0.5, 1.0, 1.0))
            print ('\tnumC = %d: %.4f / %.4f' % (numC, bayesRisk.compute_minDCF_binary_fast(SLLR, LVAL, 0.5, 1.0, 1.0), bayesRisk.compute_actDCF_binary_fast(SLLR, LVAL, 0.5, 1.0, 1.0)))
        print()
