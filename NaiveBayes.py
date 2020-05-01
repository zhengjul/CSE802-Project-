import numpy as np
import pandas as pd
import math
import collections 
from sklearn import metrics
from sklearn.metrics import classification_report
import logging

def prior_fun(truth):
    truth_dict = collections.Counter(truth)
    prior = np.zeros(8)
    for i in range(0, 8):
        prior[i] = truth_dict[i]/truth.shape[0]
    return prior

def mean_and_variance(df, truth):
    df = pd.DataFrame(df)
    num_features = df.shape[1]
    _mean = np.ones((8, num_features))
    _var = np.ones((8, num_features))
    test0 = pd.DataFrame()
    test1 = pd.DataFrame()
    test2 = pd.DataFrame()
    test3 = pd.DataFrame()
    test4 = pd.DataFrame()
    test5 = pd.DataFrame()
    test6 = pd.DataFrame()
    test7 = pd.DataFrame()
    k0 = 0
    k1 = 1
    k2 = 2
    k3 = 3
    k4 = 4
    k5 = 5
    k6 = 6
    k7 = 7
    for i in range(0, df.shape[0]):
        if truth[i] == 0:
            test0 = test0.append(df.iloc[i])
            k0 = k0 + 1
        elif truth[i] == 1:
            test1 = test1.append(df.iloc[i])
            k1 = k1 + 1
        elif truth[i] == 2:
            test2 = test2.append(df.iloc[i])
            k2 = k2 + 1        
        elif truth[i] == 3:
            test3 = test3.append(df.iloc[i])
            k3 = k3 + 1        
        elif truth[i] == 4:
            test4 = test4.append(df.iloc[i])
            k4 = k4 + 1
        elif truth[i] == 5:
            test5 = test5.append(df.iloc[i])
            k5 = k5 + 1
        elif truth[i] == 6:
            test6  = test6.append(df.iloc[i])
            k6 = k6 + 1        
        else:
            test7 = test7.append(df.iloc[i])
            k7 = k7 + 1
    test0.reset_index(drop=True, inplace=True)
    test1.reset_index(drop=True, inplace=True)
    test2.reset_index(drop=True, inplace=True)
    test3.reset_index(drop=True, inplace=True)
    test4.reset_index(drop=True, inplace=True)
    test5.reset_index(drop=True, inplace=True)
    test6.reset_index(drop=True, inplace=True)
    test7.reset_index(drop=True, inplace=True)
    for j in range(0, num_features):
        _mean[0][j] = np.mean(test0[j])
        _mean[1][j] = np.mean(test1[j])
        _mean[2][j] = np.mean(test2[j])
        _mean[3][j] = np.mean(test3[j])
        _mean[4][j] = np.mean(test4[j])
        _mean[5][j] = np.mean(test5[j])
        _mean[6][j] = np.mean(test6[j])
        _mean[7][j] = np.mean(test7[j])
    v0 = np.cov(test0.T.astype(np.float64))
    v1 = np.cov(test1.T.astype(np.float64))
    v2 = np.cov(test2.T.astype(np.float64))
    v3 = np.cov(test3.T.astype(np.float64))
    v4 = np.cov(test4.T.astype(np.float64))
    v5 = np.cov(test5.T.astype(np.float64))
    v6 = np.cov(test6.T.astype(np.float64))
    v7 = np.cov(test7.T.astype(np.float64))
    _var = [v0, v1, v2, v3, v4, v5, v6, v7]
    return _mean, _var # _mean and _variance 

def posterior_fun(_mean, _var, t):
    num_features = _mean.shape[1]
    t = pd.DataFrame(t)
    posterior = np.zeros(8)
    t = t.T
    t.reset_index(drop=True, inplace=True)
    for i in range(0, 8):
        p = 1
        #for j in range(0, num_features):
        if np.linalg.det(_var[i]) > 0.0:
            p = p*(1/(math.pow(2*math.pi, num_features/2)*math.sqrt(np.linalg.det(_var[i])))*math.exp((-0.5*pd.DataFrame(t-_mean[i]).T*pd.DataFrame(np.linalg.inv(_var[i]))*pd.DataFrame(t-_mean[i]))[0][0]))
        else:
            p = 0
        posterior[i] = p
    return posterior

def NaiveBayes(df, truth, test, test_truth):
    test = pd.DataFrame(test)
    df = pd.DataFrame(df)
    _mean, _var = mean_and_variance(df, truth)
    posteriors = pd.DataFrame()
    for j in range(len(test)): 
        posteriors = posteriors.append(pd.DataFrame(posterior_fun(_mean, _var, test.iloc[j])).T)
    posteriors.reset_index(drop=True, inplace=True)
    priors = pd.DataFrame(prior_fun(truth))
    conditional = pd.DataFrame()
    total_probabilities = np.zeros(len(test))
    for j in range(len(test)):
        total_prob = 0
        for i in range(0, 8): 
            total_prob += (posteriors.iloc[j][i] * priors[0][i])
        total_probabilities[j] = total_prob  
        c = np.zeros(8)
        for i in range(0, 8):
            if total_probabilities[j] != 0.0:
                c[i] = (posteriors.iloc[j][i] * priors[0][i])/total_probabilities[j]
        conditional = conditional.append(pd.DataFrame(c).T)
    conditional.reset_index(drop=True, inplace=True)
    prediction = np.zeros(len(test))
    for j in range(len(test)):
        prediction[j] = np.where(conditional.iloc[0] == conditional.iloc[0].max())[0][0]  
    print(metrics.confusion_matrix(test_truth,prediction))
    accuracy = 1- metrics.accuracy_score(test_truth, prediction)
    print(accuracy)
    target_names = ['c-CS-s','c-CS-m','c-SC-s','c-SC-m','t-CS-s','t-CS-m','t-SC-s','t-SC-m']
    print(classification_report(test_truth, prediction,target_names=target_names))
    return accuracy #_mean, _var, priors, posteriors, conditional, prediction,metrics.confusion_matrix(test_truth,prediction), 1- metrics.accuracy_score(test_truth, prediction)
