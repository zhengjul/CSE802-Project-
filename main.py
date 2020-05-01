import numpy as np
import pandas as pd
import math
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor    
import autosklearn.classification
from scipy import stats
import collections 
import NaiveBayes
import classifiers_and_graphs
import matplotlib.pyplot as plt
from plotly.offline import plot 
import seaborn as sns
import logging

def pca_fun(data, n, truth, validate, test):
    data = stats.zscore(data,ddof=1)
    validate = stats.zscore(validate,ddof=1)
    test = stats.zscore(test,ddof=1)
    pca = PCA(n_components = n)
    data_pca = pca.fit_transform(data)    
    explained_variance = pca.explained_variance_ratio_
    #print('PCA variances:'+str(explained_variance))
    validate = pca.transform(validate)
    test = pca.transform(test)
    return (data_pca, validate,test)

def lda_fun(data, truth, validate, test):
    data,dropped = remove_collinear(data)
    validate = pd.DataFrame(validate)
    test = pd.DataFrame(test)
    for i in dropped:
        validate = validate.drop(i, 1)
        test = test.drop(i, 1)
    data = stats.zscore(data,ddof=1)
    validate = stats.zscore(validate,ddof=1)
    test = stats.zscore(test,ddof=1)
    lda = LDA()
    data_lda = lda.fit_transform(data, truth.ravel())    
    validate = lda.transform(validate)
    test = lda.transform(test)
    return (data_lda, validate,test)

def remove_collinear(df):
    df = pd.DataFrame(df)
    corr_matrix= df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column]>0.95)]
    j = []
    for i in to_drop:
        df = df.drop(i, 1)
        j.append(i)
    return df, j

def graphing(train, train_truth):
    n = 77
    ########### correlation graph
    corr = pd.DataFrame(train).corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool))    # Generate a mask for the upper triangle
    f, ax = plt.subplots(figsize=(11, 9))    # Set up the matplotlib figure
    cmap = sns.diverging_palette(220, 10, as_cmap=True)     # Generate a custom diverging colormap
    sns.set(style="white")
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})     # Draw the heatmap with the mask and correct aspect ratio
    f.savefig('correlation_map.png')
    ########## PCA
    data = train
    data = stats.zscore(data,ddof=1)
    #scaler = MinMaxScaler()
    #data = scaler.fit_transform(data)
    pca = PCA(n_components = n)
    graph_df = pca.fit_transform(data) 
    graph_df_var = pca.explained_variance_ratio_    
    graph_df = pd.DataFrame(graph_df)
    graph_df['class'] = pd.DataFrame(train_truth)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = ['0', '1', '2','3','4','5','6','7','8']
    colors = ['b', 'g', 'r','c','m','y','k','sienna']
    for t, color in zip(targets,colors):
        i = []
        for ind, j in enumerate(graph_df['class']):
            if int(j) == int(t):
                i.append(ind)
        ax.scatter(graph_df.loc[i, 0], graph_df.loc[i, 1], c = color, alpha=0.65,s = 30)
    ax.legend(targets)
    ax.grid()
    fig.savefig('2_pca.png')
    ######################## PCA important components histogram
    graph_var_cumulative = np.cumsum(graph_df_var)
    trace1 = dict(type='bar', x=['PC %s' %i for i in range(1,n)], y=graph_df_var,name='Individual')
    trace2 = dict(type='scatter', x=['PC %s' %i for i in range(1,n)], y=graph_var_cumulative,name='Cumulative')
    data = [trace1, trace2]
    layout=dict(title='Explained variance by different principal components',yaxis=dict(title='Explained variance in percent'), annotations=list([dict(x=1.16,y=1.05,xref='paper', yref='paper', text='Explained Variance',showarrow=False,)]))
    fig = dict(data=data, layout=layout)
    plot(fig, filename='selecting-principal-components.png')
    ######################## LDA
    data = train
    data = remove_collinear(data)                                       #LDA
    data = pd.DataFrame(data)
    #scaler = MinMaxScaler()
    #data = scaler.fit_transform(data)
    data = stats.zscore(data,ddof=1)
    lda = LDA(n_components = 2)
    graph_df = lda.fit_transform(data, train_truth.ravel())    
    graph_df = pd.DataFrame(graph_df)
    graph_df['class'] = pd.DataFrame(train_truth)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Linear Discriminant 1', fontsize = 15)
    ax.set_ylabel('Linear Discriminant 2', fontsize = 15)
    ax.set_title('2 Discriminant LDA', fontsize = 20)
    targets = ['0', '1', '2','3','4','5','6','7','8']
    colors = ['b', 'g', 'r','c','m','y','k','sienna']
    for t, color in zip(targets,colors):
        i = []
        for ind, j in enumerate(graph_df['class']):
            if int(j) == int(t):
                i.append(ind)
        ax.scatter(graph_df.loc[i, 0], graph_df.loc[i, 1], c = color, alpha=0.65,s = 30)
    ax.legend(targets)
    ax.grid()
    fig.savefig('2_lda.png')
    return

def parse_data():
    df = pd.read_csv("Data_Cortex_Nuclear.csv")
    df.describe().transpose()
    df = df.fillna(df.mean()) #can skew results
    df = df.drop('MouseID', axis=1)
    for i in range(len(df)):
        if df['class'][i] == 'c-CS-s':
            df['class'][i] = 0
        elif df['class'][i] == 'c-CS-m':
            df['class'][i] = 1        
        elif df['class'][i] == 'c-SC-s':
            df['class'][i] = 2    
        elif df['class'][i] == 'c-SC-m':
            df['class'][i] = 3            
        elif df['class'][i] == 't-CS-s':
            df['class'][i] = 4            
        elif df['class'][i] == 't-CS-m':
            df['class'][i] = 5    
        elif df['class'][i] == 't-SC-s':
            df['class'][i] = 6    
        else: #'t-SC-m':
            df['class'][i] = 7   
        if df['Genotype'][i] == 'Control':  
            df['Genotype'][i] = 0
        else:           #Ts65Dn
            df['Genotype'][i] = 1        
        if df['Treatment'][i] == 'Memantine':  
            df['Treatment'][i] = 0
        else:           #Saline
            df['Treatment'][i] = 1         
        if df['Behavior'][i] == 'C/S':  
            df['Behavior'][i] = 0
        else:           #S/C
            df['Behavior'][i] = 1 
    train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))]) #20% test, 20% validation, 60% training
    train.to_csv('train.csv', sep=',', encoding='utf-8',index=False)
    validate.to_csv('validate.csv', sep=',', encoding='utf-8',index=False)
    test.to_csv('test.csv', sep=',', encoding='utf-8',index=False)
    df = df.drop('Genotype', axis=1)
    df = df.drop('Treatment', axis=1)
    df = df.drop('Behavior', axis=1)
    train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))]) #20% test, 20% validation, 60% training
    train = pd.DataFrame(train)
    validate = pd.DataFrame(validate)
    test = pd.DataFrame(test)
    train_truth = train['class']
    train = train.drop('class', axis=1)
    train.reset_index(drop=True, inplace=True)
    train = train.to_numpy()
    train = pd.DataFrame(train)
    truth = pd.to_numeric(train_truth)
    validate.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    test_truth = test['class']
    test = test.drop('class', axis=1)
    validate_truth = validate['class']
    validate = validate.drop('class', axis=1)
    train_truth = pd.DataFrame(train_truth).apply(pd.to_numeric)
    validate = pd.DataFrame(validate).apply(pd.to_numeric)
    validate_truth = pd.DataFrame(validate_truth).apply(pd.to_numeric)
    test = pd.DataFrame(test).apply(pd.to_numeric)
    test_truth = pd.DataFrame(test_truth).apply(pd.to_numeric)
    validate_truth = validate_truth.to_numpy()
    test_truth = test_truth.to_numpy()
    validate = validate.to_numpy()
    test = test.to_numpy()
    train_truth = train_truth.to_numpy()
    return train, validate, test, train_truth, validate_truth, test_truth, df

##########################################################################
##########################################################################
if __name__ == '__main__':
    train, validate, test, train_truth, validate_truth, test_truth,df = parse_data()
    train_truth = train_truth.ravel()
    validate_truth = validate_truth.ravel()
    test_truth = test_truth.ravel()
    df1 = train
    df2 = validate
    df3 = test
    df1_t = train_truth
    df2_t = validate_truth
    df3_t = test_truth
    ################################ Graphing
    #graphing(train, train_truth)
    n = 22 #manually found 22 features from pca makes up 92% of information in this dataset. 
    ########## feature selection
    train_pca, validate_pca,test_pca = pca_fun(train, n, train_truth, validate, test)
    train_lda, validate_lda, test_lda = lda_fun(train, train_truth, validate, test)
    ############################ Classify
    logger = logging.getLogger('myapp')
    hdlr = logging.FileHandler('classifiers.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.WARN)
    '''
    for i in range(1):
        ############ SVM
        print('\nSVM tuned classifier')
        classifiers_and_graphs.classifier_graph_mode('svm',df,'',train_truth)
        print('\nSVM tuned classifier - PCA')
        classifiers_and_graphs.classifier_graph_mode('svm',df, 'PCA',train_truth)
        print('\nSVM tuned classifier - lda')
        classifiers_and_graphs.classifier_graph_mode('svm',df, 'LDA',train_truth)
        
        ############ Random Forest
        print('\nRandom Forest tuned classifier')
        classifiers_and_graphs.classifier_graph_mode('randomforest',df,'',train_truth)
        print('\nRandom Forest tuned classifier - PCA')
        classifiers_and_graphs.classifier_graph_mode('randomforest',df, 'PCA',train_truth)
        print('\nRandom Forest tuned classifier - lda')
        classifiers_and_graphs.classifier_graph_mode('randomforest',df, 'LDA',train_truth)
        
        ############ KNN
        print('\nKNN tuned classifier')
        classifiers_and_graphs.classifier_graph_mode('knn',df,'',train_truth)
        print('\nKNN tuned classifier - PCA')
        classifiers_and_graphs.classifier_graph_mode('knn',df, 'PCA',train_truth)
        print('\nKNN tuned classifier - lda')
        classifiers_and_graphs.classifier_graph_mode('knn',df, 'LDA',train_truth)
        
        ############ Neural Networks
        print('\nMLP tuned classifier')
        classifiers_and_graphs.classifier_graph_mode('mlp',df,'',train_truth)
        print('\nMLP tuned classifier - PCA')
        classifiers_and_graphs.classifier_graph_mode('mlp',df, 'PCA',train_truth)
        print('\nMLP tuned classifier - lda')
        classifiers_and_graphs.classifier_graph_mode('mlp',df, 'LDA',train_truth)
        
        ############ Ensemble -- AdaBoost
        print('\nAdaBoost tuned classifier')
        classifiers_and_graphs.classifier_graph_mode('AdaBoost',df,'',train_truth)
        print('\nAdaBoost tuned classifier - PCA')
        classifiers_and_graphs.classifier_graph_mode('AdaBoost',df, 'PCA',train_truth)
        print('\nAdaBoost tuned classifier - lda')
        classifiers_and_graphs.classifier_graph_mode('AdaBoost',df, 'LDA',train_truth)
    '''
    runs = 1
    for i in range(runs):
        ########## Naive Bayes
        print('Naive Bayes cross validation set')
        #NaiveBayes.NaiveBayes(train,train_truth,validate, validate_truth)
        print('\nNaive Bayes cross test set')
        accuracy = NaiveBayes.NaiveBayes(train,train_truth,test, test_truth)
        logger.warn('Naive Bayes:     '+str(accuracy))
        
        print('Naive Bayes cross validation set - PCA')
        #NaiveBayes.NaiveBayes(train_pca,train_truth,validate_pca, validate_truth)
        print('\nNaive Bayes cross test set - PCA')
        accuracy = NaiveBayes.NaiveBayes(train_pca,train_truth,test_pca, test_truth)
        logger.warn('Naive Bayes PCA:     '+str(accuracy))
        
        print('Naive Bayes cross validation set - lda')
        #NaiveBayes.NaiveBayes(train_lda,train_truth,validate_lda, validate_truth)
        print('\nNaive Bayes cross test set - lda')
        accuracy = NaiveBayes.NaiveBayes(train_lda,train_truth,test_lda, test_truth)
        logger.warn('Naive Bayes LDA:     '+str(accuracy))
        ############ SVM
        print('\nSVM tuned classifier')
        classifiers_and_graphs.classifier('svm',train,train_truth,validate, validate_truth,test, test_truth ,'')
        print('\nSVM tuned classifier - PCA')
        classifiers_and_graphs.classifier('svm',train_pca,train_truth,validate_pca, validate_truth,test_pca, test_truth, 'PCA')
        print('\nSVM tuned classifier - lda')
        classifiers_and_graphs.classifier('svm',train_lda,train_truth,validate_lda, validate_truth,test_lda, test_truth, 'LDA')
        
        ############ Random Forest
        print('\nRandom Forest tuned classifier')
        classifiers_and_graphs.classifier('randomforest',train,train_truth,validate, validate_truth,test, test_truth,'')
        print('\nRandom Forest tuned classifier - PCA')
        classifiers_and_graphs.classifier('randomforest',train_pca,train_truth,validate_pca, validate_truth,test_pca, test_truth, 'PCA')
        print('\nRandom Forest tuned classifier - lda')
        classifiers_and_graphs.classifier('randomforest',train_lda,train_truth,validate_lda, validate_truth,test_lda, test_truth, 'LDA')
        
        ############ KNN
        print('\nKNN tuned classifier')
        classifiers_and_graphs.classifier('knn',train,train_truth,validate, validate_truth,test, test_truth,'')
        print('\nKNN tuned classifier - PCA')
        classifiers_and_graphs.classifier('knn',train_pca,train_truth,validate_pca, validate_truth,test_pca, test_truth, 'PCA')
        print('\nKNN tuned classifier - lda')
        classifiers_and_graphs.classifier('knn',train_lda,train_truth,validate_lda, validate_truth,test_lda, test_truth, 'LDA')
        
        ############ Neural Networks
        print('\nMLP tuned classifier')
        classifiers_and_graphs.classifier('mlp',train,train_truth,validate, validate_truth,test, test_truth,'')
        print('\nMLP tuned classifier - PCA')
        classifiers_and_graphs.classifier('mlp',train_pca,train_truth,validate_pca, validate_truth,test_pca, test_truth, 'PCA')
        print('\nMLP tuned classifier - lda')
        classifiers_and_graphs.classifier('mlp',train_lda,train_truth,validate_lda, validate_truth,test_lda, test_truth, 'LDA')
        
        ############ Ensemble -- AdaBoost
        print('\nAdaBoost tuned classifier')
        classifiers_and_graphs.classifier('AdaBoost',train,train_truth,validate, validate_truth,test, test_truth,'')
        print('\nAdaBoost tuned classifier - PCA')
        classifiers_and_graphs.classifier('AdaBoost',train_pca,train_truth,validate_pca, validate_truth,test_pca, test_truth, 'PCA')
        print('\nAdaBoost tuned classifier - lda')
        classifiers_and_graphs.classifier('AdaBoost',train_lda,train_truth,validate_lda, validate_truth,test_lda, test_truth, 'LDA')
        
        ############ python sklearn auto
        print('\nSklearn autoclassifier')
        cls = autosklearn.classification.AutoSklearnClassifier()
        cls.fit(train, train_truth)
        prediction = cls.predict(test)
        target_names = ['c-CS-s','c-CS-m','c-SC-s','c-SC-m','t-CS-s','t-CS-m','t-SC-s','t-SC-m']
        print(classification_report(test_truth, prediction,target_names=target_names))
        print(metrics.confusion_matrix(test_truth,prediction))
        print(metrics.accuracy_score(test_truth, prediction))
        logger.warn('Autoclassifier:     '+str(accuracy))
        
        print('\nSklearn autoclassifier - PCA')
        cls = autosklearn.classification.AutoSklearnClassifier()
        cls.fit(train_pca, train_truth)
        prediction = cls.predict(test_pca)
        target_names = ['c-CS-s','c-CS-m','c-SC-s','c-SC-m','t-CS-s','t-CS-m','t-SC-s','t-SC-m']
        print(classification_report(test_truth, prediction,target_names=target_names))
        print(metrics.confusion_matrix(test_truth,prediction))
        print(metrics.accuracy_score(test_truth, prediction))
        logger.warn('Autoclassifier PCA:     '+str(accuracy))
        
        print('\nSklearn autoclassifier - LDA')
        cls = autosklearn.classification.AutoSklearnClassifier()
        cls.fit(train_lda, train_truth)
        prediction = cls.predict(test_lda)
        target_names = ['c-CS-s','c-CS-m','c-SC-s','c-SC-m','t-CS-s','t-CS-m','t-SC-s','t-SC-m']
        print(classification_report(test_truth, prediction,target_names=target_names))
        print(metrics.confusion_matrix(test_truth,prediction))
        print(metrics.accuracy_score(test_truth, prediction))
        logger.warn('Autoclassifier LDA:     '+str(accuracy))
        