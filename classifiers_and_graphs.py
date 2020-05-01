import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier,VotingClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from hypopt import GridSearch
import logging
from sklearn.metrics import precision_recall_curve, roc_auc_score, classification_report
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import main 
def classifier(classifier,train, truth, validate, validate_truth,test, test_truth, datatype):
    np.random.seed(0) 
    rng = np.random.permutation(1)[0]
    train = pd.DataFrame(train)
    validate = pd.DataFrame(validate)
    test = pd.DataFrame(test)
    logger = logging.getLogger('myapp')
    hdlr = logging.FileHandler('classifiers.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.WARN)
    if classifier.lower() == 'svm':     #best: C = 50, gamma = 0.0001, kernel = rbf
        model = svm.SVC(random_state=rng) 
        hyperparameter = {'kernel': ('linear', 'rbf'), 'C':[1,1.5, 10,50,100,200],'gamma': [1e-7, 1e-4]}
    elif classifier.lower() == 'randomforest': #120
        model = RandomForestClassifier(random_state=rng)
        hyperparameter = {'n_estimators' : np.arange(10,300,10)}
    elif classifier.lower() == 'adaboost':
        model = AdaBoostClassifier(random_state=rng)
        hyperparameter = {'n_estimators': np.arange(10,300,10), 'algorithm' :('SAMME','SAMME.R')}
    elif classifier.lower() == 'knn':           #120
        model = KNeighborsClassifier()
        hyperparameter = dict(n_neighbors=list(range(1, 100)))
    else: ## assume it's asking for neural network (multi-layer perceptron) 
        model = MLPClassifier(max_iter=100) #activation=tanh, hiddenlayersize=(20,20), 'learning_rate'=adaptive,solver=lbfgs
        hyperparameter ={'hidden_layer_sizes': [(20,20),(80,20),(80,20,20),(80,40,40,20),(40,40,20,20,20,10)], 'learning_rate': ['adaptive'],'activation': ['tanh', 'relu','logistic'],'solver': ['lbfgs','sgd', 'adam']}
    tuned_model = GridSearch(model=model, param_grid=hyperparameter)
    tuned_model.fit(train,truth)
    prediction = tuned_model.score(test,test_truth)
    logger.warn(classifier+' '+datatype+' validate    '+str(prediction))
    tuned_model.fit(train,truth, validate, validate_truth)
    prediction = tuned_model.score(test,test_truth)
    target_names = ['c-CS-s','c-CS-m','c-SC-s','c-SC-m','t-CS-s','t-CS-m','t-SC-s','t-SC-m']
    prediction = tuned_model.predict(test)
    print(classification_report(test_truth, prediction,target_names=target_names))
    logger.warn(classifier+' '+datatype+'    '+str(prediction))
    return

def classifier_graph_mode(classifier,df, datatype, train_truth):
    np.random.seed(0) 
    rng = np.random.permutation(1)[0]
    df = pd.DataFrame(df)
    df_truth = df['class']
    adf = df.drop('class', axis=1)
    n_classes = 8
    df_truth = pd.DataFrame(df_truth).apply(pd.to_numeric)
    Y = label_binarize(df_truth, classes=[*range(n_classes)])
    train, test, truth, test_truth = train_test_split(adf,Y, test_size=0.4, random_state=1)
    test, validate, test_truth, validate_truth = train_test_split(test,test_truth, test_size=0.5, random_state=1)
    train = train.to_numpy()
    validate = validate.to_numpy()
    test = test.to_numpy()
    logger = logging.getLogger('myapp')
    hdlr = logging.FileHandler('classifiers.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.WARN)
    if datatype.lower() == 'pca':
        train_pca, validate_pca,test_pca = main.pca_fun(train, 22, truth, validate, test)
    elif datatype.lower()=='lda':
        train_lda, validate_lda, test_lda = main.lda_fun(train, train_truth, validate, test)
    if classifier.lower() == 'svm':     
        model = OneVsRestClassifier(svm.SVC(probability=True,C = 50, gamma = 0.0001, kernel = 'rbf',random_state=rng))
    elif classifier.lower() == 'randomforest': #120
        model = OneVsRestClassifier(RandomForestClassifier(n_estimators=80,random_state=rng))
    elif classifier.lower() == 'adaboost':
        model = OneVsRestClassifier(AdaBoostClassifier(n_estimators=80,algorithm='SAMME',random_state=rng))
    elif classifier.lower() == 'knn':           #120
        model = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=50))
    else: ## assume it's asking for neural network (multi-layer perceptron) 
        model = OneVsRestClassifier(MLPClassifier(max_iter=100,activation='tanh', learning_rate='adaptive',solver='lbfgs')) #activation=tanh, hiddenlayersize=(20,20), 'learning_rate'=adaptive,solver=lbfgs
    tuned_model = model
    tuned_model.fit(train,truth)
    prediction = tuned_model.predict_proba(test)
    precision = dict()
    recall = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(test_truth[:, i],prediction[:, i])
        plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.savefig(classifier+'precision-recall'+datatype+'.png')
    plt.clf()
    return


