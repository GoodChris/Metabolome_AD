# -*- coding: utf-8 -*-
"""
@author: Xiaohui Luo
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from numpy import interp
from sklearn.metrics import roc_auc_score, confusion_matrix



def get_best_params(data,lp=['AD','NC']):
    #the best value of C was determined using cross validation with GridSearchCV in discovery cohort
    data=data[data['label'].isin(lp)]
    label=data['label'].values.tolist()
    y_label = [0 if x == 'NC' else 1 for x in label]
    y = y_label   
    X=data.drop(['Subject','label'],axis=1)
    X=np.array(X)
    cv = StratifiedKFold(n_splits=10)
    classifier = LogisticRegression(max_iter=1000000) 
    param_grid={'C':np.linspace(1.5,2.5,11)}
    GS = GridSearchCV(classifier, param_grid, cv=cv, scoring='roc_auc')
    GS.fit(X, y)  
    C=GS.best_params_['C']
    return C



def get_95CI(y_test,y_pred,y_pred_proba):
    
    auc_value=roc_auc_score(y_test, y_pred_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    
    n_bootstraps = 1000
    auc_bootstrap = []
    specificity_bootstrap = []
    sensitivity_bootstrap = []
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(len(y_test), len(y_test), replace=True)
        y_test_bootstrap = y_test[indices]
        y_pred_proba_bootstrap = y_pred_proba[indices]
        y_pred_bootstrap = (y_pred_proba_bootstrap > 0.5).astype(int)
    
        auc_bootstrap.append(roc_auc_score(y_test_bootstrap, y_pred_proba_bootstrap))
        tn, fp, fn, tp = confusion_matrix(y_test_bootstrap, y_pred_bootstrap).ravel()
        specificity_bootstrap.append(tn / (tn + fp))
        sensitivity_bootstrap.append(tp / (tp + fn))
    
    confidence_level = 0.95
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    auc_ci = np.percentile(auc_bootstrap, [lower_percentile, upper_percentile])
    specificity_ci = np.percentile(specificity_bootstrap, [lower_percentile, upper_percentile])
    sensitivity_ci = np.percentile(sensitivity_bootstrap, [lower_percentile, upper_percentile])
    
    print("AUC:",auc_value)
    print("AUC 95% Confidence Interval:", auc_ci)
    print("specificity:",specificity)
    print("Specificity 95% Confidence Interval:", specificity_ci)
    print("sensitivity:",sensitivity)
    print("Sensitivity 95% Confidence Interval:", sensitivity_ci)
    
    return auc_ci
    
def cls_SP(data_train,data_test,C,lp=['AD','NC']):
    
    data_train=data_train[data_train['label'].isin(lp)]
    label_train=data_train['label'].values.tolist()
    y_train = [0 if x == 'NC' else 1 for x in label_train]
    X_train=np.array(data_train.drop(['Subject','label'],axis=1))
    
    data_test=data_test[data_test['label'].isin(lp)]
    label_test=data_test['label'].values.tolist()
    y_test = [0 if x == 'NC' else 1 for x in label_test]
    X_test=np.array(data_test.drop(['Subject','label'],axis=1))
    
        
    Elr=LogisticRegression(C=C,max_iter=1000000)
    Elr.fit(X_train,y_train)
    
    y_true1, y_pred1 = y_train, Elr.predict(X_train)
    
    accuracy1=metrics.accuracy_score(y_true1,y_pred1)
    precision1=metrics.precision_score(y_true1,y_pred1,zero_division=0)
    recall1=metrics.recall_score(y_true1,y_pred1)
    f11=metrics.f1_score(y_true1,y_pred1)
    



    print("The scores are computed on the train set.")
    print()
    print('accuracy：%s'%accuracy1)
    print('precision：%s'%precision1)
    print('recall：%s'%recall1)
    print('F1：%s'%f11)
    print() 
       
    predictions = Elr.predict_proba(X_train)
    y_pred_proba=predictions[:, 1]
    auc_ci=get_95CI(np.array(y_true1),y_pred1,y_pred_proba)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, predictions[:, 1])  
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.figure(figsize=(6,5))
    plt.title('Discovery')  
    plt.plot(false_positive_rate, true_positive_rate, color='b',lw=2.5, alpha=.5,label='AUC = %0.3f\n95%% CI:%.3f-%.3f' % (roc_auc,auc_ci[0],auc_ci[1]))
    plt.legend(loc='lower right')  
    plt.plot([0, 1], [0, 1], '--',color='grey',lw=1.5, alpha=.8)  
    plt.xlim([-0.05, 1.05])  
    plt.ylim([-0.05, 1.05])  
    plt.ylabel('True Positive Rate')  
    plt.xlabel('False Positive Rate')
    plt.savefig('../PIC/AUC_discovery.pdf',dpi=150,format='pdf',bbox_inches='tight')
    plt.show()
    
    
    y_true2, y_pred2 = y_test, Elr.predict(X_test)
    
    accuracy2=metrics.accuracy_score(y_true2,y_pred2)
    precision2=metrics.precision_score(y_true2,y_pred2,zero_division=0)
    recall2=metrics.recall_score(y_true2,y_pred2)
    f12=metrics.f1_score(y_true2,y_pred2)
    
    
    print("The scores are computed on the test set.")
    print()
    print('accuracy：%s'%accuracy2)
    print('precision：%s'%precision2)
    print('recall：%s'%recall2)
    print('F1：%s'%f12)
    print() 
       
    predictions = Elr.predict_proba(X_test)
    y_pred_proba=predictions[:, 1]
    auc_ci=get_95CI(np.array(y_true2),y_pred2,y_pred_proba)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions[:, 1])  
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.figure(figsize=(6,5))
    plt.title('Replication')  
    plt.plot(false_positive_rate, true_positive_rate, color='mediumpurple',lw=2.5, alpha=.8,label='AUC = %0.3f\n95%% CI:%.3f-%.3f' % (roc_auc,auc_ci[0],auc_ci[1]))
    plt.legend(loc='lower right')  
    plt.plot([0, 1], [0, 1], '--',color='grey',lw=1.5, alpha=.8)  
    plt.xlim([-0.05, 1.05])  
    plt.ylim([-0.05, 1.05])  
    plt.ylabel('True Positive Rate')  
    plt.xlabel('False Positive Rate')
    plt.savefig('../PIC/AUC_replication.pdf',dpi=150,format='pdf',bbox_inches='tight')
    plt.show()



def main(D1,D2,lp=['AD','NC']):
    
    C=get_best_params(D1,lp=lp)
    print("the trained param C is:",C) 
    cls_SP(D1,D2,C,lp=['AD','NC'])     

if __name__=='__main__':
    

    
    discoveryAD=pd.read_csv('../data/discovery_AD_data.csv')
    replicationAD=pd.read_csv('../data/replication_AD_data.csv')
    
    main(discoveryAD,replicationAD,lp=['AD','NC'])

