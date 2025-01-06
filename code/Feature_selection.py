# -*- coding: utf-8 -*-
"""
@author: Xiaohui Luo
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def calculate_AUC(X_train,y_train,X_test,y_test):

    classifier = LogisticRegression(max_iter=100000)
    classifier.fit(X_train,y_train)
    y_pred_proba=classifier.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    return auc


def forward_select(X_train,y_train,X_test,y_test,flist):
    '''
    Parameters
    ----------
    flist : List.
        Candidate feature sets.

    Returns
    -------
    selected_f : List
        Identified Features 
    C_AUC : List
        Trajectory of AUC
    '''
    
    covariates=['Age','Gender','Edu_yrs','BMI']
    variates=set(flist)
    
    selected_f=[]
    cur_auc,best_auc=0,0
    C_AUC=[]
    
    while variates:        
        variate_auc=[]       
        for var in tqdm(variates):
            selected_f.append(var)
            features=covariates+selected_f
            
            X_train=X_train[features]
            X_test=X_test[features]
            auc=calculate_AUC(X_train,y_train,X_test,y_test)
            variate_auc.append((auc,var))
            selected_f.remove(var)
                
        variate_auc.sort(reverse=False)
        best_auc, best_var=variate_auc.pop()
        variates.remove(best_var)
        selected_f.append(best_var)
        cur_auc=best_auc
        C_AUC.append(cur_auc)
        
    return selected_f,C_AUC


if __name__=='__main__':
    
    pass

