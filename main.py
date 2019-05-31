#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:38:34 2019
Last modified on Thu May 31 12:05:00 2019

@author: Alberto Fernandez - University of Granada - (alberto@decsai.ugr.es)
"""
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,roc_curve, auc
from sklearn import datasets 

from ChiRWClassifier import ChiRWClassifier

import time


dataset = datasets.load_iris()
X,y = dataset.data, dataset.target

#Ad-hoc train-test partition
X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.4, random_state=42)

#Load data from file:
"""
f = open("../data/iris/iris-10-1tra.csv")
data = np.loadtxt(f,delimiter=",")

X_tr = data[:, 0:-1]
y_tr = data[:,-1].astype(int)

f = open("../data/iris/iris-10-1tst.csv")
data = np.loadtxt(f,delimiter=",")
X_tst = data[:, 0:-1]
y_tst = data[:,-1].astype(int)
"""

start_time = time.time()
chi = ChiRWClassifier(frm="wr")

chi.fit(X_tr,y_tr)
y_pred = chi.predict(X_tr)
print("The accuracy of Chi-FRBCS model (train) is: ", accuracy_score(y_tr,y_pred))
y_pred = chi.predict(X_tst)
print("The accuracy of Chi-FRBCS model (test) is: ", accuracy_score(y_tst,y_pred))

#Only for two-class problems
"""
probas_ = chi.predict_proba(X_tst)
fpr, tpr, thresholds = roc_curve(y_tst, probas_[:, 1])
auc_tst = auc(fpr, tpr)
print("The AUC of Chi-FRBCS model  (test) es: ", auc_tst)
"""

t_exec = time.time() - start_time
hours = int(t_exec / 3600);
rest = t_exec % 3600;
minutes = int(rest / 60);
seconds = rest % 60;

print("Execution Time: ", hours , ":" , minutes , ":" , '{0:.4g}'.format(seconds))

#Standard cross-validation is also available
#scores = cross_val_score(chi, iris.data, iris.target, cv=5,scoring='accuracy')
#print(scores)