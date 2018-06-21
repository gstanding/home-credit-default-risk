# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 16:49:49 2018

@author: gstanding
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

class blending(object):
    
    def __init__(self, train_pred, train_y, val_pred, val_y, test_pred):
        self.train_pred = train_pred.reshape(-1,1)
        self.train_y = train_y.reshape(-1,1)
        self.val_pred = val_pred.reshape(-1,1)
        self.val_y = val_y.reshape(-1,1)
        self.test_pred = test_pred.reshape(-1,1)
        
    def train(self):
        clf = LogisticRegression(
            penalty = 'l1',
            C = 10.0,
            class_weight = 'balanced',
            random_state = 14,
            n_jobs = 72,
            solver= 'saga'
            )
        clf.fit(self.train_pred, self.train_y.ravel())
        lr_val_pred = clf.predict_proba(self.val_pred)[:, 1]
        lr_test_pred = clf.predict_proba(self.test_pred)[:, 1]
        print('Validate AUC score:', roc_auc_score(self.val_y.ravel(), lr_val_pred))
        return lr_test_pred
'''train_pred, train_y, val_pred, val_y, test_pred = np.random.random((7000,)), np.random.randint(0,2,size=(7000,)), np.random.random((3000,)), np.random.randint(0,2,size=(3000,)), np.random.random((3000,))
bld = blending(train_pred, train_y, val_pred, val_y, test_pred)
lr_test_pred = bld.train()'''
        
        