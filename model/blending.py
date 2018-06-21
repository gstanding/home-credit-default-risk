# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 16:49:49 2018

@author: gstanding
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

class blending(object):
    
    def __init__(self, train_pred, train_y, val_pred, val_y, test_pred):
        self.train_pred = train_pred
        self.train_y = train_y
        self.val_pred = val_pred
        self.val_y = val_y
        self.test_pred = test_pred
        
    def train(self):
        clf = LogisticRegression(
            penalty = 'l1',
            C = 10.0,
            class_weight = 'balanced',
            random_state = 14,
            n_jobs = 72
            )
        clf.fit(self.train_pred, self.train_y)
        lr_val_pred = clf.predict_proba(self.val_pred)
        lr_test_pred = clf.predict_proba(self.test_pred)
        print('Validate AUC score:', roc_auc_score(lr_val_pred, self.val_y))
        return lr_test_pred
    
        
        