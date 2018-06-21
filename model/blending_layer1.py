# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 10:56:04 2018

@author: gstanding
"""

import numpy as np
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score
class blending_layer1(object):
    
    def __init__(self, train_x, train_y, val_x, val_y, test_x):
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.test_x = test_x
        
    def shuffle_data(self, model=None):
        ss = ShuffleSplit(n_splits=4, random_state=0, test_size=.25)
        for n_fold, (trn_idx, val_idx) in enumerate(ss.split(self.train_x)):
            if n_fold == 0:
                rf_train_x = self.train_x[val_idx]
                rf_train_y = self.train_y[val_idx]
            elif n_fold == 1:
                et_train_x = self.train_x[val_idx]
                et_train_y = self.train_y[val_idx]
            elif n_fold == 2:
                gbdt_train_x = self.train_x[val_idx]
                gbdt_train_y = self.train_y[val_idx]
            else:
                xgb_train_x = self.train_x[val_idx]
                xgb_train_y = self.train_y[val_idx]
        if model == 'rf':
            return rf_train_x, rf_train_y
        elif model == 'et':
            return et_train_x, et_train_y
        elif model == 'gbdt':
            return gbdt_train_x, gbdt_train_y
        elif model == 'xgb':
            return xgb_train_x, xgb_train_y
        else:
            return ss
        
    def rf_model(self, parameters=None):
        rf_train_x, rf_train_y = self.shuffle_data('rf')
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        rf_val_pred = np.zeros((rf_train_x.shape[0],))
        rf_off_val_pred = np.zeros((self.val_x.shape[0],))
        rf_test_pred = np.zeros((self.test_x.shape[0],))
        for n_fold, (trn_idx, val_idx) in enumerate(kf.split(rf_train_x)):
            rf = RandomForestClassifier(**parameters)
            rf.fit(rf_train_x[trn_idx], rf_train_y[trn_idx])
            rf_val_pred[val_idx] = rf.predict_proba(rf_train_x[val_idx])
            rf_off_val_pred[val_idx] += rf.predict_proba(self.val_x) / kf.n_splits
            rf_test_pred += rf.predict_proba(self.test_x) / kf.n_splits
            print('Fold %d auc score: %.6f'%(n_fold+1, roc_auc_score(rf_train_y[trn_idx], rf_val_pred[val_idx])))
        print('Validate auc score:', roc_auc_score(rf_train_y, rf_val_pred))
        print('Off auc score:', roc_auc_score(self.val_y, rf_off_val_pred))
        return rf_val_pred, rf_off_val_pred, rf_test_pred
    
    def et_model(self, parameters=None):
        et_train_x, et_train_y = self.shuffle_data('et')
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        et_val_pred = np.zeros((et_train_x.shape[0],))
        et_off_val_pred = np.zeros((self.val_x.shape[0],))
        et_test_pred = np.zeros((self.test_x.shape[0],))
        for n_fold, (trn_idx, val_idx) in enumerate(kf.split(et_train_x)):
            et = ExtraTreesClassifier(**parameters)
            et.fit(et_train_x[trn_idx], et_train_y[trn_idx])
            et_val_pred[val_idx] = et.predict_proba(et_train_x[val_idx])
            et_off_val_pred[val_idx] += et.predict_proba(self.val_x) / kf.n_splits
            et_test_pred += et.predict_proba(self.test_x) / kf.n_splits
            print('Fold %d auc score: %.6f'%(n_fold+1, roc_auc_score(et_train_y[trn_idx], et_val_pred[val_idx])))
        print('Validate auc score:', roc_auc_score(et_train_y, et_val_pred))
        print('Off auc score:', roc_auc_score(self.val_y, et_off_val_pred))
        return et_val_pred, et_off_val_pred, et_test_pred
    
    def gbdt_model(self, parameters=None):
        gbdt_train_x, gbdt_train_y = self.shuffle_data('gbdt')
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        gbdt_val_pred = np.zeros((gbdt_train_x.shape[0],))
        gbdt_off_val_pred = np.zeros((self.val_x.shape[0],))
        gbdt_test_pred = np.zeros((self.test_x.shape[0],))
        for n_fold, (trn_idx, val_idx) in enumerate(kf.split(gbdt_train_x)):
            gbdt = GradientBoostingClassifier(**parameters)
            gbdt.fit(gbdt_train_x[trn_idx], gbdt_train_y[trn_idx])
            gbdt_val_pred[val_idx] = gbdt.predict_proba(gbdt_train_x[val_idx])
            gbdt_off_val_pred[val_idx] += gbdt.predict_proba(self.val_x) / kf.n_splits
            gbdt_test_pred += gbdt.predict_proba(self.test_x) / kf.n_splits
            print('Fold %d auc score: %.6f'%(n_fold+1, roc_auc_score(gbdt_train_y[trn_idx], gbdt_val_pred[val_idx])))
        print('Validate auc score:', roc_auc_score(gbdt_train_y, gbdt_val_pred))
        print('Off auc score:', roc_auc_score(self.val_y, gbdt_off_val_pred))
        return gbdt_val_pred, gbdt_off_val_pred, gbdt_test_pred
    
    def xgb_model(self, parameters=None):
        xgb_train_x, xgb_train_y = self.shuffle_data('xgb')
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        xgb_val_pred = np.zeros((xgb_train_x.shape[0],))
        xgb_off_val_pred = np.zeros((self.val_x.shape[0],))
        xgb_test_pred = np.zeros((self.test_x.shape[0],))
        for n_fold, (trn_idx, val_idx) in enumerate(kf.split(xgb_train_x)):
            xgb = xgb(**parameters)
            xgb.fit(xgb_train_x[trn_idx], xgb_train_y[trn_idx])
            xgb_val_pred[val_idx] = xgb.predict_proba(xgb_train_x[val_idx])
            xgb_off_val_pred[val_idx] += xgb.predict_proba(self.val_x) / kf.n_splits
            xgb_test_pred += xgb.predict_proba(self.test_x) / kf.n_splits
            print('Fold %d auc score: %.6f'%(n_fold+1, roc_auc_score(xgb_train_y[trn_idx], xgb_val_pred[val_idx])))
        print('Validate auc score:', roc_auc_score(xgb_train_y, xgb_val_pred))
        print('Off auc score:', roc_auc_score(self.val_y, xgb_off_val_pred))
        return xgb_val_pred, xgb_off_val_pred, xgb_test_pred
    
    def merge_data(self):
        rf_val_pred, rf_off_val_pred, rf_test_pred = self.rf_model()
        et_val_pred, et_off_val_pred, et_test_pred = self.et_model()
        gbdt_val_pred, gbdt_off_val_pred, gbdt_test_pred = self.gbdt_model()
        xgb_val_pred, xgb_off_val_pred, xgb_test_pred = self.xgb_model()
        val_pred = np.zeros((self.train_x.shape[0],))
        ss = self.shuffle_data()
        for n_fold, (trn_idx, val_idx) in enumerate(ss.split(self.train_x)):
            if n_fold == 0:
                val_pred[val_idx] = rf_val_pred
            elif n_fold == 1:
                val_pred[val_idx] = et_val_pred
            elif n_fold == 2:
                val_pred[val_idx] = gbdt_val_pred
            else:
                val_pred[val_idx] = xgb_val_pred
        off_val_pred = (rf_off_val_pred + et_off_val_pred + gbdt_off_val_pred + xgb_off_val_pred) / 4
        test_pred = (rf_test_pred + et_test_pred + gbdt_test_pred + xgb_test_pred) / 4
        return val_pred, off_val_pred, test_pred
                
            
        
        
        
        
    
        