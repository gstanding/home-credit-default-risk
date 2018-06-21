# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 19:09:45 2018

@author: gstanding
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import random

def pipeline(iteration,C,gamma,random_seed):
    x_train, _x , y_train, _y = train_test_split(train_x,train_y,test_size=0.98,random_state=random_seed)
    print(x_train.shape)
    clf = SVC(C=C,kernel='rbf',gamma=gamma,probability=True,cache_size=7000,class_weight='balanced',verbose=True,random_state=random_seed)
    clf.fit(x_train,y_train)
    
    pred = clf.predict_proba(test_x)
    val_res  = clf.predict_proba(val_x)
    return pred, val_res
#    #predict test set
#    pred = clf.predict_proba(test_x)
#    test_result = pd.DataFrame(columns=["Idx","score"])
#    test_result.Idx = test_Idx
#    test_result.score = pred[:,1]
#    test_result.to_csv('./test/svm_{0}.csv'.format(iteration),index=None)
#    #predict val set
#    pred = clf.predict_proba(val_x)
#    val_result = pd.DataFrame(columns=["Idx","score"])
#    val_result.Idx = val_Idx
#    val_result.score = pred[:,1]
#    val_result.to_csv('./val/svm_{0}.csv'.format(iteration),index=None)


if __name__ == "__main__":
    random_seed = [i for i in range(2016,2066)]
    C = [i/10.0 for i in range(10,60)]
    gamma = [i/1000.0 for i in range(1,51)]
    random.shuffle(random_seed)
    random.shuffle(C)
    random.shuffle(gamma)
    train_x, train_y = np.random.random((5000,10)), np.random.randint(0,2,size=(5000,))
    val_x, val_y = np.random.random((50000,10)), np.random.randint(0,2,size=(50000,))
    test_x = np.random.random((30000,10))

    for i in range(50):
       pred, val_res = pipeline(i,C[i],gamma[i],random_seed[i])
       print(pred.shape, val_res.shape)