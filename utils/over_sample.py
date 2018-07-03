
# coding: utf-8

# In[1]:


from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import sys
sys.path.append('../feature engineer/')
from nan_process import nan_process
from sklearn.model_selection import train_test_split


# In[ ]:


def smote():
    print('smote ...'.center(50, '*'))
    train, test, y = nan_process()
    train, val_x, y, val_y = train_test_split(train, y, test_size=0.1, random_state = 14)
    features = train.columns
    #sm = SMOTE(random_state=42, kind='borderline2', n_jobs=36)
    #train, y = sm.fit_sample(train, y)
    return train, test, y, val_x, val_y, features

