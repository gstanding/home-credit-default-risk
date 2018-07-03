
# coding: utf-8

# In[1]:


import pandas as pd
import sys
sys.path.append('../feature engineer/')
from feature_combine import feature_combine

# In[5]:


def nan_process():
    print('NaN process ...'.center(50, '*'))
    train, test, y = feature_combine()
    print(train.shape, test.shape)
    train = train.fillna(-1)
    test = test.fillna(-1)
    return train, test, y


# In[6]:


#train, test = nan_process()

