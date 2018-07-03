
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sys
sys.path.append('../data cleanning/')
from merge_data import merge_data

# In[2]:


def feat_ext_source(df):
    x1 = df['EXT_SOURCE_1'].fillna(-1) + 1e-1
    x2 = df['EXT_SOURCE_2'].fillna(-1) + 1e-1
    x3 = df['EXT_SOURCE_3'].fillna(-1) + 1e-1
    
    df['EXT_SOURCE_1over2_NAminus1_Add0.1'] = x1/x2
    df['EXT_SOURCE_2over1_NAminus1_Add0.1'] = x2/x1
    df['EXT_SOURCE_1over3_NAminus1_Add0.1'] = x1/x3
    df['EXT_SOURCE_3over1_NAminus1_Add0.1'] = x3/x1
    df['EXT_SOURCE_2over3_NAminus1_Add0.1'] = x2/x3
    df['EXT_SOURCE_3over2_NAminus1_Add0.1'] = x3/x2
    df['EXT_SOURCE_1_log'] = np.log(df['EXT_SOURCE_1'] + 1)
    df['EXT_SOURCE_2_log'] = np.log(df['EXT_SOURCE_2'] + 1)
    df['EXT_SOURCE_3_log'] = np.log(df['EXT_SOURCE_3'] + 1) 
    return df
def feature_combine():
    print('feature combine...'.center(50, '*'))
    train, test, y = merge_data()
    train = feat_ext_source(train)
    test = feat_ext_source(test)
    return train, test, y


# In[ ]:


#train = feat_ext_source(train)
#test = feat_ext_source(test)

