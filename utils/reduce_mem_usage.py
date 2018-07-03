# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from over_sample import smote


# In[2]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def reduce():
    print('reduce memory usage...'.center(50, '*'))
    train, test, y, val_x, val_y, features = smote()
    train = pd.DataFrame(train)
    val_x = pd.DataFrame(val_x)
    test = pd.DataFrame(test)
    train = reduce_mem_usage(train)
    val_x = reduce_mem_usage(val_x)
    test = reduce_mem_usage(test)
    feats = [f_ for f_ in features if f_ not in ['SK_ID_CURR']]
    y = pd.DataFrame(y)
    val_y = pd.DataFrame(val_y)
    train.columns = features
    train = train[feats]
    test.columns = features
    train.to_csv('../data/train_.csv', index=False)
    y.to_csv('../data/y_.csv', index=False)
    test.to_csv('../data/test_.csv', index=False)
    #train.to_csv('../data/train_.csv')
    val_x.to_csv('../data/val_x_.csv', index=False)
    val_y.to_csv('../data/val_y_.csv', index=False)
    return train, test, y,val_x, val_y, features, feats

train, test, y,val_x, val_y, features, feats = reduce()
