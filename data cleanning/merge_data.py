
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc


# In[2]:


def merge_data():
    print('merge data ...'.center(50, '*'))
    gc.enable()
    bur_bal = pd.read_csv('../data/bureau_balance.csv')
    print('bureau_balance shape:', bur_bal.shape)
    #bur_bal.head()
    bur_bal = pd.concat([bur_bal, pd.get_dummies(bur_bal.STATUS, prefix='bur_bal_status')],
                       axis=1).drop('STATUS', axis=1)
    bur_cnts = bur_bal[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').count()
    bur_bal['bur_cnt'] = bur_bal['SK_ID_BUREAU'].map(bur_cnts['MONTHS_BALANCE'])
    avg_bur_bal = bur_bal.groupby('SK_ID_BUREAU').mean()
    avg_bur_bal.columns = ['bur_bal_' + f_ for f_ in avg_bur_bal.columns]
    del bur_bal
    gc.collect()

    bur = pd.read_csv('../data/bureau.csv')
    print('bureau shape:', bur.shape)
    #bur.head()
    bur_credit_active_dum = pd.get_dummies(bur.CREDIT_ACTIVE, prefix='ca')
    bur_credit_currency_dum = pd.get_dummies(bur.CREDIT_CURRENCY, prefix='cc')
    bur_credit_type_dum = pd.get_dummies(bur.CREDIT_TYPE, prefix='ct')

    bur_full = pd.concat([bur, bur_credit_active_dum, bur_credit_currency_dum, bur_credit_type_dum], axis=1).drop(['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE'], axis=1)
    del bur_credit_active_dum, bur_credit_currency_dum, bur_credit_type_dum
    gc.collect()
    bur_full = bur_full.merge(right=avg_bur_bal.reset_index(), how='left', on='SK_ID_BUREAU',suffixes=('', '_bur_bal'))
    nb_bureau_per_curr = bur_full[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
    bur_full['SK_ID_BUREAU'] = bur_full['SK_ID_CURR'].map(nb_bureau_per_curr['SK_ID_BUREAU'])
    avg_bur = bur_full.groupby('SK_ID_CURR').mean()
    avg_bur.columns = ['bur_' + f_ for f_ in avg_bur.columns]
    del bur, bur_full, avg_bur_bal
    gc.collect()

    prev = pd.read_csv('../data/previous_application.csv')
    print('previous_application shape:', prev.shape)
    #prev.head()
    prev_cat_features = [f_ for f_ in prev.columns if prev[f_].dtype == 'object']
    prev_dum = pd.DataFrame()
    for f_ in prev_cat_features:
        prev_dum = pd.concat([prev_dum, pd.get_dummies(prev[f_], prefix=f_)], axis=1)
    prev = pd.concat([prev, prev_dum],axis=1)
    del prev_dum
    gc.collect()
    nb_prev_per_curr = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    prev['SK_ID_PREV'] = prev['SK_ID_CURR'].map(nb_prev_per_curr['SK_ID_PREV'])
    avg_prev = prev.groupby('SK_ID_CURR').mean()
    avg_prev.columns = ['prev_' + f_ for f_ in avg_prev.columns]
    del prev
    gc.collect()

    pos = pd.read_csv('../data/POS_CASH_balance.csv')
    print('pos_cash_balance shape:', pos.shape)
    #pos.head()
    pos = pd.concat([pos, pd.get_dummies(pos['NAME_CONTRACT_STATUS'], prefix='ncs')], axis=1)
    nb_prevs = pos[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    pos['SK_ID_PREV'] = pos['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    avg_pos = pos.groupby('SK_ID_CURR').mean()
    avg_pos.columns = ['pos_' + f_ for f_ in avg_pos.columns]
    del pos, nb_prevs
    gc.collect()

    cc_bal = pd.read_csv('../data/credit_card_balance.csv')
    print('credit_card_balance shape:', cc_bal.shape)
    cc_bal = pd.concat([cc_bal, pd.get_dummies(cc_bal['NAME_CONTRACT_STATUS'], prefix='ncs')], axis=1)
    nb_prevs = cc_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    cc_bal['SK_ID_PREV'] = cc_bal['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    avg_cc_bal = cc_bal.groupby('SK_ID_CURR').mean()
    avg_cc_bal.columns = ['cc_bal_' + f_ for f_ in avg_cc_bal.columns]
    del cc_bal, nb_prevs
    gc.collect()

    inst = pd.read_csv('../data/installments_payments.csv')
    print('installment_payment shape:', inst.shape)
    nb_prevs = inst[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    inst['SK_ID_PREV'] = inst['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    avg_inst = inst.groupby('SK_ID_CURR').mean()
    avg_inst.columns = ['inst_' + f_ for f_ in avg_inst.columns]
    del inst, nb_prevs
    gc.collect()

    train = pd.read_csv('../data/application_train.csv')
    test = pd.read_csv('../data/application_test.csv')
    print('train shape:', train.shape)
    print('test shape:', test.shape)
    y = train['TARGET']
    del train['TARGET']
    cat_feats = [f_ for f_ in train.columns if train[f_].dtype == 'object']
    for f_ in cat_feats:
        train[f_], indexer = pd.factorize(train[f_])#类似于类似于类似于label encoder
        test[f_] = indexer.get_indexer(test[f_])
    train = train.merge(right = avg_bur.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right = avg_bur.reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(right = avg_prev.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right = avg_prev.reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(right = avg_pos.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right = avg_pos.reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(right = avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right = avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(right = avg_inst.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right = avg_inst.reset_index(), how='left', on='SK_ID_CURR')
    del avg_bur, avg_prev, avg_pos, avg_cc_bal, avg_inst
    gc.collect()
    return train, test, y


# In[3]:


#train, test, y = merge_data()


# In[ ]:




