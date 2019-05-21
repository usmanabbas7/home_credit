import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
    
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('../input/bureau.csv', nrows = num_rows)
    
    bureau.replace(365243, np.nan, inplace= True)
    bureau.replace('XAN', np.nan, inplace= True)
    bureau.replace('XNA', np.nan, inplace= True)
    bureau.drop(bureau[bureau['AMT_CREDIT_SUM_DEBT']>50000000].index,axis=0,inplace=True)
    bureau.drop(bureau[bureau['AMT_ANNUITY']>10000000].index,axis=0,inplace=True)

    bureau['AMT_CREDIT_SUM_DEBT']=bureau['AMT_CREDIT_SUM_DEBT'].apply(lambda x:0 if x<0 else x)
    
    #some more features
    bureau['AMT_CREDIT_TO_LIMIT']=bureau['AMT_CREDIT_SUM']/(bureau['AMT_CREDIT_SUM_LIMIT']+1)
    bureau['CREDIT_TO_ANNUITY']=bureau['AMT_CREDIT_SUM']/(bureau['AMT_ANNUITY']+1)
    bureau['BURO_AMT_ANNUITY_TO_AMT_CREDIT_SUM_DEBT']=bureau['AMT_ANNUITY']/(bureau['AMT_CREDIT_SUM_DEBT']+1)
    bureau['CREDIT_TO_OVERDUE']=bureau['AMT_CREDIT_SUM_OVERDUE']/(bureau['AMT_CREDIT_SUM']+1)
    bureau['FLAG_ACTIVE']=bureau['CREDIT_ACTIVE'].apply(lambda x:1 if x=='Active' else 0)
    bureau['FLAG_CLOSED']=bureau['CREDIT_ACTIVE'].apply(lambda x:1 if x=='Closed' else 0)
    bureau['FLAG_CONSUMER_LOAN']=bureau['CREDIT_TYPE'].apply(lambda x:1 if x=='Consumer credit' else 0)
    bureau['FLAG_CREDIT_CARD_LOAN']=bureau['CREDIT_TYPE'].apply(lambda x:1 if x=='Credit card' else 0)
    bureau['FLAG_CAR_LOAN']=bureau['CREDIT_TYPE'].apply(lambda x:1 if x=='Car loan' else 0)
    bureau['FLAG_MORTGAGE_LOAN']=bureau['CREDIT_TYPE'].apply(lambda x:1 if x=='Mortgage' else 0)
    bureau['FLAG_MICROLOAN']=bureau['CREDIT_TYPE'].apply(lambda x:1 if x=='Microloan' else 0)
    
    consumer_df=bureau[bureau['FLAG_CONSUMER_LOAN']==1]
    credit_card_df=bureau[bureau['FLAG_CREDIT_CARD_LOAN']==1]
    car_df=bureau[bureau['FLAG_CAR_LOAN']==1]
    mortgage_df=bureau[bureau['FLAG_MORTGAGE_LOAN']==1]
    microloan_df=bureau[bureau['FLAG_MICROLOAN']==1]
    
    bb = pd.read_csv('../input/bureau_balance.csv', nrows = num_rows)
    bb.replace(365243, np.nan, inplace= True)
    bb.replace('XAN', np.nan, inplace= True)
    bb.replace('XNA', np.nan, inplace= True)
    bb['STATUS'].replace({'C':'CLOSED','X':'UNKNOWN','0':'NO_DPD','1':'MAXIMAL_DURING_MONTH_1_30','2':'DPD_31_60','3':'DPD_61_90','4':'DPD_91_120','5':'DPD_120+'},inplace=True)
    bb['FLAG_NO_DPD']=bb['STATUS'].apply(lambda x:1 if x=='NO_DPD' else 0)
    bb['FLAG_UNKNOWN']=bb['STATUS'].apply(lambda x:1 if x=='UNKNOWN' else 0)
    bb['FLAG_MAXIMAL']=bb['STATUS'].apply(lambda x:1 if x=='MAXIMAL_DURING_MONTH_1_30' else 0)
    bb['FLAG_DPD_120+']=bb['STATUS'].apply(lambda x:1 if x=='DPD_120+' else 0)
    bb['FLAG_DPD_31_60']=bb['STATUS'].apply(lambda x:1 if x=='DPD_31_60' else 0)
    bb['FLAG_DPD_61_90']=bb['STATUS'].apply(lambda x:1 if x=='DPD_61_90' else 0)
    bb['FLAG_DPD_91_120']=bb['STATUS'].apply(lambda x:1 if x=='DPD_91_120' else 0)

    #bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    #bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size'],
                       'FLAG_NO_DPD':['sum'],
                       'FLAG_UNKNOWN':['sum'],
                       'FLAG_MAXIMAL':['sum'],
                       'FLAG_DPD_120+':['sum'],
                       'FLAG_DPD_31_60':['sum'],
                       'FLAG_DPD_61_90':['sum'],
                       'FLAG_DPD_91_120':['sum']
                      }
    #for col in bb_cat:
    #    bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    
    credit_card_df = credit_card_df.join(bb_agg, how='left', on='SK_ID_BUREAU')
    consumer_df = consumer_df.join(bb_agg, how='left', on='SK_ID_BUREAU')
    car_df = car_df.join(bb_agg, how='left', on='SK_ID_BUREAU')
    microloan_df = microloan_df.join(bb_agg, how='left', on='SK_ID_BUREAU')
    mortgage_df = mortgage_df.join(bb_agg, how='left', on='SK_ID_BUREAU')
    
    #bureau.drop(columns= 'SK_ID_BUREAU', inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        #'AMT_CREDIT_SUM_DEBT_TO_AMT_CREDIT': ['min', 'max', 'mean', 'var'],
        #'AMT_CREDIT_SUM_DEBT_+_AMT_CREDIT': ['min', 'max', 'mean', 'var'],
        'AMT_CREDIT_TO_LIMIT':['min','max','mean'],
        'FLAG_MICROLOAN':['sum'],
        'FLAG_CONSUMER_LOAN':['sum'],
        'FLAG_CREDIT_CARD_LOAN':['sum'],
        'FLAG_CAR_LOAN':['sum'],
        'FLAG_MORTGAGE_LOAN':['sum'],
        'CREDIT_TO_OVERDUE':['mean','max','min'],
        'CREDIT_TO_ANNUITY':['mean','min','max','sum'],
        'BURO_AMT_ANNUITY_TO_AMT_CREDIT_SUM_DEBT': ['min', 'max', 'mean'],
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean','sum'],
        'DAYS_ENDDATE_FACT': ['min', 'max', 'mean','sum'],
        'AMT_CREDIT_MAX_OVERDUE': ['min','mean','max'],
        'CNT_CREDIT_PROLONG': ['sum','mean','min','max'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean','sum','max','min'],
        'AMT_CREDIT_SUM_LIMIT': ['mean','max','min'],
        'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean'],
        'AMT_ANNUITY': ['max', 'mean'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
        'CREDIT_CURRENCY':['nunique'],
        'SK_ID_BUREAU':['nunique'],
        'FLAG_ACTIVE':['sum'],
        'FLAG_CLOSED':['sum'],
        'CREDIT_TYPE':['nunique'],
        'FLAG_NO_DPD_SUM':['sum'],
        'FLAG_UNKNOWN_SUM':['sum'],
        'FLAG_MAXIMAL_SUM':['sum'],
        'FLAG_DPD_120+_SUM':['sum'],
        'FLAG_DPD_31_60_SUM':['sum'],
        'FLAG_DPD_61_90_SUM':['sum'],
        'FLAG_DPD_91_120_SUM':['sum']
        
    }
    # Bureau and bureau_balance categorical features
    #cat_aggregations = {}
    #for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    #for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    
    credit_card_agg = credit_card_df.groupby('SK_ID_CURR').agg({**num_aggregations})
    credit_card_agg.columns = pd.Index(['CREDIT_CARD_' + e[0] + "_" + e[1].upper() for e in credit_card_agg.columns.tolist()])
    bureau_agg=bureau_agg.merge(right=credit_card_agg.reset_index(),how='left',on='SK_ID_CURR')
    del credit_card_df
    gc.collect()
    
    
    consumer_agg = consumer_df.groupby('SK_ID_CURR').agg({**num_aggregations})
    consumer_agg.columns = pd.Index(['CONSUMER_' + e[0] + "_" + e[1].upper() for e in consumer_agg.columns.tolist()])
    bureau_agg=bureau_agg.merge(right=consumer_agg.reset_index(),how='left',on='SK_ID_CURR')
    del consumer_df
    gc.collect()
    
    
    car_agg = car_df.groupby('SK_ID_CURR').agg({**num_aggregations})
    car_agg.columns = pd.Index(['CAR_' + e[0] + "_" + e[1].upper() for e in car_agg.columns.tolist()])
    bureau_agg=bureau_agg.merge(right=car_agg.reset_index(),how='left',on='SK_ID_CURR')
    del car_df
    gc.collect()
    
    mortgage_agg = mortgage_df.groupby('SK_ID_CURR').agg({**num_aggregations})
    mortgage_agg.columns = pd.Index(['MORTGAGE_' + e[0] + "_" + e[1].upper() for e in mortgage_agg.columns.tolist()])
    bureau_agg=bureau_agg.merge(right=mortgage_agg.reset_index(),how='left',on='SK_ID_CURR')
    del mortgage_df
    gc.collect()
    
    microloan_agg = microloan_df.groupby('SK_ID_CURR').agg({**num_aggregations})
    microloan_agg.columns = pd.Index(['MICROLOAN_' + e[0] + "_" + e[1].upper() for e in microloan_agg.columns.tolist()])
    bureau_agg=bureau_agg.merge(right=microloan_agg.reset_index(),how='left',on='SK_ID_CURR')
    del microloan_df
    gc.collect()
    
    # Bureau: Active credits - using only numerical aggregations
    bureau_agg['ACTIVE_TO_TOTAL']=bureau_agg['BURO_FLAG_ACTIVE_SUM']/bureau_agg['BURO_SK_ID_BUREAU_NUNIQUE']
    bureau_agg['FLAG_MORE_THAN_ONE_CURRENCY']=bureau_agg['BURO_CREDIT_CURRENCY_NUNIQUE'].apply(lambda x:1 if x>1 else 0)
    bureau_agg['AMT_CREDIT_TO_DEBT']=bureau_agg['BURO_AMT_CREDIT_SUM_SUM']/(bureau_agg['BURO_AMT_CREDIT_SUM_DEBT_SUM']+1)
    bureau_agg['CREDIT_TO_OVERDUE_TOTAL']=bureau_agg['BURO_AMT_CREDIT_SUM_SUM']/(bureau_agg['BURO_AMT_CREDIT_SUM_OVERDUE_SUM']+1)
    bureau_agg
    
    
    active = bureau[bureau['CREDIT_ACTIVE'] == 'Active']
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACT_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE'] == 'Closed']
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLS_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    print('done bureau')
    return bureau_agg    

def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('../input/POS_CASH_balance.csv', nrows = num_rows)
    pos.replace(365243, np.nan, inplace= True)
    pos.replace('XAN', np.nan, inplace= True)
    pos.replace('XNA', np.nan, inplace= True)
    
    pos['FLAG_ACTIVE']=pos['NAME_CONTRACT_STATUS'].apply(lambda x:1 if x=='Active' else 0)
    pos['FLAG_COMPLETED']=pos['NAME_CONTRACT_STATUS'].apply(lambda x:1 if x=='Completed' else 0)
    
    pos_active=pos[pos['FLAG_ACTIVE']==1]
    pos_completed=pos[pos['FLAG_COMPLETED']==1]
    # Features
    aggregations = { 
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        #'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean','min','sum'],
        'FLAG_ACTIVE':['sum'],
        'FLAG_COMPLETED':['sum'],
        'CNT_INSTALMENT':['sum','min','max'],
        'CNT_INSTALMENT_FUTURE':['sum','min','max']
        
        
    }

    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    
    pos_active_agg=pos_active.groupby('SK_ID_CURR').agg(aggregations)
    pos_active_agg.columns = pd.Index(['POS_ACTIVE_' + e[0] + "_" + e[1].upper() for e in pos_active_agg.columns.tolist()])
    pos_agg=pos_agg.merge(right=pos_active_agg,how='left',on='SK_ID_CURR')
    del pos_active_agg,pos_active
    gc.collect()
    pos_completed_agg=pos_completed.groupby('SK_ID_CURR').agg(aggregations)
    pos_completed_agg.columns = pd.Index(['POS_COMPLETED_' + e[0] + "_" + e[1].upper() for e in pos_completed_agg.columns.tolist()])
    pos_agg=pos_agg.merge(right=pos_completed_agg,how='left',on='SK_ID_CURR')
    del pos_completed_agg,pos_completed
    gc.collect()
    print('done pos')
    return pos_agg
    
  
  


# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('../input/application_train.csv', nrows=num_rows )
    test_df = pd.read_csv('../input/application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
    appart=[f for f in df.columns if ('_MODE' in f) | ('_AVG' in f) & ('APARTMENTS_AVG' not in f)]
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['ORGANIZATION_TYPE']=df['ORGANIZATION_TYPE'].str.replace(r'(^.*Business.*$)', 'Business')
    df['ORGANIZATION_TYPE']=df['ORGANIZATION_TYPE'].str.replace(r'(^.*Trade.*$)', 'Trade')
    df['ORGANIZATION_TYPE']=df['ORGANIZATION_TYPE'].str.replace(r'(^.*Transport.*$)', 'Transport')
    df['ORGANIZATION_TYPE']=df['ORGANIZATION_TYPE'].str.replace(r'(^.*Industry.*$)', 'Industry')
    df['ORGANIZATION_TYPE']=df['ORGANIZATION_TYPE'].str.replace(r'(^.*Business.*$)', 'Business')
    df['ORGANIZATION_TYPE']=df['ORGANIZATION_TYPE'].str.replace(r'(^.*Kindergarten.*$)', 'School')
    df['OCCUPATION_TYPE']=df['OCCUPATION_TYPE'].str.replace('Managers','Core staff')
    df['OCCUPATION_TYPE']=df['OCCUPATION_TYPE'].str.replace('High skill tech staff','Core staff');
    df['OCCUPATION_TYPE']=df['OCCUPATION_TYPE'].str.replace('Low-skill Laborers','Laborers')
    df['OCCUPATION_TYPE']=df['OCCUPATION_TYPE'].str.replace('Cooking staff','Laborers')
    df['OCCUPATION_TYPE']=df['OCCUPATION_TYPE'].str.replace('Cleaning staff','Laborers')
    df['OCCUPATION_TYPE']=df['OCCUPATION_TYPE'].str.replace('Waiters/barmen staff','Laborers')
    df['OCCUPATION_TYPE']=df['OCCUPATION_TYPE'].str.replace('Drivers','Laborers')

    df['NAME_INCOME_TYPE']=df['NAME_INCOME_TYPE'].apply(lambda x:'Other' if (x=='Unemployed')|(x=='Student')|(x=='Businessman')|(x=='Maternity leave') else x)
    df.drop(df[df['NAME_FAMILY_STATUS']=='Unknown'].index,inplace=True)
    
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

    df['FLAG_BUSINESS']=df['ORGANIZATION_TYPE'].apply(lambda x:1 if (x=='Business')|(x=='Self_employed')  else  0)
    
    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY']+1)
    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE']+1)
    df['NEW_DOC_IND_AVG'] = df[docs].mean(axis=1)
    df['NEW_DOC_IND_STD'] = df[docs].std(axis=1)
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_LIVE_IND_STD'] = df[live].std(axis=1)
    df['NEW_LIVE_IND_KURT'] = df[live].kurtosis(axis=1)
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / (df['DAYS_BIRTH']+1)
    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / (df['DAYS_EMPLOYED']+1)
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / (df['DAYS_BIRTH']+1)
    df['NEW_PHONE_TO_EMPLOY_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / (df['DAYS_EMPLOYED']+1)
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL']+1)
    
    #df=df[cols]
    # Categorical features with Binary encode (0 or 1; two categories)
    cat_feats=[f for f in df.columns if df[f].dtypes=='object']
    for bin_feature in cat_feats:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    #df, cat_cols = one_hot_encoder(df, nan_as_category)
    docs.remove('FLAG_DOCUMENT_3')
    live.remove('FLAG_WORK_PHONE')
    appart.remove('TOTALAREA_MODE')
    df.drop(docs,axis=1,inplace=True)
    df.drop(live,axis=1,inplace=True)
    df.drop(appart,axis=1,inplace=True)
    print('done_application')
    
    del test_df
    gc.collect()
    return df
    
def ins(num_rows=None,nan_as_category=True,from_df=False):
    ins=pd.read_csv('../input/installments_payments.csv',nrows=num_rows)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['FLAG_LATE_PAYMENT'] = ins['DPD'].apply(lambda x: 1 if x > 0 else 0)
    mask=(ins['DAYS_ENTRY_PAYMENT']<ins['DAYS_INSTALMENT']) & (ins['AMT_PAYMENT']<ins['AMT_INSTALMENT'])
    ins['FLAG_MISSED_PAYMENT']=np.where(mask,1,0)

    a=ins[ins['FLAG_LATE_PAYMENT']==1]
    b=ins[ins['FLAG_MISSED_PAYMENT']==1]
    if from_df:
        return a,b
    else:
        
        aggregations = {
                'FLAG_MISSED_PAYMENT':['sum'],
                'FLAG_LATE_PAYMENT':['sum'],
                'NUM_INSTALMENT_VERSION': ['nunique'],
                'DPD': ['max','mean','sum'],
                'DBD': ['max','mean','sum'],
                'PAYMENT_PERC': ['max','mean','sum','var'],
                'PAYMENT_DIFF': ['max','mean','sum','var'],
                'AMT_INSTALMENT': ['max','min','mean','sum'],
                'AMT_PAYMENT': ['max','min','mean','sum'],
                'DAYS_ENTRY_PAYMENT': ['min','max','mean']
            }

        ins_agg_=ins.groupby('SK_ID_PREV').agg(aggregations)
        ins_agg_.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg_.columns.tolist()])


        ins_agg_=ins_agg_.merge(right=ins[ins.groupby(['SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].transform(min) == ins['NUM_INSTALMENT_NUMBER']][['SK_ID_PREV','AMT_PAYMENT','FLAG_LATE_PAYMENT']]
                                .rename(columns={'AMT_PAYMENT':'AMT_FIRST_PAYMENT','FLAG_LATE_PAYMENT':'FLAG_FIRST_LATE_PAYMENT'})
                                ,how='left',on='SK_ID_PREV')
        ins_agg_=ins_agg_.merge(right=ins[ins.groupby(['SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].transform(max) == ins['NUM_INSTALMENT_NUMBER']][['SK_ID_PREV','AMT_PAYMENT']].rename(columns={'AMT_PAYMENT':'AMT_LAST_PAYMENT'}),how='left',on='SK_ID_PREV')
        ins_agg_=ins_agg_.merge(right=ins.groupby('SK_ID_PREV').size().reset_index().rename(columns={0:'PAYMENTS_COUNT'}),how='left',on='SK_ID_PREV')
        
        
        ins_agg_=ins_agg_.merge(right=ins[ins.groupby(['SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].transform(min) == ins['NUM_INSTALMENT_NUMBER']][['SK_ID_PREV','FLAG_MISSED_PAYMENT']]
                                .rename(columns={'FLAG_MISSED_PAYMENT':'FLAG_FIRST_MISSED_PAYMENT'})
                                ,how='left',on='SK_ID_PREV')
        ins_agg_['PERC_LATE_PAYMENTS']=ins_agg_['INSTAL_FLAG_LATE_PAYMENT_SUM']/ins_agg_['PAYMENTS_COUNT']
        ins_agg_['PERC_MISSED_PAYMENTS']=ins_agg_['INSTAL_FLAG_MISSED_PAYMENT_SUM']/ins_agg_['PAYMENTS_COUNT']

        del ins
        gc.collect()
        print('done_ins')
        return ins_agg_


def prev_ins(num_rows=None):
    prev=pd.read_csv('../input/previous_application.csv',nrows=num_rows)
    ins_=ins(num_rows)
    prev=prev.merge(right=ins_.reset_index(),how='left',on='SK_ID_PREV')

    del ins_
    gc.collect()
    #cat_cols=[f for f in prev.columns if prev[f].dtypes=='object']
    #for f in cat_cols:
        #prev[f]=pd.factorize(prev[f])[0]

    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)

    prev['FLAG_X_SELL']=prev['PRODUCT_COMBINATION'].apply(lambda x:1 if (x=='Cash X-Sell: middle')|(x=='Cash X-Sell: low')|(x=='Cash X-Sell: high')|(x=='Card X-Sell') else 0)
    #prev['TOTAL_PREV_LOANS']=prev.groupby('SK_ID_CURR').size()
    prev['FLAG_REFUSED_LOANS']=prev['NAME_CONTRACT_STATUS'].apply(lambda x:1 if x=='Refused' else 0)
    #prev['PERC_REJECTED_LOANS']= prev['NUM_REFUSED_LOANS']/prev['TOTAL_PREV_LOANS']
    prev['FLAG_APPROVED_LOANS']=prev['NAME_CONTRACT_STATUS'].apply(lambda x:1 if x=='Approved' else 0)
    #prev['PERC_APPROVED_LOANS']= prev['NUM_APPROVED_LOANS']/prev['TOTAL_PREV_LOANS']
    prev['FLAG_WITHOUT_INTEREST']=prev['PRODUCT_COMBINATION'].str.replace(r'(^.*without interest.*$)', 'without_interest').apply(lambda x:1 if x=='without_interest' else 0)
    prev['FLAG_CASH_LOAN']=prev['NAME_CONTRACT_TYPE'].apply(lambda x:1 if x=='Cash loans' else 0)
    prev['FLAG_CONSUMER_LOAN']=prev['NAME_CONTRACT_TYPE'].apply(lambda x:1 if x=='Consumer loans' else 0)
    prev['FLAG_REVOLVING_LOAN']=prev['NAME_CONTRACT_TYPE'].apply(lambda x:1 if x=='Revolving loans' else 0)

    prev['CREDIT_TO_ANNUITY']=prev['AMT_CREDIT']/(prev['AMT_ANNUITY']+1)
    prev['CREDIT_TO_GOODS']=prev['AMT_CREDIT']/(prev['AMT_GOODS_PRICE']+1)
    prev['FIRST_PAYMENT_TO_CREDIT']=prev['AMT_FIRST_PAYMENT']/(prev['AMT_CREDIT']+1)
    prev['LAST_PAYMENT_TO_CREDIT']=prev['AMT_LAST_PAYMENT']/(prev['AMT_CREDIT']+1)
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / (prev['AMT_CREDIT']+1)
    prev['INTEREST_RATE']=(prev['INSTAL_AMT_PAYMENT_SUM']-prev['AMT_CREDIT'])/(prev['AMT_CREDIT']+1)
    prev['INTEREST_AMT']=prev['INSTAL_AMT_PAYMENT_SUM']-prev['AMT_CREDIT']
    prev['INTEREST_AMT_TO_DOWN_PAYMENT_AMT']=prev['INTEREST_AMT']/(prev['AMT_DOWN_PAYMENT']+1)
    prev['AMT_CREDIT_TO_INSTALMENT_COUNT']=prev['AMT_CREDIT']/prev['PAYMENTS_COUNT']
    prev['INTEREST_AMT_TO_FIRST_PAYMENT']=prev['INTEREST_AMT']/(prev['AMT_FIRST_PAYMENT']+1)
    prev['INTEREST_AMT_TO_LAST_PAYMENT']=prev['INTEREST_AMT']/(prev['AMT_LAST_PAYMENT']+1)
    prev['INTEREST_AMT_TO_LAST_PAYMENT']=prev['INTEREST_AMT']/(prev['AMT_LAST_PAYMENT']+1)
    prev['LAST_DUR_FIRST_DUE_DIFF']=prev['DAYS_LAST_DUE']-prev['DAYS_FIRST_DUE']
    prev['DAYS_DECISION_DAYS_FIRST_DUE_DIFF']=prev['DAYS_DECISION']-prev['DAYS_FIRST_DUE']
    prev['DOWN_PAYMENT_TO_CREDIT_RATIO']=prev['AMT_DOWN_PAYMENT']/(prev['AMT_CREDIT']+1)
    prev['DOWN_PAYMENT_TO_AMT_GOODS']=prev['AMT_DOWN_PAYMENT']/(prev['AMT_GOODS_PRICE']+1)
    prev['FLAG_LATEST_ACC']=prev['DAYS_DECISION'].apply(lambda x:1 if x>=-182 else 0)
    latest_acc_df=prev[prev['FLAG_LATEST_ACC']==1]
    
    
    a=['mean','min','max','sum']
    num_aggregations = {
        'DOWN_PAYMENT_TO_AMT_GOODS':a,
        'SELLERPLACE_AREA':a,
        'FLAG_LATEST_ACC':['sum'],
        'DOWN_PAYMENT_TO_CREDIT_RATIO':a,
        'NFLAG_INSURED_ON_APPROVAL':['sum'],
        'DAYS_DECISION_DAYS_FIRST_DUE_DIFF':a,
        'CREDIT_TO_ANNUITY':a,
        'CREDIT_TO_GOODS':a,
        'PRODUCT_COMBINATION':['nunique'],
        'SK_ID_PREV':['nunique'],
        'FLAG_CASH_LOAN':['sum'],
        'FLAG_CONSUMER_LOAN':['sum'],
        'FLAG_REVOLVING_LOAN':['sum'],
        'FLAG_X_SELL':['sum'],
        'FLAG_WITHOUT_INTEREST':['sum'],
        'FLAG_REFUSED_LOANS':['sum'],
        'FLAG_APPROVED_LOANS':['sum'],
        'NAME_YIELD_GROUP':['nunique'],
        'NAME_PORTFOLIO':['nunique'],
        'NAME_CLIENT_TYPE':['nunique'],
        'NAME_CONTRACT_TYPE':['nunique'],
        'LAST_DUR_FIRST_DUE_DIFF':['mean','min','max','sum','var'],
        'INTEREST_AMT':a,
        'INTEREST_AMT_TO_DOWN_PAYMENT_AMT':a,
        'AMT_CREDIT_TO_INSTALMENT_COUNT':a,
        'DAYS_FIRST_DRAWING':a,
        'DAYS_FIRST_DUE':a,
        'DAYS_LAST_DUE_1ST_VERSION':a,
        'DAYS_LAST_DUE':a,
        'DAYS_TERMINATION':['min','max'],
        'FIRST_PAYMENT_TO_CREDIT':a,
        'INTEREST_RATE':a,
        'APP_CREDIT_PERC':a,
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],

        'INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE': a,
        'INSTAL_DPD_MEAN': a,'INSTAL_DPD_SUM': a,'INSTAL_DPD_MAX': a,
        'INSTAL_DBD_MAX': a,'INSTAL_DBD_MEAN': a,'INSTAL_DBD_SUM': a,
        'INSTAL_FLAG_LATE_PAYMENT_SUM':a,
        'INSTAL_PAYMENT_PERC_MAX': a,'INSTAL_PAYMENT_PERC_MEAN': a,'INSTAL_PAYMENT_PERC_SUM': a,'INSTAL_PAYMENT_PERC_VAR': a,
        'INSTAL_PAYMENT_DIFF_MAX': a,'INSTAL_PAYMENT_DIFF_MEAN': a,'INSTAL_PAYMENT_DIFF_SUM': a,'INSTAL_PAYMENT_DIFF_VAR': a,
        'INSTAL_AMT_INSTALMENT_MAX': a,'INSTAL_AMT_INSTALMENT_MIN': a,'INSTAL_AMT_INSTALMENT_MEAN': a,'INSTAL_AMT_INSTALMENT_SUM': a,
        'INSTAL_AMT_PAYMENT_MAX': a,'INSTAL_AMT_PAYMENT_MIN': a,'INSTAL_AMT_PAYMENT_MEAN': a,'INSTAL_AMT_PAYMENT_SUM': a,
        'INSTAL_DAYS_ENTRY_PAYMENT_MIN': ['max','mean','sum'],'INSTAL_DAYS_ENTRY_PAYMENT_MAX': ['min','mean','sum'],'INSTAL_DAYS_ENTRY_PAYMENT_MEAN': a,
        'PERC_LATE_PAYMENTS':a,
        'PERC_MISSED_PAYMENTS':a,
        'PAYMENTS_COUNT':a ,
        'AMT_FIRST_PAYMENT':a,
        'INTEREST_AMT_TO_FIRST_PAYMENT':a,
        'FIRST_PAYMENT_TO_CREDIT':a,
        'AMT_LAST_PAYMENT':a,
        'INTEREST_AMT_TO_LAST_PAYMENT':a,
        'LAST_PAYMENT_TO_CREDIT':a,
        'FLAG_FIRST_LATE_PAYMENT':['sum'],
        'FLAG_FIRST_MISSED_PAYMENT':['sum'],
        'DOWN_PAYMENT_TO_AMT_GOODS':a,
        'INSTAL_FLAG_MISSED_PAYMENT_SUM':a
    }
    
    latest_agg={'INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE': a,
        'INSTAL_DPD_MEAN': a,'INSTAL_DPD_SUM': a,'INSTAL_DPD_MAX': a,
        'INSTAL_DBD_MAX': a,'INSTAL_DBD_MEAN': a,'INSTAL_DBD_SUM': a,
        'INSTAL_FLAG_LATE_PAYMENT_SUM':a,
        'INSTAL_PAYMENT_PERC_MAX': a,'INSTAL_PAYMENT_PERC_MEAN': a,'INSTAL_PAYMENT_PERC_SUM': a,'INSTAL_PAYMENT_PERC_VAR': a,
        'INSTAL_PAYMENT_DIFF_MAX': a,'INSTAL_PAYMENT_DIFF_MEAN': a,'INSTAL_PAYMENT_DIFF_SUM': a,'INSTAL_PAYMENT_DIFF_VAR': a,
        'INSTAL_AMT_INSTALMENT_MAX': a,'INSTAL_AMT_INSTALMENT_MIN': a,'INSTAL_AMT_INSTALMENT_MEAN': a,'INSTAL_AMT_INSTALMENT_SUM': a,
        'INSTAL_AMT_PAYMENT_MAX': a,'INSTAL_AMT_PAYMENT_MIN': a,'INSTAL_AMT_PAYMENT_MEAN': a,'INSTAL_AMT_PAYMENT_SUM': a,
        'INSTAL_DAYS_ENTRY_PAYMENT_MIN': ['max','mean','sum'],'INSTAL_DAYS_ENTRY_PAYMENT_MAX': ['min','mean','sum'],'INSTAL_DAYS_ENTRY_PAYMENT_MEAN': a,
        'PERC_LATE_PAYMENTS':a,
        'PERC_MISSED_PAYMENTS':a,
        'PAYMENTS_COUNT':a ,
        'AMT_FIRST_PAYMENT':a,
        'INTEREST_AMT_TO_FIRST_PAYMENT':a,
        'FIRST_PAYMENT_TO_CREDIT':a,
        'AMT_LAST_PAYMENT':a,
        'INTEREST_AMT_TO_LAST_PAYMENT':a,
        'LAST_PAYMENT_TO_CREDIT':a,
        'FLAG_FIRST_LATE_PAYMENT':['sum'],
        'FLAG_FIRST_MISSED_PAYMENT':['sum'],
        'DOWN_PAYMENT_TO_AMT_GOODS':a,
        'INSTAL_FLAG_MISSED_PAYMENT_SUM':a}

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    prev_agg['PERC_REFUSED_LOANS']= prev_agg['PREV_FLAG_REFUSED_LOANS_SUM']/prev_agg['PREV_SK_ID_PREV_NUNIQUE']
    prev_agg['PERC_APPROVED_LOANS']= prev_agg['PREV_FLAG_APPROVED_LOANS_SUM']/prev_agg['PREV_SK_ID_PREV_NUNIQUE']
    
    prev_agg['AMT_TOTAL_RETURNED_TO_AMT_EXPECTED']=prev_agg['PREV_INSTAL_AMT_PAYMENT_SUM_SUM']/prev_agg['PREV_INSTAL_AMT_INSTALMENT_SUM_SUM']
    prev_agg['TOTAL_KHARAB_PAYMENTS']=prev_agg['PREV_INSTAL_FLAG_LATE_PAYMENT_SUM_SUM']+prev_agg['PREV_INSTAL_FLAG_MISSED_PAYMENT_SUM_SUM']
    prev_agg['TOTAL_KHARAB_PAYMENTS_TO_TOTAL_PAYMENTS']=prev_agg['TOTAL_KHARAB_PAYMENTS']/prev_agg['PREV_PAYMENTS_COUNT_SUM']
    prev_agg['FIRST_LATE_PAYMENT_TO_TOTAL_LATE_PAYMENTS']=prev_agg['PREV_FLAG_FIRST_LATE_PAYMENT_SUM']/prev_agg['PREV_INSTAL_FLAG_LATE_PAYMENT_SUM_SUM']
    prev_agg['FIRST_MISSED_PAYMENT_TO_TOTAL_LATE_PAYMENTS']=prev_agg['PREV_FLAG_FIRST_LATE_PAYMENT_SUM']/prev_agg['PREV_INSTAL_FLAG_LATE_PAYMENT_SUM_SUM']
    prev_agg['PERC_LATE_PAYMENTS_TOTAL']=prev_agg['PREV_INSTAL_FLAG_LATE_PAYMENT_SUM_SUM']/prev_agg['PREV_PAYMENTS_COUNT_SUM']
    prev_agg['PERC_MISSED_PAYMENTS_TOTAL']=prev_agg['PREV_INSTAL_FLAG_MISSED_PAYMENT_SUM_SUM']/prev_agg['PREV_PAYMENTS_COUNT_SUM']
    
    columns_list=[]
    agg_={
        'INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE': a,
        'INSTAL_DPD_MEAN': a,'INSTAL_DPD_SUM': a,'INSTAL_DPD_MAX': a,
        'INSTAL_DBD_MAX': a,'INSTAL_DBD_MEAN': a,'INSTAL_DBD_SUM': a,
        'INSTAL_FLAG_LATE_PAYMENT_SUM':a,
        'INSTAL_PAYMENT_PERC_MAX': a,'INSTAL_PAYMENT_PERC_MEAN': a,'INSTAL_PAYMENT_PERC_SUM': a,'INSTAL_PAYMENT_PERC_VAR': a,
        'INSTAL_PAYMENT_DIFF_MAX': a,'INSTAL_PAYMENT_DIFF_MEAN': a,'INSTAL_PAYMENT_DIFF_SUM': a,'INSTAL_PAYMENT_DIFF_VAR': a,
        'INSTAL_AMT_INSTALMENT_MAX': a,'INSTAL_AMT_INSTALMENT_MIN': a,'INSTAL_AMT_INSTALMENT_MEAN': a,'INSTAL_AMT_INSTALMENT_SUM': a,
        'INSTAL_AMT_PAYMENT_MAX': a,'INSTAL_AMT_PAYMENT_MIN': a,'INSTAL_AMT_PAYMENT_MEAN': a,'INSTAL_AMT_PAYMENT_SUM': a,
        'INSTAL_DAYS_ENTRY_PAYMENT_MIN': ['max','mean','sum'],'INSTAL_DAYS_ENTRY_PAYMENT_MAX': ['min','mean','sum'],'INSTAL_DAYS_ENTRY_PAYMENT_MEAN': a,
        'PERC_LATE_PAYMENTS':a,'PAYMENTS_COUNT':a ,
        'AMT_FIRST_PAYMENT':a,
        'INTEREST_AMT_TO_FIRST_PAYMENT':a,
        'FIRST_PAYMENT_TO_CREDIT':a,
        'AMT_LAST_PAYMENT':a,
        'INTEREST_AMT_TO_LAST_PAYMENT':a,
        'LAST_PAYMENT_TO_CREDIT':a,
        'FLAG_FIRST_LATE_PAYMENT':['sum']
        }
    
    approved = prev[prev['NAME_CONTRACT_STATUS'] == 'Approved']
    prev_agg=prev_agg.merge(right=approved.groupby('SK_ID_CURR')['AMT_CREDIT'].max().reset_index().rename(columns={'AMT_CREDIT':'APPROVED_MAX_CREDIT'}),how='left',on='SK_ID_CURR')
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    cols = approved_agg.columns.tolist()
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    #Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS'] == 'Refused']
    prev_agg=prev_agg.merge(right=approved.groupby('SK_ID_CURR')['AMT_CREDIT'].max().reset_index().rename(columns={'AMT_CREDIT':'MAX_CREDIT'}),how='left',on='SK_ID_CURR')
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg
    gc.collect()

    for e in cols:
        prev_agg['NEW_RATIO_PREV_' + e[0] + "_" + e[1].upper()] = prev_agg['APPROVED_' + e[0] + "_" + e[1].upper()] / prev_agg['REFUSED_' + e[0] + "_" + e[1].upper()]
    
    
    del prev
    gc.collect()
    print('done_prev_ins')
    return prev_agg


def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('../input/credit_card_balance.csv', nrows = num_rows)
    cc['FLAG_AMT_BALANCE_POS']=cc['AMT_BALANCE'].apply(lambda x:1 if x>0 else 0)
    cc['DRAWINGS_TO_CREDIT']=cc['AMT_DRAWINGS_CURRENT']/(cc['AMT_BALANCE']+1)
    cc['MIN_TO_PAYMENT']=cc['AMT_INST_MIN_REGULARITY']/(cc['AMT_PAYMENT_CURRENT']+1)
    cc['PAYMENT_TO_RECIVABLE']=cc['AMT_PAYMENT_CURRENT']/(cc['AMT_RECIVABLE']+1)
    cc['FLAG_ACTIVE']=cc['NAME_CONTRACT_STATUS'].apply(lambda x:1 if x=='Active' else 0)
    cc['DRAWINGS_TO_CREDIT_LIMIT']=cc['AMT_DRAWINGS_CURRENT']/(cc['AMT_CREDIT_LIMIT_ACTUAL']+1)
    agg_curr={
        'AMT_CREDIT_LIMIT_ACTUAL':['max','mean','min'],
        'AMT_BALANCE':['min','max','mean'],
        'DRAWINGS_TO_CREDIT_LIMIT':['min','max','mean'],
        'MONTHS_BALANCE':['min','size','max'],
        'FLAG_ACTIVE':['sum'],
        'AMT_PAYMENT_CURRENT':['sum','mean','max','min'],
        'AMT_RECIVABLE':['sum','mean','max','min'],
        'NAME_CONTRACT_STATUS':['nunique'],
        'SK_DPD':['sum','mean','max','min'],
        'MIN_TO_PAYMENT':['min','max','sum','mean'],
            'FLAG_AMT_BALANCE_POS':['sum'],
             'CNT_DRAWINGS_ATM_CURRENT':['mean'],
             'DRAWINGS_TO_CREDIT':['mean','sum','max','min'],
             'SK_ID_PREV':['nunique']}
    cc_agg=cc.groupby('SK_ID_CURR').agg({**agg_curr})
    cc_agg.columns=pd.Index(['CC_'+e[0]+'_'+e[1].upper() for e in cc_agg.columns.tolist()])
    cc_agg=cc_agg.reset_index()
    
    
    cc_agg['FLAG_LESS_CREDIT_UTLIZATION']=cc_agg['CC_DRAWINGS_TO_CREDIT_LIMIT_MEAN'].apply(lambda x:1 if x<0.06 else 0)
    cc['AMT_BALANCE_LAST_CREDIT']=cc[cc.groupby('SK_ID_CURR')['MONTHS_BALANCE'].transform(max)==cc['MONTHS_BALANCE']]['AMT_BALANCE']
    b=cc[np.isfinite(cc['AMT_BALANCE_LAST_CREDIT'])][['SK_ID_CURR','AMT_BALANCE_LAST_CREDIT']]
    b=b.groupby('SK_ID_CURR')['AMT_BALANCE_LAST_CREDIT'].max()
    cc_agg=cc_agg.merge(right=b.reset_index(),how='left',on='SK_ID_CURR')                                                                                                                    
    #cc_agg=cc_agg.merge(right=cc[cc.groupby('SK_ID_CURR')['MONTHS_BALANCE'].transform(max)==cc['MONTHS_BALANCE']][['SK_ID_CURR','AMT_BALANCE']].rename(columns={'AMT_BALANCE':'AMT_BALANCE_LAST_CREDIT'}),
    #                    how='left',on='SK_ID_CURR')
    del b
    gc.collect()
    cc_agg=cc_agg.set_index('SK_ID_CURR')
    cc_agg['CC_PAYMENT_COUNT']=cc.groupby('SK_ID_CURR').size()
    
    return cc_agg


def kfold_lightgbm(df, num_folds, stratified = False, debug= False):
# Divide in training/validation and test 
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        dtrain = lgb.Dataset(data=train_df[feats].iloc[train_idx], 
                             label=train_df['TARGET'].iloc[train_idx], 
                             free_raw_data=False, silent=True)
        dvalid = lgb.Dataset(data=train_df[feats].iloc[valid_idx], 
                             label=train_df['TARGET'].iloc[valid_idx], 
                             free_raw_data=False, silent=True)

        # LightGBM parameters found by Bayesian optimization
        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'nthread': 4,
            'learning_rate': 0.02,  # 02,
            'num_leaves': 20,
            'colsample_bytree': 0.9497036,
            'subsample': 0.8715623,
            'subsample_freq': 1,
            'max_depth': 8,
            'reg_alpha': 0.041545473,
            'reg_lambda': 0.0735294,
            'min_split_gain': 0.0222415,
            'min_child_weight': 60, # 39.3259775,
            'seed': 0,
            'verbose': -1,
            'metric': 'auc',
            'feature_fraction':0.3
        }

        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=10000,
            valid_sets=[dtrain, dvalid],
            early_stopping_rounds=100,
            verbose_eval=False
        )

        oof_preds[valid_idx] = clf.predict(dvalid.data)
        sub_preds += clf.predict(test_df[feats]) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = pd.Series(clf.feature_importance(importance_type='gain'))
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(dvalid.label, oof_preds[valid_idx])))
        del clf, dtrain, dvalid
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        sub_df = test_df[['SK_ID_CURR']].copy()
        sub_df['TARGET'] = sub_preds
        sub_df[['SK_ID_CURR', 'TARGET']].to_csv('abc.csv', index= False)
    display_importances(feature_importance_df)
    return feature_importance_df
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout
    plt.savefig('lgbm_importances01.png')

col_phele=[
'NEW_EXT_SOURCES_MEAN',
 'EXT_SOURCE_3',
 'NEW_CREDIT_TO_ANNUITY_RATIO',
 'NEW_CREDIT_TO_GOODS_RATIO',
 'EXT_SOURCE_2',
 'DAYS_BIRTH',
 'DAYS_EMPLOYED',
 'PREV_DAYS_LAST_DUE_1ST_VERSION_MAX',
 'AMT_ANNUITY',
 'EXT_SOURCE_1',
 'CODE_GENDER',
 'PREV_INSTAL_AMT_PAYMENT_MIN_SUM',
 'PREV_INSTAL_AMT_PAYMENT_MIN_MEAN',
 'NAME_EDUCATION_TYPE',
 'DAYS_ID_PUBLISH',
 'AMT_GOODS_PRICE',
 'PREV_INSTAL_DPD_MAX_MEAN',
 'NEW_ANNUITY_TO_INCOME_RATIO',
 'PREV_INSTAL_AMT_INSTALMENT_MIN_MIN',
 'PREV_DAYS_FIRST_DRAWING_MAX',
 'NEW_RATIO_PREV_DAYS_DECISION_MAX',
 'PREV_INSTAL_DPD_MAX_MIN',
 'PREV_DAYS_LAST_DUE_MAX',
 'DAYS_REGISTRATION',
 'OWN_CAR_AGE',
 'PREV_DAYS_LAST_DUE_1ST_VERSION_SUM',
 'REFUSED_AMT_DOWN_PAYMENT_MAX',
 'PREV_CNT_PAYMENT_MEAN',
 'PREV_APP_CREDIT_PERC_MEAN',
 'PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN',
 'REGION_POPULATION_RELATIVE',
 'NEW_PHONE_TO_EMPLOY_RATIO',
 'NEW_CAR_TO_EMPLOY_RATIO',
 'PREV_INSTAL_DPD_SUM_MAX',
 'PREV_INSTAL_DPD_MAX_SUM',
 'NAME_FAMILY_STATUS',
 'NEW_DOC_IND_KURT',
 'ORGANIZATION_TYPE',
 'PREV_APP_CREDIT_PERC_MIN',
 'NEW_SCORES_STD',
 'NEW_CREDIT_TO_INCOME_RATIO',
 'APPROVED_APP_CREDIT_PERC_MIN',
 'PREV_INSTAL_PAYMENT_PERC_SUM_MEAN',
 'PREV_INTEREST_RATE_MIN',
 'DAYS_LAST_PHONE_CHANGE',
 'PREV_INSTAL_PAYMENT_PERC_SUM_SUM',
 'PREV_INSTAL_FLAG_LATE_PAYMENT_SUM_MEAN',
 'NAME_INCOME_TYPE',
 'REFUSED_CNT_PAYMENT_MEAN',
 'DEF_30_CNT_SOCIAL_CIRCLE',
 'APPROVED_DAYS_DECISION_MAX',
 'PREV_INSTAL_PAYMENT_DIFF_MEAN_MEAN',
 'PREV_INSTAL_AMT_PAYMENT_SUM_SUM',
 'PREV_DAYS_DECISION_MAX',
 'PREV_INSTAL_DBD_MAX_MEAN',
 'PREV_INSTAL_DPD_SUM_SUM',
 'PREV_INSTAL_DPD_SUM_MIN',
 'AMT_REQ_CREDIT_BUREAU_QRT',
 'REGION_RATING_CLIENT_W_CITY',
 'PREV_INSTAL_AMT_INSTALMENT_MAX_MEAN',
 'AMT_INCOME_TOTAL',
 'NEW_RATIO_PREV_CNT_PAYMENT_SUM',
 'NEW_RATIO_PREV_DAYS_DECISION_MIN',
 'AMT_REQ_CREDIT_BUREAU_YEAR',
 'PREV_INSTAL_PAYMENT_PERC_MEAN_MEAN',
 'PREV_INSTAL_DPD_MEAN_MAX',
 'PREV_INSTAL_AMT_INSTALMENT_SUM_MIN',
 'REFUSED_APP_CREDIT_PERC_MEAN',
 'PREV_DAYS_DECISION_MIN',
 'PREV_AMT_DOWN_PAYMENT_MEAN',
 'REFUSED_HOUR_APPR_PROCESS_START_MEAN',
 'PREV_INSTAL_DAYS_ENTRY_PAYMENT_MEAN_MAX',
 'REFUSED_AMT_CREDIT_MIN',
 'NEW_INC_BY_ORG',
 'PREV_INSTAL_PAYMENT_DIFF_MAX_MAX',
 'PREV_INSTAL_AMT_PAYMENT_MEAN_MIN',
 'PREV_INSTAL_AMT_INSTALMENT_MAX_MIN',
 'APPROVED_AMT_CREDIT_MIN',
 'PREV_RATE_DOWN_PAYMENT_MEAN',
 'PREV_INSTAL_DBD_MAX_MIN',
 'PREV_INSTAL_DPD_MEAN_MIN',
 #'PRODUCT_COMBINATION',
 'REFUSED_APP_CREDIT_PERC_MIN',
 'REG_CITY_NOT_LIVE_CITY',
 'REFUSED_CNT_PAYMENT_SUM',
 'HOUR_APPR_PROCESS_START',
 'PREV_AMT_ANNUITY_MAX',
 'PREV_AMT_GOODS_PRICE_MIN',
 'PREV_INSTAL_AMT_PAYMENT_SUM_MEAN',
 'PREV_INSTAL_DPD_SUM_MEAN',
 'PREV_AMT_ANNUITY_MIN',
 'PREV_INSTAL_DBD_MAX_SUM',
 'PREV_INSTAL_AMT_INSTALMENT_MAX_MAX',
 'PREV_DAYS_LAST_DUE_MEAN',
 'APPROVED_AMT_ANNUITY_MIN',
 'PREV_INSTAL_PAYMENT_DIFF_MEAN_MAX',
 'PREV_AMT_GOODS_PRICE_MEAN',
 'PREV_INSTAL_PAYMENT_DIFF_SUM_MEAN',
 'PREV_INSTAL_AMT_PAYMENT_MEAN_MEAN',
 'REFUSED_AMT_CREDIT_MAX',
 'WEEKDAY_APPR_PROCESS_START',
 'PREV_CNT_PAYMENT_SUM',
 'PREV_INSTAL_PAYMENT_DIFF_MEAN_SUM',
 'NEW_RATIO_PREV_AMT_CREDIT_MAX',
 'NEW_RATIO_PREV_AMT_APPLICATION_MIN',
 'NEW_RATIO_PREV_AMT_ANNUITY_MAX',
 'NEW_RATIO_PREV_HOUR_APPR_PROCESS_START_MEAN',
 'APARTMENTS_AVG',
 'NEW_RATIO_PREV_AMT_ANNUITY_MIN',
 'NEW_RATIO_PREV_HOUR_APPR_PROCESS_START_MIN',
 'PREV_AMT_CREDIT_MIN',
 'PREV_HOUR_APPR_PROCESS_START_MAX',
 'PREV_INSTAL_PAYMENT_PERC_VAR_MEAN',
 'NEW_DOC_IND_AVG',
 'YEARS_BEGINEXPLUATATION_MEDI',
 'PREV_INSTAL_DPD_MEAN_MEAN',
 'NEW_RATIO_PREV_HOUR_APPR_PROCESS_START_MAX',
 'REFUSED_RATE_DOWN_PAYMENT_MAX',
 'PREV_AMT_CREDIT_MAX',
 'SK_ID_CURR',
 'TARGET',
 'PREV_PERC_LATE_PAYMENTS_MEAN',
 'PREV_PERC_MISSED_PAYMENTS_MEAN','PREV_PERC_MISSED_PAYMENTS_MIN','PREV_PERC_MISSED_PAYMENTS_MAX',
 'PREV_PAYMENTS_COUNT_MIN',
 'PREV_AMT_CREDIT_TO_INSTALMENT_COUNT_MAX',
 'PREV_INTEREST_AMT_MEAN',
 'PREV_INTEREST_AMT_MIN',
 'PREV_INTEREST_AMT_MAX',
 'PREV_INTEREST_AMT_TO_DOWN_PAYMENT_AMT_MEAN',
 'PREV_INTEREST_AMT_TO_DOWN_PAYMENT_AMT_MIN',
 'PREV_AMT_FIRST_PAYMENT_MEAN',
 'PREV_AMT_FIRST_PAYMENT_MIN',
 'PREV_AMT_FIRST_PAYMENT_MAX',
 'PREV_INTEREST_AMT_TO_FIRST_PAYMENT_MEAN',
 'PREV_INTEREST_AMT_TO_FIRST_PAYMENT_MAX',
 'PREV_INTEREST_AMT_TO_FIRST_PAYMENT_MIN',
 'PREV_INTEREST_AMT_TO_LAST_PAYMENT_MEAN',
 'PREV_INTEREST_AMT_TO_LAST_PAYMENT_MAX',
 'PREV_INTEREST_AMT_TO_LAST_PAYMENT_MIN',
 'PREV_LAST_PAYMENT_TO_CREDIT_MEAN',
 'PREV_LAST_DUR_FIRST_DUE_DIFF_MEAN',
 'PREV_LAST_DUR_FIRST_DUE_DIFF_MIN',
'PREV_SK_ID_PREV_NUNIQUE','PREV_FLAG_X_SELL_SUM','PREV_FLAG_WITHOUT_INTEREST_SUM','PREV_FLAG_REFUSED_LOANS_SUM',
'PREV_FLAG_APPROVED_LOANS_SUM',
'PREV_NAME_YIELD_GROUP_NUNIQUE',
'PREV_NAME_PORTFOLIO_NUNIQUE',
'PREV_NAME_CLIENT_TYPE_NUNIQUE',
'PREV_NAME_CONTRACT_TYPE_NUNIQUE',
'PERC_REFUSED_LOANS','PERC_APPROVED_LOANS','OCCUPATION_TYPE','FLAG_BUSINESS','FLAG_DOCUMENT_3',
'PREV_NFLAG_INSURED_ON_APPROVAL_SUM',
'PREV_DAYS_DECISION_DAYS_FIRST_DUE_DIFF_MEAN','PREV_DAYS_DECISION_DAYS_FIRST_DUE_DIFF_MIN','PREV_DAYS_DECISION_DAYS_FIRST_DUE_DIFF_MAX','PREV_DAYS_DECISION_DAYS_FIRST_DUE_DIFF_SUM',
'PREV_CREDIT_TO_ANNUITY_MEAN','PREV_CREDIT_TO_ANNUITY_MIN','PREV_CREDIT_TO_ANNUITY_MAX','PREV_CREDIT_TO_ANNUITY_SUM',
'PREV_CREDIT_TO_GOODS_MEAN','PREV_CREDIT_TO_GOODS_MIN','PREV_CREDIT_TO_GOODS_MAX','PREV_CREDIT_TO_GOODS_SUM',
#'DAYS_LAST_LATE_PAYMENT',
'PREV_DOWN_PAYMENT_TO_CREDIT_RATIO_MEAN','PREV_DOWN_PAYMENT_TO_CREDIT_RATIO_MIN','PREV_DOWN_PAYMENT_TO_CREDIT_RATIO_MAX','PREV_DOWN_PAYMENT_TO_CREDIT_RATIO_SUM',
#'MAX_PAYMENT_DIFF',
#'LATEST_LATE_PAYMENT','DAYS_SEVERE_PAYMENT',
#'PAYMENT_DIFF_LATEST_PAYMENT',
'CREDIT_TO_APPROVED_MAX_CREDIT',
'APPROVED_MAX_CREDIT',
'PREV_DOWN_PAYMENT_TO_AMT_GOODS_MEAN','PREV_DOWN_PAYMENT_TO_AMT_GOODS_SUM','PREV_DOWN_PAYMENT_TO_AMT_GOODS_MIN','PREV_DOWN_PAYMENT_TO_AMT_GOODS_MAX',
'PREV_SELLERPLACE_AREA_MEAN','PREV_SELLERPLACE_AREA_SUM','PREV_SELLERPLACE_AREA_MIN','PREV_SELLERPLACE_AREA_MAX',
#'AMT_BALANCE_LAST_CREDIT',
#'CC_NAME_CONTRACT_STATUS_NUNIQUE',
#'CC_FLAG_AMT_BALANCE_POS_SUM',
#'CC_CNT_DRAWINGS_ATM_CURRENT_MEAN',
#'CC_PAYMENT_COUNT',
#'CC_DRAWINGS_TO_CREDIT_MEAN','CC_DRAWINGS_TO_CREDIT_MIN','CC_DRAWINGS_TO_CREDIT_MAX','CC_DRAWINGS_TO_CREDIT_SUM',
#'CC_SK_ID_PREV_NUNIQUE',

#'CC_MONTHS_BALANCE_SIZE','CC_MONTHS_BALANCE_MIN','CC_MONTHS_BALANCE_MAX',
#'CC_DRAWINGS_TO_CREDIT_LIMIT_MEAN','CC_DRAWINGS_TO_CREDIT_LIMIT_MAX','CC_DRAWINGS_TO_CREDIT_LIMIT_MIN',
#'FLAG_LESS_CREDIT_UTLIZATION',
#'CC_AMT_BALANCE_MIN','CC_AMT_BALANCE_MAX','CC_AMT_BALANCE_MEAN',
#'CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN','CC_AMT_CREDIT_LIMIT_ACTUAL_MAX','CC_AMT_CREDIT_LIMIT_ACTUAL_MIN',
#'CC_FLAG_ACTIVE_SUM',
'PREV_FLAG_LATEST_ACC_SUM',
'PREV_INSTAL_FLAG_MISSED_PAYMENT_SUM_SUM','PREV_INSTAL_FLAG_MISSED_PAYMENT_SUM_MEAN','PREV_INSTAL_FLAG_MISSED_PAYMENT_SUM_MIN','PREV_INSTAL_FLAG_MISSED_PAYMENT_SUM_MAX',
'TOTAL_KHARAB_PAYMENTS_TO_TOTAL_PAYMENTS','TOTAL_KHARAB_PAYMENTS',
'FIRST_LATE_PAYMENT_TO_TOTAL_LATE_PAYMENTS',
'FIRST_MISSED_PAYMENT_TO_TOTAL_LATE_PAYMENTS',
'PERC_LATE_PAYMENTS_TOTAL',
'PERC_MISSED_PAYMENTS_TOTAL',
'PREV_FLAG_FIRST_LATE_PAYMENT_SUM','PREV_FLAG_FIRST_MISSED_PAYMENT_SUM'
] #DO THIS FEATURE
print(len(col_phele))
print(len(set(col_phele)))
#df[col_phele]

num_rows=None
df =application_train_test(num_rows)
prev=prev_ins(num_rows)
print(df.shape)
df=df.merge(right=prev,how='left',on='SK_ID_CURR')
del prev
gc.collect()


df['CREDIT_TO_APPROVED_MAX_CREDIT']=df['AMT_CREDIT']/(df['APPROVED_MAX_CREDIT']+1)
df=df[col_phele]


a,b=ins(num_rows,from_df=True)
df=df.merge(right=a[['SK_ID_CURR','DAYS_INSTALMENT','PAYMENT_DIFF']].groupby('SK_ID_CURR').max().reset_index().rename(columns={'DAYS_INSTALMENT':'DAYS_LAST_LATE_PAYMENT','PAYMENT_DIFF':'MAX_PAYMENT_DIFF'}),how='left',on='SK_ID_CURR')
df.shape
a["LATEST_LATE_PAYMENT"]=a[a.groupby('SK_ID_CURR')['DAYS_INSTALMENT'].transform(max)==a['DAYS_INSTALMENT']]['AMT_PAYMENT']
c=a[np.isfinite(a['LATEST_LATE_PAYMENT'])][['SK_ID_CURR','LATEST_LATE_PAYMENT']]
c=c.groupby('SK_ID_CURR')['LATEST_LATE_PAYMENT'].max()
df=df.merge(right=c.reset_index(),how='left',on='SK_ID_CURR')
print(df.shape)
del c
gc.collect()
a['PAYMENT_DIFF_LATEST_PAYMENT']=a[a.groupby('SK_ID_CURR')['DAYS_INSTALMENT'].transform(max)==a['DAYS_INSTALMENT']]['PAYMENT_DIFF']
c=a[np.isfinite(a['PAYMENT_DIFF_LATEST_PAYMENT'])][['SK_ID_CURR','PAYMENT_DIFF_LATEST_PAYMENT']]
c=c.groupby('SK_ID_CURR')['PAYMENT_DIFF_LATEST_PAYMENT'].max()
df=df.merge(right=c.reset_index(),how='left',on='SK_ID_CURR')
print(df.shape)
del c
gc.collect()
a["DAYS_SEVERE_PAYMENT"]=a[a.groupby('SK_ID_CURR')['PAYMENT_DIFF'].transform(max)==a['PAYMENT_DIFF']]['DAYS_INSTALMENT']
c=a[np.isfinite(a['DAYS_SEVERE_PAYMENT'])][['SK_ID_CURR','DAYS_SEVERE_PAYMENT']]
c=c.groupby('SK_ID_CURR')['DAYS_SEVERE_PAYMENT'].max()
df=df.merge(right=c.reset_index(),how='left',on='SK_ID_CURR')
print(df.shape)
del c
gc.collect()

del a
gc.collect()

df=df.merge(right=b[['SK_ID_CURR','DAYS_INSTALMENT','PAYMENT_DIFF']].groupby('SK_ID_CURR').max().reset_index().rename(columns={'DAYS_INSTALMENT':'DAYS_LAST_MISSED_PAYMENT','PAYMENT_DIFF':'MISSED_MAX_PAYMENT_DIFF'}),how='left',on='SK_ID_CURR')
b["MISSED_DAYS_SEVERE_PAYMENT"]=b[b.groupby('SK_ID_CURR')['PAYMENT_DIFF'].transform(max)==b['PAYMENT_DIFF']]['DAYS_INSTALMENT']
c=b[np.isfinite(b['MISSED_DAYS_SEVERE_PAYMENT'])][['SK_ID_CURR','MISSED_DAYS_SEVERE_PAYMENT']]
c=c.groupby('SK_ID_CURR')['MISSED_DAYS_SEVERE_PAYMENT'].max()
df=df.merge(right=c.reset_index(),how='left',on='SK_ID_CURR')
print(df.shape)
del c,b
gc.collect()

cred=credit_card_balance(num_rows)
cred=cred[['AMT_BALANCE_LAST_CREDIT',
'CC_NAME_CONTRACT_STATUS_NUNIQUE',
'CC_FLAG_AMT_BALANCE_POS_SUM',
'CC_CNT_DRAWINGS_ATM_CURRENT_MEAN',
'CC_PAYMENT_COUNT',
'CC_DRAWINGS_TO_CREDIT_MEAN','CC_DRAWINGS_TO_CREDIT_MIN','CC_DRAWINGS_TO_CREDIT_MAX','CC_DRAWINGS_TO_CREDIT_SUM',
'CC_SK_ID_PREV_NUNIQUE',
'CC_MONTHS_BALANCE_SIZE','CC_MONTHS_BALANCE_MIN','CC_MONTHS_BALANCE_MAX',
'CC_DRAWINGS_TO_CREDIT_LIMIT_MEAN','CC_DRAWINGS_TO_CREDIT_LIMIT_MAX','CC_DRAWINGS_TO_CREDIT_LIMIT_MIN',
'FLAG_LESS_CREDIT_UTLIZATION',
'CC_AMT_BALANCE_MIN','CC_AMT_BALANCE_MAX','CC_AMT_BALANCE_MEAN',
'CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN','CC_AMT_CREDIT_LIMIT_ACTUAL_MAX','CC_AMT_CREDIT_LIMIT_ACTUAL_MIN',
'CC_FLAG_ACTIVE_SUM']]
df=df.merge(right=cred.reset_index(),how='left',on='SK_ID_CURR')
del cred
gc.collect()

#df=df[col_phele]
#pos=pos_cash(num_rows)
#df=df.merge(right=pos,how='left',on='SK_ID_CURR')
#del pos
#gc.collect()

buro=bureau_and_balance(num_rows)
buro_cols=['SK_ID_CURR',
 'BURO_AMT_CREDIT_TO_LIMIT_MAX','BURO_AMT_CREDIT_TO_LIMIT_MIN','BURO_AMT_CREDIT_TO_LIMIT_MEAN',
 'BURO_FLAG_CONSUMER_LOAN_SUM',
 'BURO_FLAG_CREDIT_CARD_LOAN_SUM',
 'BURO_CREDIT_TO_OVERDUE_MEAN',
 'BURO_CREDIT_TO_ANNUITY_MEAN',
 'BURO_CREDIT_TO_ANNUITY_MIN',
 'BURO_CREDIT_TO_ANNUITY_MAX',
 'BURO_BURO_AMT_ANNUITY_TO_AMT_CREDIT_SUM_DEBT_MIN',
 'BURO_BURO_AMT_ANNUITY_TO_AMT_CREDIT_SUM_DEBT_MAX',
 'BURO_BURO_AMT_ANNUITY_TO_AMT_CREDIT_SUM_DEBT_MEAN',
 'BURO_DAYS_CREDIT_MIN',
 'BURO_DAYS_CREDIT_MAX',
 'BURO_DAYS_CREDIT_MEAN',
 'BURO_DAYS_CREDIT_VAR',
 'BURO_CREDIT_DAY_OVERDUE_MAX',
 'BURO_CREDIT_DAY_OVERDUE_MEAN',
 'BURO_DAYS_CREDIT_ENDDATE_MIN',
 'BURO_DAYS_CREDIT_ENDDATE_MAX',
 'BURO_DAYS_CREDIT_ENDDATE_MEAN',
 'BURO_DAYS_ENDDATE_FACT_MIN',
 'BURO_DAYS_ENDDATE_FACT_MAX',
 'BURO_DAYS_ENDDATE_FACT_MEAN',
 'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN',
 'BURO_AMT_CREDIT_SUM_MAX',
 'BURO_AMT_CREDIT_SUM_MEAN',
 'BURO_AMT_CREDIT_SUM_SUM',
 'BURO_AMT_CREDIT_SUM_DEBT_MAX',
 'BURO_AMT_CREDIT_SUM_DEBT_MEAN',
 'BURO_AMT_CREDIT_SUM_OVERDUE_MEAN',
 'BURO_AMT_CREDIT_SUM_OVERDUE_SUM',
 'BURO_AMT_CREDIT_SUM_LIMIT_MEAN',
 'BURO_AMT_CREDIT_SUM_LIMIT_MAX',
 'BURO_AMT_CREDIT_SUM_LIMIT_MIN',
 'BURO_DAYS_CREDIT_UPDATE_MIN',
 'BURO_DAYS_CREDIT_UPDATE_MAX',
 'BURO_DAYS_CREDIT_UPDATE_MEAN',
 'BURO_AMT_ANNUITY_MAX',
 'BURO_MONTHS_BALANCE_MAX_MAX',
 'BURO_MONTHS_BALANCE_SIZE_MEAN',
 'BURO_MONTHS_BALANCE_SIZE_SUM',
 'BURO_FLAG_ACTIVE_SUM',
 'BURO_CREDIT_TYPE_NUNIQUE',
 'BURO_FLAG_NO_DPD_SUM_SUM',
 'BURO_FLAG_UNKNOWN_SUM_SUM',
 'BURO_FLAG_MAXIMAL_SUM_SUM',
 'BURO_FLAG_DPD_120+_SUM_SUM',
 'BURO_FLAG_DPD_31_60_SUM_SUM',
 'CREDIT_CARD_CREDIT_TO_ANNUITY_MEAN',
 'CREDIT_CARD_CREDIT_TO_ANNUITY_MIN',
 'CREDIT_CARD_CREDIT_TO_ANNUITY_MAX',
 'CREDIT_CARD_BURO_AMT_ANNUITY_TO_AMT_CREDIT_SUM_DEBT_MIN',
 'CREDIT_CARD_BURO_AMT_ANNUITY_TO_AMT_CREDIT_SUM_DEBT_MAX',
 'CREDIT_CARD_DAYS_CREDIT_MIN',
 'CREDIT_CARD_DAYS_CREDIT_MAX',
 'CREDIT_CARD_DAYS_CREDIT_MEAN',
 'CREDIT_CARD_DAYS_CREDIT_VAR',
 'CREDIT_CARD_DAYS_CREDIT_ENDDATE_MIN',
 'CREDIT_CARD_DAYS_CREDIT_ENDDATE_MEAN',
 'CREDIT_CARD_DAYS_ENDDATE_FACT_MIN',
 'CREDIT_CARD_DAYS_ENDDATE_FACT_MAX',
 'CREDIT_CARD_AMT_CREDIT_MAX_OVERDUE_MEAN',
 'CREDIT_CARD_CNT_CREDIT_PROLONG_SUM',
 'CREDIT_CARD_AMT_CREDIT_SUM_MAX',
 'CREDIT_CARD_AMT_CREDIT_SUM_MEAN',
 'CREDIT_CARD_AMT_CREDIT_SUM_DEBT_MAX',
 'CREDIT_CARD_AMT_CREDIT_SUM_DEBT_MEAN',
 'CREDIT_CARD_AMT_CREDIT_SUM_LIMIT_MEAN',
 'CREDIT_CARD_AMT_CREDIT_SUM_LIMIT_MIN',
 'CREDIT_CARD_DAYS_CREDIT_UPDATE_MIN',
 'CREDIT_CARD_DAYS_CREDIT_UPDATE_MAX',
 'CREDIT_CARD_DAYS_CREDIT_UPDATE_MEAN',
 'CREDIT_CARD_MONTHS_BALANCE_MAX_MAX',
 'CREDIT_CARD_MONTHS_BALANCE_SIZE_MEAN',
 'CREDIT_CARD_MONTHS_BALANCE_SIZE_SUM',
 'CREDIT_CARD_FLAG_ACTIVE_SUM',
 'CREDIT_CARD_FLAG_CLOSED_SUM',
 'CREDIT_CARD_FLAG_NO_DPD_SUM_SUM',
 'CREDIT_CARD_FLAG_UNKNOWN_SUM_SUM',
 'CREDIT_CARD_FLAG_MAXIMAL_SUM_SUM',
 'CONSUMER_CREDIT_TO_OVERDUE_MEAN',
 'CONSUMER_CREDIT_TO_ANNUITY_MEAN',
 'CONSUMER_CREDIT_TO_ANNUITY_MIN',
 'CONSUMER_CREDIT_TO_ANNUITY_MAX',
 'CONSUMER_CREDIT_TO_ANNUITY_SUM',
 'CONSUMER_BURO_AMT_ANNUITY_TO_AMT_CREDIT_SUM_DEBT_MIN',
 'CONSUMER_BURO_AMT_ANNUITY_TO_AMT_CREDIT_SUM_DEBT_MEAN',
 'CONSUMER_DAYS_CREDIT_MAX',
 'CONSUMER_DAYS_CREDIT_VAR',
 'CONSUMER_DAYS_CREDIT_ENDDATE_MIN',
 'CONSUMER_DAYS_CREDIT_ENDDATE_MAX',
 'CONSUMER_DAYS_CREDIT_ENDDATE_MEAN',
 'CONSUMER_CNT_CREDIT_PROLONG_SUM',
 'CONSUMER_AMT_CREDIT_SUM_MAX',
 'CONSUMER_AMT_CREDIT_SUM_MEAN',
 'CONSUMER_AMT_CREDIT_SUM_SUM',
 'CONSUMER_AMT_CREDIT_SUM_DEBT_MAX',
 'CONSUMER_AMT_CREDIT_SUM_DEBT_MEAN',
 'CONSUMER_AMT_CREDIT_SUM_OVERDUE_MEAN',
 'CONSUMER_AMT_CREDIT_SUM_OVERDUE_SUM',
 'CONSUMER_AMT_CREDIT_SUM_LIMIT_MEAN',
 'CONSUMER_AMT_CREDIT_SUM_LIMIT_MAX',
 'CONSUMER_DAYS_CREDIT_UPDATE_MIN',
 'CONSUMER_DAYS_CREDIT_UPDATE_MAX',
 'CONSUMER_DAYS_CREDIT_UPDATE_MEAN',
 'CONSUMER_FLAG_ACTIVE_SUM',
 'CONSUMER_FLAG_DPD_61_90_SUM_SUM',
 'CAR_CREDIT_TO_ANNUITY_MEAN',
 'CAR_CREDIT_TO_ANNUITY_MIN',
 'CAR_BURO_AMT_ANNUITY_TO_AMT_CREDIT_SUM_DEBT_MIN',
 'CAR_DAYS_CREDIT_MIN',
 'CAR_DAYS_CREDIT_MAX',
 'CAR_DAYS_CREDIT_ENDDATE_MIN',
 'CAR_DAYS_CREDIT_ENDDATE_MAX',
 'CAR_DAYS_ENDDATE_FACT_MIN',
 'CAR_DAYS_ENDDATE_FACT_MAX',
 'CAR_AMT_CREDIT_SUM_MAX',
 'CAR_AMT_CREDIT_SUM_SUM',
 'CAR_AMT_CREDIT_SUM_DEBT_MAX',
 'CAR_AMT_CREDIT_SUM_LIMIT_MEAN',
 'CAR_DAYS_CREDIT_UPDATE_MIN',
 'CAR_DAYS_CREDIT_UPDATE_MAX',
 'CAR_MONTHS_BALANCE_SIZE_MEAN',
 'CAR_MONTHS_BALANCE_SIZE_SUM',
 'MORTGAGE_CREDIT_TO_ANNUITY_SUM',
 'MORTGAGE_DAYS_CREDIT_MIN',
 'MORTGAGE_DAYS_CREDIT_ENDDATE_MIN',
 'MORTGAGE_DAYS_CREDIT_UPDATE_MIN',
 'MORTGAGE_MONTHS_BALANCE_SIZE_SUM',
 'MICROLOAN_CREDIT_TO_ANNUITY_SUM',
 'MICROLOAN_BURO_AMT_ANNUITY_TO_AMT_CREDIT_SUM_DEBT_MIN',
 'MICROLOAN_BURO_AMT_ANNUITY_TO_AMT_CREDIT_SUM_DEBT_MEAN',
 'MICROLOAN_DAYS_CREDIT_MIN',
 'MICROLOAN_DAYS_CREDIT_MAX',
 'MICROLOAN_DAYS_CREDIT_ENDDATE_MIN',
 'MICROLOAN_DAYS_CREDIT_ENDDATE_MAX',
 'MICROLOAN_DAYS_ENDDATE_FACT_MAX',
 'MICROLOAN_AMT_CREDIT_SUM_MAX',
 'MICROLOAN_AMT_CREDIT_SUM_SUM',
 'MICROLOAN_AMT_CREDIT_SUM_DEBT_MAX',
 'MICROLOAN_DAYS_CREDIT_UPDATE_MIN',
 'MICROLOAN_DAYS_CREDIT_UPDATE_MAX',
 'MICROLOAN_AMT_ANNUITY_MEAN',
 'MICROLOAN_MONTHS_BALANCE_SIZE_SUM',
 'ACTIVE_TO_TOTAL',
 'AMT_CREDIT_TO_DEBT',
 'ACT_CREDIT_TO_ANNUITY_MEAN',
 'ACT_CREDIT_TO_ANNUITY_MIN',
 'ACT_CREDIT_TO_ANNUITY_MAX',
 'ACT_BURO_AMT_ANNUITY_TO_AMT_CREDIT_SUM_DEBT_MIN',
 'ACT_BURO_AMT_ANNUITY_TO_AMT_CREDIT_SUM_DEBT_MAX',
 'ACT_DAYS_CREDIT_MIN',
 'ACT_DAYS_CREDIT_MAX',
 'ACT_DAYS_CREDIT_MEAN',
 'ACT_DAYS_CREDIT_VAR',
 'ACT_CREDIT_DAY_OVERDUE_MEAN',
 'ACT_DAYS_CREDIT_ENDDATE_MIN',
 'ACT_DAYS_CREDIT_ENDDATE_MEAN',
 'ACT_AMT_CREDIT_MAX_OVERDUE_MEAN',
 'ACT_AMT_CREDIT_MAX_OVERDUE_MAX',
 'ACT_AMT_CREDIT_SUM_MAX',
 'ACT_AMT_CREDIT_SUM_MEAN',
 'ACT_AMT_CREDIT_SUM_DEBT_MEAN',
 'ACT_AMT_CREDIT_SUM_OVERDUE_MEAN',
 'ACT_AMT_CREDIT_SUM_LIMIT_MEAN',
 'ACT_AMT_CREDIT_SUM_LIMIT_MIN',
 'ACT_DAYS_CREDIT_UPDATE_MIN',
 'ACT_DAYS_CREDIT_UPDATE_MAX',
 'ACT_DAYS_CREDIT_UPDATE_MEAN',
 'ACT_AMT_ANNUITY_MAX',
 'ACT_MONTHS_BALANCE_MAX_MAX',
 'ACT_MONTHS_BALANCE_SIZE_SUM',
 'ACT_CREDIT_TYPE_NUNIQUE',
 'ACT_FLAG_NO_DPD_SUM_SUM',
 'ACT_FLAG_UNKNOWN_SUM_SUM',
 'ACT_FLAG_MAXIMAL_SUM_SUM',
 'CLS_CREDIT_TO_ANNUITY_MEAN',
 'CLS_CREDIT_TO_ANNUITY_MIN',
 'CLS_CREDIT_TO_ANNUITY_MAX',
 'CLS_BURO_AMT_ANNUITY_TO_AMT_CREDIT_SUM_DEBT_MIN',
 'CLS_DAYS_CREDIT_MAX',
 'CLS_DAYS_CREDIT_MEAN',
 'CLS_DAYS_CREDIT_VAR',
 'CLS_DAYS_CREDIT_ENDDATE_MIN',
 'CLS_DAYS_CREDIT_ENDDATE_MAX',
 'CLS_DAYS_CREDIT_ENDDATE_MEAN',
 'CLS_CNT_CREDIT_PROLONG_SUM',
 'CLS_AMT_CREDIT_SUM_MAX',
 'CLS_AMT_CREDIT_SUM_MEAN',
 'CLS_AMT_CREDIT_SUM_SUM',
 'CLS_AMT_CREDIT_SUM_DEBT_MAX',
 'CLS_AMT_CREDIT_SUM_DEBT_MEAN',
 'CLS_AMT_CREDIT_SUM_LIMIT_MEAN',
 'CLS_AMT_CREDIT_SUM_LIMIT_MAX',
 'CLS_DAYS_CREDIT_UPDATE_MAX',
 'CLS_DAYS_CREDIT_UPDATE_MEAN',
 'CLS_MONTHS_BALANCE_MAX_MAX',
 'CLS_CREDIT_TYPE_NUNIQUE',
 'BURO_AMT_CREDIT_SUM_DEBT_SUM']
buro=buro[buro_cols]
df=df.merge(right=buro,how='left',on='SK_ID_CURR')
del buro
gc.collect()

pos=pos_cash(num_rows)
df=df.merge(right=pos.reset_index(),how='left',on='SK_ID_CURR')
del pos
gc.collect()

df['TOTAL_DEBT']=df['AMT_BALANCE_LAST_CREDIT']+df['BURO_AMT_CREDIT_SUM_DEBT_SUM']
df['INCOME_TO_DEBT']=df['AMT_INCOME_TOTAL']/(df['TOTAL_DEBT']+1)
df['INCOME_TO_CREDIT_UTILIZATION']=df['AMT_INCOME_TOTAL']/(df['CC_DRAWINGS_TO_CREDIT_LIMIT_MEAN']+1)

feat_importance= kfold_lightgbm(df, num_folds= 5, stratified= True, debug= False)
