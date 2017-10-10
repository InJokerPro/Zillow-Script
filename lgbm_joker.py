import pandas as pd 
import numpy as np 
import lightgbm as lgb 
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder
import gc


print('Loading data...')

train_2016 = pd.read_csv('../input/train_2016_v2.csv',low_memory=False)
train_2017 = pd.read_csv('../input/train_2017.csv',low_memory=False)
df_train = pd.concat([train_2016,train_2017])
del train_2016,train_2017
gc.collect()

props_2016 = pd.read_csv('../input/properties_2016.csv',low_memory=False)
props_2017 = pd.read_csv('../input/properties_2017.csv',low_memory=False)
df_props = pd.concat([props_2016,props_2017])
del props_2017;gc.collect()

df_test = pd.read_csv('../input/sample_submission.csv',usecols=['ParcelId'])
df_test = pd.merge(df_test,props_2016.rename(columns={'parcelid' : 'ParcelId'}),
					how='left',on='ParcelId')

del props_2016;gc.collect()

df_train = pd.merge(left=df_train,right=df_props,on='parcelid',how='left')


sample_submission = pd.read_csv('../input/sample_submission.csv')


print('Feature engineering...')

df_train['month'] = (pd.to_datetime(df_train['transactiondate']).dt.year - 2016)*12 + pd.to_datetime(df_train['transactiondate']).dt.month
df_train.drop(['transactiondate'], axis = 1,inplace=True)

object_columns = df_train.dtypes[df_train.dtypes == 'object'].index.values

for column in df_test.columns:
    if df_test[column].dtype == int:
        df_test[column] = df_test[column].astype(np.int32)
    if df_test[column].dtype == float:
        df_test[column] = df_test[column].astype(np.float32)
      

for column in object_columns:
    train_test = pd.concat([df_train[column], df_test[column]], axis = 0)
    encoder = LabelEncoder().fit(train_test.astype(str))
    df_train[column] = encoder.transform(df_train[column].astype(str)).astype(np.int32)
    df_test[column] = encoder.transform(df_test[column].astype(str)).astype(np.int32)


#Get rid of ourliers    
#df_train = df_train.loc[np.abs(df_train['logerror'])<0.4]
#df_train.reset_index(inplace=True)
sp = ShuffleSplit(n_splits=1)

print('Training with LGBM...')

for train_index, val_index in sp.split(df_train) : 
	d_train = lgb.Dataset(df_train.drop(['logerror','parcelid'],axis=1).loc[train_index,:],label=df_train['logerror'].loc[train_index])
	d_val = lgb.Dataset(df_train.drop(['logerror','parcelid'],axis=1).loc[val_index,:],label=df_train['logerror'].loc[val_index])

	param = {'objective' : 'regression',
			'boosting_type' : 'gbdt',
			'learning_rate' : 0.01,
			'metric' : 'mae',
			'num_leaves' : 32,
			'max_depth' : 100,
			'bagging_fraction' : 0.95,
			'feature_fraction' : 0.85,
			'verbosity' : 0,
			'num_boost_round': 3000,
			'early_stopping_round' : 50}


	bst = lgb.train(param,d_train,1000,valid_sets=[d_val])

	print('Predicting OCT 2016')
	df_test['month']=10
	sample_submission['201610'] = bst.predict(df_test)

	print('Predicting NOV 2016')
	df_test['month']=11
	sample_submission['201611'] = bst.predict(df_test)

	print('Predicting DEC 2016')
	df_test['month']=12
	sample_submission['201612'] = bst.predict(df_test)

	print('Predicting OCT 2017')
	df_test['month']=22
	sample_submission['201710'] = bst.predict(df_test)

	print('Predicting NOV 2017')
	df_test['month']=23
	sample_submission['201711'] = bst.predict(df_test)

	print('Predicting DEC 2017')
	df_test['month']=24
	sample_submission['201712'] = bst.predict(df_test)


	sample_submission.to_csv('submission.csv',index=False)

