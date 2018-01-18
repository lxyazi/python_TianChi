import time
import datetime
from dateutil.parser import parse
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import skew
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series,DataFrame
import matplotlib
from pyplotz.pyplotz import PyplotZ
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib.font_manager import FontProperties
from sklearn import preprocessing
import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm, grid_search, datasets
from sklearn import grid_search
from sklearn.cross_validation import KFold
import lightgbm as lgb
from dateutil.parser import parse

data_path = 'G:/ali/train.csv'

train_raw = pd.read_csv('G:/ali/train.csv',encoding='gb2312')
test_raw = pd.read_csv('G:/ali/test.csv',encoding='gb2312')

train_df1=train_raw[train_raw['血糖']>8]
train_df2=train_df.sample(500)
train_df1=pd.concat([train_df1,train_df3])

#训练集和测试集特征一起处理
def make_feat(train_raw,test_raw):
    train_id = train_raw.id.values.copy()
    test_id = test_raw.id.values.copy()
    data = pd.concat([train_raw,test_raw])
    
    data['性别'] = data['性别'].map({'男':1,'女':0})
    #data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2017-10-09')).dt.days
    data.drop(['乙肝表面抗原','乙肝表面抗体','乙肝e抗体','乙肝核心抗体','乙肝e抗原','嗜酸细胞%','单核细胞%'],axis=1,inplace=True)
    data.fillna(data.median(axis=0),inplace=True)
    train = data[data.id.isin(train_id)]
    
    test = data[data.id.isin(test_id)]
    train.drop(['id','体检日期'],axis=1,inplace=True)
    test.drop(['id','体检日期'],axis=1,inplace=True)
    return train,test

train,test = make_feat(train_raw,test_raw)
train1,test1 = make_feat(train_df1,test_raw)

predictors = [f for f in test.columns if f not in ['血糖']]


def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label,pred)*0.5
    return ('0.5mse',score,False)

print('开始训练...')

print('开始CV 5折训练...')
t0 = time.time()
#
train_preds = np.zeros(train.shape[0])
test_preds = np.zeros((test.shape[0], 5))

kf = KFold(len(train), n_folds = 5, shuffle=True, random_state=2180)

for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    X_train = train.iloc[train_index]
    X_validate = train.iloc[test_index]
    
    gbm = lgb.LGBMRegressor(boosting_type='gbdt', max_depth=6, learning_rate=0.0095, n_estimators=1500,
                        min_split_gain=0.0, min_child_weight=0.001, min_child_samples=16,
                        subsample=0.45, subsample_freq=1, colsample_bytree=1.0, reg_alpha=2, reg_lambda=2.8,seed=218)
    
    gbm.fit(X_train[predictors],  X_train['血糖'])

#     xgb1 = xgb.XGBRegressor(n_estimators=1400, learning_rate=0.0082, subsample=0.55, colsample_bytree=0.4, 
#                         max_depth=6, gamma=8, reg_alpha=2, reg_lambda=2.8, min_child_weight=0.001,num_round=80000,early_stopping_rounds=50,
#                        random_state=218)
#     xgb1.fit(X_train[predictors],  X_train['血糖'])
    
    #feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
    train_preds[test_index] += gbm.predict(X_validate[predictors])
    X_validate.['血糖']
    test_preds[:,i] = gbm.predict(test[predictors])

print('线下得分：    {}'.format(mean_squared_error(train['血糖'],train_preds)*0.5))
print('CV训练用时{}秒'.format(time.time() - t0))

submission = pd.DataFrame({'pred':test_preds.mean(axis=1)})
submission.to_csv(r'G:/跑分结果/{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),header=None,
                  index=False, float_format='%.4f')