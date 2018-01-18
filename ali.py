from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import pandas as pd
from pandas import Series,DataFrame
#numpy, matplotlib, seaborn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pyplotz.pyplotz import PyplotZ
import seaborn as sns
sns.set(style='whitegrid',font="simhei")
# %matplotlib inline     //TODO

train_df = pd.read_csv("G:/ali/train.csv",encoding='gb2312')
test_df    = pd.read_csv("G:/ali/test.csv",encoding='gb2312')

def Feature_processing(x):
    x.性别 = x.性别.apply(lambda s: 1 if s == '男' else 0) 
#def feature_select():
#train_df = titanic_df.drop([], axis=1)
#def value_fill():
#train_df["Embarked"] = train_df[].fillna()
#def one-hot():
#train_df.info()

Feature_processing(train_df)
Feature_processing(test_df)
kf = KFold(n_splits=2)
X_test=test_df.drop(['id','体检日期','乙肝表面抗原','乙肝表面抗体','乙肝e抗原','乙肝e抗体','乙肝核心抗体'], axis=1,inplace=True)

X_test=test_df.fillna(0)


for train_index, validate_index in kf.split(train_df):
    #print("TRAIN:", train_index, "TEST:", validate_index)
    #print(type(train_df.loc[train_index]))
    X_train, X_validate = train_df.loc[train_index], train_df.loc[validate_index]
#sns.factorplot('性别','血糖',data=train_df,size=4,aspect=3,encoding='gb2312')#平均值？
    Y_train=X_train["血糖"]
    X_train=X_train.drop(['id','体检日期','乙肝表面抗原','乙肝表面抗体','乙肝e抗原','乙肝e抗体','乙肝核心抗体','血糖'], axis=1)
    
    Y_validate=X_validate["血糖"]
    X_validate=X_validate.drop(['id','体检日期','乙肝表面抗原','乙肝表面抗体','乙肝e抗原','乙肝e抗体','乙肝核心抗体','血糖'], axis=1)
    
    X_train=X_train.fillna(0.0)
    X_validate=X_validate.fillna(0.0)
    
    from sklearn import linear_model
    model_LinearRegression = linear_model.LinearRegression()
    model_LinearRegression.fit(X_train, Y_train)
    print(model_LinearRegression.score(X_train, Y_train))
    print(model_LinearRegression.score(X_validate,Y_validate))
    print(mean_squared_error(Y_train, model_LinearRegression.predict(X_train)))
    print(mean_squared_error(Y_validate, model_LinearRegression.predict(X_validate)))
#     from sklearn import svm
#     model_SVR=svm.SVR()
#     model_SVR.fit(X_train, Y_train)
#     print(model_SVR.score(X_train, Y_train))
#     print(model_SVR.score(X_validate,Y_validate))
    
    from sklearn.ensemble import RandomForestRegressor
    model_RandomForestRegressor=RandomForestRegressor(n_estimators=200,max_features=10,min_samples_split=5)
    model_RandomForestRegressor.fit(X_train, Y_train)
    #print(model_RandomForestRegressor.score(X_train, Y_train))
    #print(model_RandomForestRegressor.score(X_validate,Y_validate))
    break
    

pd.set_option('float_format', '{:20,.3f}'.format)
Y_pred=model_LinearRegression.predict(X_test)
submission = pd.DataFrame({
        "血糖": Y_pred
    })
#submission.血糖=submission.血糖.apply(lambda s:("%.3f" %s) if s!=0 else 0)
#print(submission)
submission.to_csv('G:/ali21.csv', index=False,header=False)